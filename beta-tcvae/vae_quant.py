import os
import time
import math
from numbers import Number
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import visdom
from torch.autograd import Variable
from torch.utils.data import DataLoader

import lib.dist as dist
import lib.utils as utils
import lib.datasets as dset
from lib.flows import FactorialNormalizingFlow

from elbo_decomposition import elbo_decomposition
# these are used in an `eval('plot_vs_gt...')` call
from plot_latent_vs_true import plot_vs_gt_shapes, plot_vs_gt_faces  # noqa: F401; 

# from thesis_losses import k_factor_sim_loss_samples
from thesis_losses import k_factor_sim_losses_params
from thesis_augmentations import augmented_batch

from tqdm import tqdm 


class MLPEncoder(nn.Module):
    def __init__(self, output_dim):
        super(MLPEncoder, self).__init__()
        self.output_dim = output_dim

        self.fc1 = nn.Linear(4096, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, output_dim)

        self.conv_z = nn.Conv2d(64, output_dim, 4, 1, 0)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x.view(-1, 64 * 64)
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        h = self.fc3(h)
        z = h.view(x.size(0), self.output_dim)
        return z


class MLPDecoder(nn.Module):
    def __init__(self, input_dim):
        super(MLPDecoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1200),
            nn.Tanh(),
            nn.Linear(1200, 1200),
            nn.Tanh(),
            nn.Linear(1200, 1200),
            nn.Tanh(),
            nn.Linear(1200, 4096)
        )

    def forward(self, z):
        h = z.view(z.size(0), -1)
        h = self.net(h)
        mu_img = h.view(z.size(0), 1, 64, 64)
        return mu_img


class ConvEncoder(nn.Module):
    def __init__(self, output_dim):
        super(ConvEncoder, self).__init__()
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(1, 32, 4, 2, 1)  # 32 x 32
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1)  # 16 x 16
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1)  # 8 x 8
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 4, 2, 1)  # 4 x 4
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 512, 4)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv_z = nn.Conv2d(512, output_dim, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = x.view(-1, 1, 64, 64)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        h = self.act(self.bn5(self.conv5(h)))
        z = self.conv_z(h).view(x.size(0), self.output_dim)
        return z


class ConvDecoder(nn.Module):
    def __init__(self, input_dim):
        super(ConvDecoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(input_dim, 512, 1, 1, 0)  # 1 x 1
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.ConvTranspose2d(512, 64, 4, 1, 0)  # 4 x 4
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.ConvTranspose2d(64, 64, 4, 2, 1)  # 8 x 8
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64, 32, 4, 2, 1)  # 16 x 16
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.ConvTranspose2d(32, 32, 4, 2, 1)  # 32 x 32
        self.bn5 = nn.BatchNorm2d(32)
        self.conv_final = nn.ConvTranspose2d(32, 1, 4, 2, 1)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)

    def forward(self, z):
        h = z.view(z.size(0), z.size(1), 1, 1)
        h = self.act(self.bn1(self.conv1(h)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.act(self.bn3(self.conv3(h)))
        h = self.act(self.bn4(self.conv4(h)))
        h = self.act(self.bn5(self.conv5(h)))
        mu_img = self.conv_final(h)
        return mu_img


class VAE(nn.Module):
    def __init__(self, z_dim, use_cuda=False, prior_dist=dist.Normal(), q_dist=dist.Normal(),
                 include_mutinfo=True, tcvae=False, conv=False, mss=False):
        super(VAE, self).__init__()

        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.include_mutinfo = include_mutinfo
        self.tcvae = tcvae
        self.lamb = 0
        self.beta = 1
        self.mss = mss
        self.x_dist = dist.Bernoulli()

        # Model-specific
        # distribution family of p(z)
        self.prior_dist = prior_dist
        self.q_dist = q_dist
        # hyperparameters for prior p(z)
        self.register_buffer('prior_params', torch.zeros(self.z_dim, 2))

        # create the encoder and decoder networks
        if conv:
            self.encoder = ConvEncoder(z_dim * self.q_dist.nparams)
            self.decoder = ConvDecoder(z_dim)
        else:
            self.encoder = MLPEncoder(z_dim * self.q_dist.nparams)
            self.decoder = MLPDecoder(z_dim)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()

    # return prior parameters wrapped in a suitable Variable
    def _get_prior_params(self, batch_size=1):
        expanded_size = (batch_size,) + self.prior_params.size()
        prior_params = Variable(self.prior_params.expand(expanded_size))
        return prior_params

    # samples from the model p(x|z)p(z)
    def model_sample(self, batch_size=1):
        # sample from prior (value will be sampled by guide when computing the ELBO)
        prior_params = self._get_prior_params(batch_size)
        zs = self.prior_dist.sample(params=prior_params)
        # decode the latent code z
        x_params = self.decoder.forward(zs)
        return x_params

    # define the guide (i.e. variational distribution) q(z|x)
    def encode(self, x):
        x = x.view(x.size(0), 1, 64, 64)
        # use the encoder to get the parameters used to define q(z|x)
        z_params = self.encoder.forward(x).view(x.size(0), self.z_dim, self.q_dist.nparams)
        # sample the latent code z
        zs = self.q_dist.sample(params=z_params)
        return zs, z_params

    def decode(self, z):
        x_params = self.decoder.forward(z).view(z.size(0), 1, 64, 64)
        xs = self.x_dist.sample(params=x_params)
        return xs, x_params

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        zs, z_params = self.encode(x)
        xs, x_params = self.decode(zs)
        return xs, x_params, zs, z_params

    def _log_importance_weight_matrix(self, batch_size, dataset_size):
        N = dataset_size
        M = batch_size - 1
        strat_weight = (N - M) / (N * M)
        W = torch.Tensor(batch_size, batch_size).fill_(1 / M)
        W.view(-1)[::M+1] = 1 / N
        W.view(-1)[1::M+1] = strat_weight
        W[M-1, 0] = strat_weight
        return W.log()

    def get_objectives(self, x, dataset_size, num_sim_factors):
        """Returns the estimated training ELBO, estimated ELBO, and k-factor similiarity loss"""
        # log p(x|z) + log p(z) - log q(z|x)
        batch_size = x.size(0)
        x = x.view(batch_size, 1, 64, 64)
        prior_params = self._get_prior_params(batch_size)
        x_recon, x_params, zs, z_params = self.reconstruct_img(x)

        logpx = self.x_dist.log_density(x, params=x_params).view(batch_size, -1).sum(1)
        logpz = self.prior_dist.log_density(zs, params=prior_params).view(batch_size, -1).sum(1)
        logqz_condx = self.q_dist.log_density(zs, params=z_params).view(batch_size, -1).sum(1)

        elbo = logpx + logpz - logqz_condx

        if self.beta == 1 and self.include_mutinfo and self.lamb == 0:
            print("You're using beta = 1, which excludes TC!")
            return elbo, elbo.detach()

        # compute log q(z) ~= log 1/(NM) sum_m=1^M q(z|x_m) = - log(MN) + logsumexp_m(q(z|x_m))
        _logqz = self.q_dist.log_density(
            zs.view(batch_size, 1, self.z_dim),
            z_params.view(1, batch_size, self.z_dim, self.q_dist.nparams)
        )

        if not self.mss:
            # minibatch stratified sampling
            logqz_prodmarginals = (logsumexp(_logqz, dim=1, keepdim=False) - math.log(batch_size * dataset_size)).sum(1)
            logqz = (logsumexp(_logqz.sum(2), dim=1, keepdim=False) - math.log(batch_size * dataset_size))
        else:
            # minibatch weighted sampling
            logiw_matrix = Variable(self._log_importance_weight_matrix(batch_size, dataset_size).type_as(_logqz.data))
            logqz = logsumexp(logiw_matrix + _logqz.sum(2), dim=1, keepdim=False)
            logqz_prodmarginals = logsumexp(
                logiw_matrix.view(batch_size, batch_size, 1) + _logqz, dim=1, keepdim=False).sum(1)

        if not self.tcvae:
            if self.include_mutinfo:
                modified_elbo = logpx - self.beta * (
                    (logqz_condx - logpz) -
                    self.lamb * (logqz_prodmarginals - logpz)
                )
            else:
                modified_elbo = logpx - self.beta * (
                    (logqz - logqz_prodmarginals) +
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)
                )
        else:
            if self.include_mutinfo:
                modified_elbo = logpx - \
                    (logqz_condx - logqz) - \
                    self.beta * (logqz - logqz_prodmarginals) - \
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)
            else:
                modified_elbo = logpx - \
                    self.beta * (logqz - logqz_prodmarginals) - \
                    (1 - self.lamb) * (logqz_prodmarginals - logpz)

        # return modified_elbo, elbo.detach(), k_factor_sim_loss_samples(zs, num_sim_factors)
        return modified_elbo, elbo.detach(), \
        k_factor_sim_losses_params(z_params.select(-1,0), z_params.select(-1, 1), num_sim_factors)


def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)

# for loading and batching datasets
def setup_data_loaders(args, use_cuda=False):
    if args.dataset == 'shapes':
        train_set = dset.Shapes()
    elif args.dataset == 'faces':
        train_set = dset.Faces()
    else:
        raise ValueError('Unknown dataset ' + str(args.dataset))

    kwargs = {'num_workers': 4, 'pin_memory': use_cuda}
    train_loader = DataLoader(dataset=train_set,
        batch_size=args.batch_size, shuffle=True, 
        collate_fn=augmented_batch if \
              (hasattr(args, 'use_augment_dataloader') and args.use_augment_dataloader) else None,
        **kwargs)

    return train_loader


# win_samples = None
# win_test_reco = None
# win_latent_walk = None
win_train_elbo = None
win_k_sim, win_k_contrast = None, None

@torch.no_grad()
def display_samples(model, x, vis, env_name, curr_iter):

    # plot random samples
    sample_mu = model.model_sample(batch_size=100).sigmoid()
    images = sample_mu.view(-1, 1, 64, 64).data.cpu()
    # win_samples = vis.images(images, 10, 2, opts={'caption': 'samples'}, win=win_samples)
    vis.images(images, 10, 2, opts={'title': f'samples_iter_{curr_iter}'}, env=f"{env_name}_samples")

    # plot the reconstructed distribution for the first 50 test images
    test_imgs = x[:50, :]
    _, reco_imgs, zs, _ = model.reconstruct_img(test_imgs)
    reco_imgs = reco_imgs.sigmoid()

    # concat. the two sets of 50 images --> (2, 50, 64, 64)
    # imagine you have two rows, 50 columns. (Each entry is an image.)
    # A row represents either the set of input images or the set of reconstructed images. 

    # Now you transpose this table so you have 50 rows, two columns, 
    # where each new row consist of an image and its attempted reconstruction.
    # Therefore, an image and its reconstruction sit next to each other.
    # final shape: (50, 2, 64, 64)
    test_reco_imgs = torch.cat([
        test_imgs.view(1, -1, 64, 64), reco_imgs.view(1, -1, 64, 64)], 0).transpose(0, 1)

    # We will need to respect the 1 channel, so it's (100, 1, 64, 64)
    vis.images(
        test_reco_imgs.contiguous().view(-1, 1, 64, 64).data.cpu(), 10, 2,
        opts={'title': f'recon_iter_{curr_iter}'}, env=f'{env_name}_reconstructions')

    # plot latent walks (change one variable while all others stay the same)
    zs = zs[0:2]
    batch_size, z_dim = zs.size()
    xs = []

    delta = torch.linspace(-2, 2, 7).type_as(zs)

    for i in range(z_dim):
        # vec: 0.0's, shape (7, z_dim)
        vec = Variable(torch.zeros(z_dim)).view(1, z_dim).expand(7, z_dim).contiguous().type_as(zs)

        # set the z_dim value to the appropriate z_dim index
        vec[:, i] = 1
        # delta[:, None] broadcasts delta to (7, 1) shape
        # we now have a column with the current z_dim val. we are traversing
        vec = vec * delta[:, None]

        zs_delta = zs.clone().view(batch_size, 1, z_dim)
        zs_delta[:, :, i] = 0
        # e.g. shapes (2,1,10) and (1, 7, 10) --> (2, 7, 10)
        # 2 from chosen batch size, 7 from len. of traversal, 10 from num. z dims.
        zs_walk = zs_delta + vec[None]
        xs_walk = model.decoder.forward(zs_walk.view(-1, z_dim)).sigmoid()
        xs.append(xs_walk)

    xs = torch.cat(xs, 0).data.cpu()

    vis.images(xs, 7, 2, opts={'title': f'latent walk_{curr_iter}'}, env=f"{env_name}_traverse")


def plot_avg_elbos(iters, avg_elbos, vis, env_name):
    global win_train_elbo
    win_train_elbo = vis.line(X=torch.Tensor(iters), Y= torch.Tensor(avg_elbos), 
                              opts={'title': 'Running Avg. ELBO vs. iterations', 'markers': True}, win=win_train_elbo, 
                              update=None if win_train_elbo is None else 'append', env=f"{env_name}_lines")

def plot_k_factor_losses(iters, avg_kFactSim_losses, vis, env_name):
    # global win_k_sim, win_k_contrast
    global win_k_sim

    win_k_sim = vis.line(X=torch.Tensor(iters), Y=torch.Tensor(avg_kFactSim_losses),
                         opts={'title': 'Running Avg. K-similarity loss vs. iterations', 'markers': True}, win=win_k_sim,
                         update=None if win_k_sim is None else 'append', 
                         env=f"{env_name}_lines")

    # win_k_contrast = vis.line(X=torch.Tensor(iters), Y=torch.Tensor(avg_kFactContrast_losses),
    #                      opts={'title': 'Running Avg. K-contrastive loss vs. iterations', 'markers': True}, win=win_k_contrast,
    #                      update=None if win_k_sim is None else 'append', 
    #                      env=f"{env_name}_lines")



def anneal_kl(args, vae, iteration):
    if args.dataset == 'shapes':
        warmup_iter = 7000
    elif args.dataset == 'faces':
        warmup_iter = 2500

    if args.lambda_anneal:
        vae.lamb = max(0, 0.95 - 1 / warmup_iter * iteration)  # 1 --> 0
    else:
        vae.lamb = 0
    if args.beta_anneal:
        # vae.beta = min(args.beta, args.beta / warmup_iter * iteration)  # 0 --> 1
        vae.beta = min(args.beta, 1 / warmup_iter * iteration)
    else:
        vae.beta = args.beta


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-d', '--dataset', default='shapes', type=str, help='dataset name',
        choices=['shapes', 'faces'])
    parser.add_argument('-dist', default='normal', type=str, choices=['normal', 'laplace', 'flow'])
    parser.add_argument('-n', '--num-epochs', default=50, type=int, help='number of training epochs')
    parser.add_argument('-b', '--batch-size', default=2048, type=int, help='batch size')
    parser.add_argument('-l', '--learning-rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('-z', '--latent-dim', default=10, type=int, help='size of latent dimension')
    parser.add_argument('--beta', default=1, type=float, help='ELBO penalty term')
    parser.add_argument('--tcvae', action='store_true')
    parser.add_argument('--exclude-mutinfo', action='store_true')
    parser.add_argument('--beta-anneal', action='store_true')
    parser.add_argument('--lambda-anneal', action='store_true')
    
    # honors thesis concepts
    parser.add_argument('--use_augment_dataloader', action='store_true', help='whether to load images + their augmentations per batch')
    parser.add_argument('--num_sim_factors', type=int, default=None, help='k: for k-factor similarity loss')
    parser.add_argument('--augment_factor', type=int, default=None, help='weight of mean-squared err. of k-factor similarity loss')
    # end

    parser.add_argument('--mss', action='store_true', help='use the improved minibatch estimator')
    parser.add_argument('--conv', action='store_true')
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--visdom', action='store_true', help='whether plotting in visdom is desired')
    parser.add_argument('--visdom_port', type=int, default=4500, help='visdom port')

    parser.add_argument('--save', default='test1')
    parser.add_argument('--log_freq', default=200, type=int, help='num iterations per log and model checkpoint')
    parser.add_argument('--checkpt_fp', type=str, default = None, help="filepath of checkpoint to use")
    args = parser.parse_args()

    # confirm mutual existence of args.
    if args.use_augment_dataloader:
        print("Using augmentation dataloader")
        assert args.num_sim_factors and args.augment_factor

    # set up the saving directory
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    torch.cuda.set_device(args.gpu)

    if args.checkpt_fp:
        print("Loading your checkpoint...")
        # import must stay here to avoid circular import errors
        from metric_helpers.loader import load_model_and_dataset

        vae, _, _ = load_model_and_dataset(args.checkpt_fp)

    else:
        # setup the VAE
        if args.dist == 'normal':
            prior_dist = dist.Normal()
            q_dist = dist.Normal()
        elif args.dist == 'laplace':
            prior_dist = dist.Laplace()
            q_dist = dist.Laplace()
        elif args.dist == 'flow':
            prior_dist = FactorialNormalizingFlow(dim=args.latent_dim, nsteps=32)
            q_dist = dist.Normal()

        vae = VAE(z_dim=args.latent_dim, use_cuda=True, prior_dist=prior_dist, q_dist=q_dist,
            include_mutinfo=not args.exclude_mutinfo, tcvae=args.tcvae, conv=args.conv, mss=args.mss)
    

    if args.beta_anneal:
        print("Beta annealing will occur.")

    if args.lambda_anneal:
        print("Lambda annealing will occur.")

    augment_factor = args.augment_factor

    # setup the optimizer
    optimizer = optim.Adam(vae.parameters(), lr=args.learning_rate)


    # data loader
    train_loader = setup_data_loaders(args, use_cuda=True)

    # setup visdom for visualization
    if args.visdom:
        print('Visdom visualization enabled')
        if not os.path.exists("./vis_logs"):
            os.mkdir("./vis_logs")
        vis = visdom.Visdom(port=args.visdom_port, log_to_filename=f"./vis_logs/{args.save}")

    avg_elbos = []
    avg_k_sim_losses = []

    # training loop
    dataset_size = len(train_loader.dataset)
    num_iterations = len(train_loader) * args.num_epochs
    iteration = 0

    # initialize objective term accumulators
    # elbo_running_mean = utils.RunningAverageMeter()
    elbo_running_mean = utils.AverageMeter()
    kSimLoss_running_mean = utils.AverageMeter()


    # visualize training loop
    pbar = tqdm(total=num_iterations)

    # helper for "x" on plots
    logging_iterations = []

    while iteration < num_iterations:
        for i, x in enumerate(train_loader):
            pbar.update(1)
            iteration += 1

            batch_time = time.time()
            vae.train()

            # gradually adjust weights for total correlation and dim.-wise KL div.
            anneal_kl(args, vae, iteration)

            # reset gradient and update params. with new gradient
            optimizer.zero_grad()
            # transfer to GPU
            x = x.cuda()
            # wrap the mini-batch in a PyTorch Variable
            x = Variable(x)
            # do ELBO gradient and accumulate loss
            obj, elbo, k_sim_loss = vae.get_objectives(x, dataset_size, args.num_sim_factors)

            # numerical stability checks
            if utils.isnan(obj).any():
                raise ValueError('NaN spotted in objective.')
            if utils.isnan(elbo).any():
                raise ValueError('NaN spotted in elbo')
            
            # torch.Tensor indicates honors thesis concepts/args. were specified
            # Otherwise, it is returned as a mere 0
            if isinstance(k_sim_loss, torch.Tensor): 
                if utils.isnan(k_sim_loss).any():
                    raise ValueError('NaN spotted in k_sim_loss')
            
            # if honors thesis concepts/args. were not specified, simply just
            # proceeed normally, treating my weighted k_sim_loss as 0
            if augment_factor is None:
                augment_factor = 0

            # (obj.mean().mul(-1) + augment_factor * k_sim_loss).backward()
            obj.mean().mul(-1).backward()
            
            # Keep track of ongoing average of objective metrics
            elbo_running_mean.update(elbo.mean().item())

            if isinstance(k_sim_loss, torch.Tensor):
                kSimLoss_running_mean.update(k_sim_loss.item())

            # adjust params. accr. to gradient and optimizer (e.g. Adam)
            optimizer.step()

            # report training diagnostics
            if iteration % args.log_freq == 0:
                logging_iterations.append(iteration)

                avg_elbos.append(elbo_running_mean.avg)
                avg_k_sim_losses.append(kSimLoss_running_mean.avg)

                if kSimLoss_running_mean.val is None:
                    curr_k_sim_loss = 'None'
                else:
                    curr_k_sim_loss = '%.2f' % kSimLoss_running_mean.val

                pbar.write('[iteration %03d] time: %.2f \tbeta %.2f \tlambda %.2f training ELBO: %.4f (%.4f) \t\
                           k_sim_loss: %s (%.2f)' % (
                    iteration, time.time() - batch_time, vae.beta, vae.lamb,
                    elbo_running_mean.val, elbo_running_mean.avg,
                    curr_k_sim_loss, kSimLoss_running_mean.avg))

                vae.eval()

                # plot training and test ELBOs
                if args.visdom:
                    display_samples(vae, x, vis, args.save, iteration)
                    plot_avg_elbos(logging_iterations, avg_elbos, vis, args.save)

                    if args.use_augment_dataloader:
                        plot_k_factor_losses(logging_iterations, avg_k_sim_losses, vis, args.save)

                utils.save_checkpoint({
                    'state_dict': vae.state_dict(),
                    'args': args}, args.save, iteration)
                eval('plot_vs_gt_' + args.dataset)(vae, train_loader.dataset,
                    os.path.join(args.save, 'gt_vs_latent_{:05d}.png'.format(iteration)), pbar)

    # Report statistics after training
    vae.eval()
    utils.save_checkpoint({
        'state_dict': vae.state_dict(),
        'args': args}, args.save, 0)
    dataset_loader = DataLoader(train_loader.dataset, batch_size=1000, num_workers=1, shuffle=False)
    logpx, dependence, information, dimwise_kl, analytical_cond_kl, marginal_entropies, joint_entropy = \
        elbo_decomposition(vae, dataset_loader)
    torch.save({
        'logpx': logpx,
        'dependence': dependence,
        'information': information,
        'dimwise_kl': dimwise_kl,
        'analytical_cond_kl': analytical_cond_kl,
        'marginal_entropies': marginal_entropies,
        'joint_entropy': joint_entropy
    }, os.path.join(args.save, 'elbo_decomposition.pth'))
    eval('plot_vs_gt_' + args.dataset)(vae, dataset_loader.dataset, os.path.join(args.save, 'gt_vs_latent.png'), pbar)
    return vae


if __name__ == '__main__':
    model = main()
