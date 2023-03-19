"""solver.py"""

# import matplotlib.pyplot as plt
# from P_PID import PIDControl
from dataset import return_data
from model import BetaVAE_H, BetaVAE_B
from utils import cuda, grid2gif
from torchvision.utils import make_grid, save_image
import torch.distributions as dist
# from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import visdom
from tqdm import tqdm
import os
import torch
# torch.cuda.set_device(0)
import warnings
warnings.filterwarnings("ignore")


# from I_PID import PIDControl


def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(
            x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'gaussian':
        x_recon = torch.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
        recon_loss = None

    return recon_loss


# mutual info. and total correlations provided by ChatGPT
import torch.distributions as dist

def total_corr(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Computes the total correlation for VAEs, which encourages independence between latent variables.
    
    Args:
        mu (torch.Tensor): The mean tensor of the latent variables' Gaussian distribution.
        logvar (torch.Tensor): The log variance tensor of the latent variables' Gaussian distribution.
    
    Returns:
        torch.Tensor: The total correlation loss.
    """
    batch_size = mu.size(0)
    assert batch_size != 0

    # Create the q(z) and q(z_i) distributions
    q_z = dist.Normal(mu, logvar.mul(0.5).exp())
    q_z_product = dist.Normal(torch.zeros_like(mu), torch.ones_like(logvar))

    # Calculate the KL divergence between q(z) and q(z_i)
    tc = dist.kl_divergence(q_z, q_z_product).sum(dim=-1).mean()

    return tc

def mutual_information(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Computes the mutual information between input data and latent variables for VAEs.
    
    Args:
        mu (torch.Tensor): The mean tensor of the latent variables' Gaussian distribution.
        logvar (torch.Tensor): The log variance tensor of the latent variables' Gaussian distribution.
    
    Returns:
        torch.Tensor: The mutual information between input data and latent variables.
    """
    batch_size = mu.size(0)
    assert batch_size != 0

    # Create the q(z) and q(z_i) distributions
    q_z = dist.Normal(mu, logvar.mul(0.5).exp())
    q_z_product = dist.Normal(torch.zeros_like(mu), torch.ones_like(logvar))

    # Sample from q(z) using the reparameterization trick
    epsilon = torch.randn_like(mu)  # Random noise
    z = mu + epsilon * torch.exp(0.5 * logvar)

    # Compute the joint entropy of input data and latent variables
    joint_entropy = -q_z.log_prob(z).sum(dim=-1)

    # Compute the marginal entropy of the latent variables
    marginal_entropies = -q_z_product.log_prob(epsilon).sum(dim=-1)

    # Calculate the mutual information
    mi = (joint_entropy - marginal_entropies).mean()

    return mi



def kl_divergence(mu, logvar):
    """
    Calculates the Kullback-Leibler (KL) divergence between two Gaussian distributions,
    one with mean `mu` and variance `inverse_log(logvar)`, and one with mean `0` and variance `1`. (by ChatGPT)

    Parameters
    ----------
    mu: shape (batch_size, z_dim)
        Tensor of means for each dimension of the Gaussian distribution.
    logvar: shape (batch_size, z_dim)
        Tensor of log variances for each dimension of the Gaussian distribution.

    Returns
    -------
    total_kld: (1,)-shape Tensor
        summed across latent dimensions, then averaged by minibatch size
    dimension_wise_kld: (z_dim,)-shape Tensor
        the KL divergence listed for each latent dimension
    mean_kld: (1,)-shape Tensor
        averaged across minibatch size and latent dimensions 
    """
    batch_size = mu.size(0)
    assert batch_size != 0

    """
    klds is approximation of KL divergence, found in original paper
    https://arxiv.org/pdf/1312.6114.pdf appendix B in "Example: Variational Auto-Encoder"

    It makes the approximation that the posterior distribution of a VAE involves
    a diagonal covariance matrix, where the variance of each dimension is independent
    of the others (info. from ChatGPT)
    """

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())

    # .sum(axis_num_1).mean(axis_num_2, keep_dim = True or False)
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


# gathers info. about model during training loop
class DataGather(object):
    def __init__(self):
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):
        return dict(iter=[],
                    recon_loss=[],
                    total_kld=[],
                    dim_wise_kld=[],
                    mean_kld=[],
                    mu=[],
                    var=[],
                    images=[], beta=[])

    def insert(self, **kwargs):
        for key in kwargs:
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()


class Solver(object):
    def __init__(self, args):
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.max_iter = args.max_iter
        self.global_iter = 0

        self.z_dim = args.z_dim
        self.beta = args.beta
        self.gamma = args.gamma
        self.C_max = args.C_max
        self.C_max_org = args.C_max
        self.C_stop_iter = args.C_stop_iter
        self.objective = args.objective
        self.model = args.model
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        # self.KL_loss = args.KL_loss
        # self.pid_fixed = args.pid_fixed
        # self.is_PID = args.is_PID
        self.step_value = args.step_val
        self.C_start = args.C_start

        if args.dataset.lower() == 'dsprites':
            self.nc = 1
            self.decoder_dist = 'bernoulli'
        elif args.dataset.lower() == '3dchairs':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        elif args.dataset.lower() == 'celeba':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        else:
            raise NotImplementedError

        if args.model == 'H':
            net = BetaVAE_H
        elif args.model == 'B':
            net = BetaVAE_B
        else:
            raise NotImplementedError('only support model H or B')

        self.net = cuda(net(self.z_dim, self.nc), self.use_cuda)
        self.optim = optim.Adam(self.net.parameters(), lr=self.lr,
                                betas=(self.beta1, self.beta2))

        self.viz_name = args.viz_name
        self.viz_port = args.viz_port
        self.viz_on = args.viz_on

        # `win` is short for "window"
        self.win_recon = None
        self.win_beta = None
        self.win_kld = None
        self.win_mu = None
        self.win_var = None

        if self.viz_on:
            self.viz = visdom.Visdom(port=self.viz_port)

        self.ckpt_dir = os.path.join(args.ckpt_dir, args.viz_name)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_name = args.ckpt_name
        if self.ckpt_name is not None:
            self.load_checkpoint(self.ckpt_name)

        self.save_output = args.save_output
        self.output_dir = os.path.join(args.output_dir, args.viz_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        self.gather_step = args.gather_step
        # self.display_step = args.display_step
        self.save_step = args.save_step

        self.dset_dir = args.dset_dir
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.data_loader = return_data(args)

        self.gather = DataGather()
        self.gather2 = DataGather()

    # nopep8

    def train(self):
        """Runs training loop and provides Visdom server visualizations of images and training stats. These stats.
        include loss quanitities and beta"""

        self.net_mode(train=True)
        self.C_max = cuda(torch.FloatTensor([self.C_max]), self.use_cuda)
        out = False

        pbar = tqdm(total=self.max_iter)
        pbar.update(self.global_iter)

        # write log to log file
        outfile = os.path.join(self.ckpt_dir, "train.log")
        kl_file = os.path.join(self.ckpt_dir, "train.kl")
        fw_log = open(outfile, "w")
        fw_kl = open(kl_file, "w")
        # fw_kl.write('total KL\tz_dim' + '\n')

        # init PID control
        # PID = PIDControl()

        # Kp = 0.01
        # Ki = -0.001
        # # Kd = 0.0
        # C = 0.5
        # period = 5000
        # fw_log.write("Kp:{0:.5f} Ki: {1:.6f} C_iter:{2:.1f} period:{3} step_val:{4:.4f}\n"
        #              .format(Kp, Ki, self.C_stop_iter, period, self.step_value))

        C = 0.5

        while not out:
            for x in self.data_loader:
                self.global_iter += 1
                pbar.update(1)

                """Feedforward and calculating quantities for loss"""

                x = cuda(x, self.use_cuda)
                x_recon, mu, logvar = self.net(x)
                recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

                # not focused on ControlVAE

                # if self.is_PID and self.objective == 'H':
                #     if self.global_iter % period == 0:
                #         C += self.step_value
                #     if C > self.C_max_org:
                #         C = self.C_max_org
                #     # dynamic pid
                #     self.beta, _ = PID.pid(C, total_kld.item(), Kp, Ki, Kd)

                """Calculating loss"""

                if self.objective == 'H':
                    beta_vae_loss = recon_loss + self.beta * total_kld

                elif self.objective == 'B':
                    # tricks for C
                    C = torch.clamp(self.C_max/self.C_stop_iter * self.global_iter, 
                                    self.C_start, self.C_max.data[0])

                    beta_vae_loss = recon_loss + self.gamma*(total_kld-C).abs()
                

                """Backpropagation"""

                # re-initialize gradients to zero
                self.optim.zero_grad()
                # compute the gradient (i.e. partial derivs. w.r.t to neural net parameters)
                """
                note: beta_vae_loss has x_recon, which is from a forward pass of net, which is
                why calling .backward() has access to all the neural net parameters
                """
                beta_vae_loss.backward()
                # update neural net params. based on gradient
                self.optim.step()

                """Store lots of training stats."""

                if self.viz_on and self.global_iter % self.gather_step == 0:
                    self.gather.insert(iter=self.global_iter,
                                       mu=mu.mean(0).data, 
                                       var=logvar.exp().mean(0).data,
                                       recon_loss=recon_loss.data, 
                                       total_kld=total_kld.data,
                                       dim_wise_kld=dim_wise_kld.data, mean_kld=mean_kld.data, 
                                       beta=self.beta)
                
                """Log lots of training stats."""

                if self.global_iter % 20 == 0:
                    # write log to file
                    if self.objective == 'B':
                        C = C.item()
                    fw_log.write('[{}] recon_loss:{:.3f} total_kld:{:.3f} exp_kld:{:.3f} beta:{:.4f}\n'.format(
                        self.global_iter, recon_loss.item(), total_kld.item(), C, self.beta))
                    # write KL to file
                    dim_kl = dim_wise_kld.data.cpu().numpy()
                    dim_kl = [str(k) for k in dim_kl]
                    fw_kl.write('total_kld:{0:.3f}\t'.format(total_kld.item()))
                    fw_kl.write('z_dim:' + ','.join(dim_kl) + '\n')

                    if self.global_iter % 500 == 0:
                        fw_log.flush()
                        fw_kl.flush()

                # visualization (like Tensorboard)
                if self.viz_on and self.global_iter % self.gather_step == 0:
                    # gather current mini-batch image input and output, then visualize
                    self.gather.insert(images=x.data)
                    self.gather.insert(images=F.sigmoid(x_recon).data)
                    self.viz_reconstruction()
                    # visualize training stats. over time
                    self.viz_lines()
                    # restart for the next set of gathering steps
                    self.gather.flush()

                if (self.viz_on or self.save_output) and self.global_iter % 20000 == 0:
                    self.viz_traverse()

                # regarding checkpoints
                if self.global_iter % self.save_step == 0:
                    self.save_checkpoint('last')
                    pbar.write(f'Saved checkpoint(iter:{self.global_iter})')

                if self.global_iter % 50000 == 0:
                    self.save_checkpoint(str(self.global_iter))

                # when to stop
                if self.global_iter >= self.max_iter:
                    out = True
                    break

        pbar.write("[Training Finished]")
        pbar.close()
        fw_log.close()


    def viz_reconstruction(self):
        """See, at a maximum, the first 100 image inputs and their respective
        reconstructions for a certain mini-batch."""

        self.net_mode(train=False)

        # Obtain mini-batch input and output
        x = self.gather.data['images'][0][:100]
        x = make_grid(x, normalize=True) # min-max rescaling
        x_recon = self.gather.data['images'][1][:100]
        x_recon = make_grid(x_recon, normalize=True)
        images = torch.stack([x, x_recon], dim=0).cpu()

        # display grid of images on Visdom server
        self.viz.images(images, env=self.viz_name+'_reconstruction',
                        opts=dict(title=str(self.global_iter)), nrow=10)

        if self.save_output:
            output_dir = os.path.join(self.output_dir, str(self.global_iter))
            os.makedirs(output_dir, exist_ok=True)
            save_image(tensor=images, fp=os.path.join(output_dir, 'recon.jpg'), 
                       pad_value=1)

        self.net_mode(train=True)


    def viz_lines(self):
        self.net_mode(train=False)

        """
        Diff. between `stack` and `cat`: 
        `stack` concatenates sequence of tensors along a **new** dimension.
        `cat` ||-------------------------------- in the **given** dimension.
        """

        recon_losses = torch.stack(self.gather.data['recon_loss']).cpu()
        betas = torch.Tensor(self.gather.data['beta'])
        dim_wise_klds = torch.stack(self.gather.data['dim_wise_kld'])
        total_klds = torch.stack(self.gather.data['total_kld'])
        klds = torch.cat([dim_wise_klds, total_klds], 1).cpu()
        iters = torch.Tensor(self.gather.data['iter'])

        # legend
        legend = []
        for j in range(self.z_dim):
            legend.append(f'z_{j}')
        # legend.append('mean')
        legend.append('total')

        # "win" is short for "window"
        self.win_recon = self.viz.line(
            X=iters,
            Y=recon_losses,
            env=self.viz_name+'_lines',
            win=self.win_recon,
            update=None if self.win_recon is None else 'append',
            opts=dict(
                width=400,
                height=400,
                xlabel='iteration',
                title='reconsturction loss')
            )

        self.win_beta = self.viz.line(
            X=iters,
            Y=betas,
            env=self.viz_name+'_lines',
            win=self.win_beta,
            update=None if self.win_beta is None else 'append',
            opts=dict(
                width=400,
                height=400,
                xlabel='iteration',
                title='beta')
            )

        self.win_kld = self.viz.line(
            X=iters,
            Y=klds,
            env=self.viz_name+'_lines',
            win=self.win_kld,
            update=None if self.win_kld is None else 'append',
            opts=dict(
                width=400,
                height=400,
                legend=legend,
                xlabel='iteration',
                title='kl divergence')
            )

        # if self.win_mu is None:
        #     self.win_mu = self.viz.line(
        #                                 X=iters,
        #                                 Y=mus,
        #                                 env=self.viz_name+'_lines',
        #                                 opts=dict(
        #                                     width=400,
        #                                     height=400,
        #                                     legend=legend[:self.z_dim],
        #                                     xlabel='iteration',
        #                                     title='posterior mean',))
        # else:
        #     self.win_mu = self.viz.line(
        #                                 X=iters,
        #                                 Y=vars,
        #                                 env=self.viz_name+'_lines',
        #                                 win=self.win_mu,
        #                                 update='append',
        #                                 opts=dict(
        #                                     width=400,
        #                                     height=400,
        #                                     legend=legend[:self.z_dim],
        #                                     xlabel='iteration',
        #                                     title='posterior mean',))

        # if self.win_var is None:
        #     self.win_var = self.viz.line(
        #                                 X=iters,
        #                                 Y=vars,
        #                                 env=self.viz_name+'_lines',
        #                                 opts=dict(
        #                                     width=400,
        #                                     height=400,
        #                                     legend=legend[:self.z_dim],
        #                                     xlabel='iteration',
        #                                     title='posterior variance',))
        # else:
        #     self.win_var = self.viz.line(
        #                                 X=iters,
        #                                 Y=vars,
        #                                 env=self.viz_name+'_lines',
        #                                 win=self.win_var,
        #                                 update='append',
        #                                 opts=dict(
        #                                     width=400,
        #                                     height=400,
        #                                     legend=legend[:self.z_dim],
        #                                     xlabel='iteration',
        #                                     title='posterior variance',))
        self.net_mode(train=True)


    def viz_traverse(self, limit=3, inter=2/3, loc=-1):
        """
        Create visualization(s) for seeing the effect of traversing
        one latent variable's value, across a range, on reconstruction, for various selected images
        from a dataset, or from a randomly sampled latent vector.

        Parameters
        ----------
        limit: absolute value limit for latent variable range
        inter: traversal rate of latent variable value
        loc: if non-zero, the latent dimension/index to look at. If -1, we look at all the latent dimensions.


        Output (also, see visualization comments I made below)
        ------
        - Visdom image grids (one grid per selected img.) showcasing the effects of latent traversal on reconstruction
        -  Image grid GIFs (one GIF per selected img.) animating the traversals
        """

        self.net_mode(train=False)
        import random

        decoder = self.net.decoder
        encoder = self.net.encoder

        # interpolation range for each latent var.
        interpolation = torch.arange(-limit, limit+0.1, inter)

        n_dsets = len(self.data_loader.dataset)
        rand_idx = random.randint(1, n_dsets-1)

        # random_img = self.data_loader.dataset.__getitem__(rand_idx)
        random_img = self.data_loader.dataset[rand_idx]

        # .unsqueeze(0) allows us to have a batch size of one: 
        # e.g. shape (# channels, 64, 64) --> (1, # channels, 64, 64)
        random_img = cuda(random_img, self.use_cuda).unsqueeze(0)
        random_img_z = encoder(random_img)[:, :self.z_dim]

        random_z = cuda(torch.rand(1, self.z_dim), self.use_cuda)

        if self.dataset == 'dsprites':
            fixed_idx1 = 87040  # square
            fixed_idx2 = 332800  # ellipse
            fixed_idx3 = 578560  # heart

            # fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)
            fixed_img1 = self.data_loader.dataset[fixed_idx1]
            fixed_img1 = cuda(fixed_img1, self.use_cuda).unsqueeze(0)
            fixed_img_z1 = encoder(fixed_img1)[:, :self.z_dim]

            fixed_img2 = self.data_loader.dataset[fixed_idx2]
            fixed_img2 = cuda(fixed_img2, self.use_cuda).unsqueeze(0)
            fixed_img_z2 = encoder(fixed_img2)[:, :self.z_dim]

            fixed_img3 = self.data_loader.dataset[fixed_idx3]
            fixed_img3 = cuda(fixed_img3, self.use_cuda).unsqueeze(0)
            fixed_img_z3 = encoder(fixed_img3)[:, :self.z_dim]

            Z = {'fixed_square': fixed_img_z1, 'fixed_ellipse': fixed_img_z2,
                 'fixed_heart': fixed_img_z3, 'random_img': random_img_z}
        else:
            fixed_idx = 0
            fixed_img = self.data_loader.dataset[fixed_idx]
            fixed_img = cuda(fixed_img, self.use_cuda).unsqueeze(0)
            fixed_img_z = encoder(fixed_img)[:, :self.z_dim]

            Z = {'fixed_img': fixed_img_z,
                 'random_img': random_img_z, 'random_z': random_z}
        
        # samples --> Visdom image grid visualization (across interpolation range for all latent dimensions)
        # gifs --> creation of GIFs (each image on image grid animates through its singular latent value traversal)

        gifs = []
        for key in Z.keys():
            z_ori = Z[key]

            samples = []
            for row in range(self.z_dim):
                if loc != -1 and row != loc:
                    continue

                z = z_ori.clone()
                for val in interpolation:
                    z[:, row] = val  # row is the z latent variable
                    sample = torch.sigmoid(decoder(z)).data
                    samples.append(sample)
                    gifs.append(sample)
            
            """
            Visualization of `samples`. Each rectangular cell represents a reconstructed image
            from a latent representation modified with a specified latent variable value 
            (format: z_{dimension_number} = {value}):

             ----------       ---------
            |z_1 = -1.5| ... |z_1 = 1.5|
             ----------       ---------
             ----------       ---------
            |z_2 = -1.5| ... |z_2 = 1.5|
             ----------       ---------
                  .               .   
                  .               .
                  .               .
             ----------       ---------
            |z_n = -1.5| ... |z_n = 1.5|
             ----------       ---------

            Each row shows the traversal of one latent variable across the range.
            """

            samples = torch.cat(samples, dim=0).cpu()
            title = f'{key}_latent_traversal(iter:{self.global_iter})'

            if self.viz_on:
                self.viz.images(samples, env=self.viz_name+'_traverse',
                                opts=dict(title=title), nrow=len(interpolation))

        if self.save_output:
            output_dir = os.path.join(self.output_dir, str(self.global_iter))
            os.makedirs(output_dir, exist_ok=True)

            gifs = torch.cat(gifs)

            # I was tempted to do
            # gifs.view(len(Z), len(interpolation), self.z_dim, self.nc, 64, 64) instead.
            # but with toy testing, I saw that .view and .transpose work differently!
            """Specifically, .view preserves sequential order, and .transpose switches the analog
            of rows and columns in upper dimensions"""
            gifs = gifs.view(len(Z), self.z_dim, len(interpolation), self.nc, 64, 64).transpose(1, 2)

            """
            GIF generation:

            For every chosen image's (mean) latent representation, save every image grid of 
            reconstructed images for a fixed latent value across every dimension.

            Then, combine these image grids to a GIF.

            Visualization of `gifs` (tranpose of the 1st 2nd dims. of `samples`):
            (format: z_{dimension_number} = {value}):

             ----------       ---------
            |z_1 = -1.5| ... |z_n = -1.5|
             ----------       ---------
             ----------       ---------
            |z_1 = -1.0| ... |z_n = -1.0|
             ----------       ---------
                  .               .   
                  .               .
                  .               .
             ----------       ---------
            |z_1 = 1.5| ... |z_n = 1.5|
             ----------       ---------

            For the actual GIF, imagine each row as an image grid, and stacking together
            the image grids, where each grid is a time snapshot.
            """

            for i, key in enumerate(Z.keys()):
                for j, val in enumerate(interpolation):
                    save_image(tensor=gifs[i][j].cpu(),
                               fp=os.path.join(output_dir, f'{key}_{j}.jpg'),
                               nrow=self.z_dim, pad_value=1)

                grid2gif(os.path.join(output_dir, key+'*.jpg'),
                         os.path.join(output_dir, key+'.gif'), delay=10)

        self.net_mode(train=True)

    # nopep8

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise('Only bool type is supported. True or False')

        if train:
            self.net.train()
        else:
            self.net.eval()

    # nopep8

    def save_checkpoint(self, filename, silent=True):
        model_states = {'net': self.net.state_dict(), }
        optim_states = {'optim': self.optim.state_dict(), }
        win_states = {'recon': self.win_recon,
                      'beta': self.win_beta,
                      'kld': self.win_kld,
                      #   'mu':self.win_mu,
                      #   'var':self.win_var,
                      }
        states = {'iter': self.global_iter,
                  'win_states': win_states,
                  'model_states': model_states,
                  'optim_states': optim_states}

        file_path = os.path.join(self.ckpt_dir, filename)

        with open(file_path, mode='wb+') as f:
            torch.save(states, f)

        if not silent:
            print(f"=> saved checkpoint {file_path} (iter {self.global_iter})")


    def load_checkpoint(self, filename):
        file_path = os.path.join(self.ckpt_dir, filename)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            self.global_iter = checkpoint['iter']
            self.win_recon = checkpoint['win_states']['recon']
            self.win_kld = checkpoint['win_states']['kld']
            # self.win_var = checkpoint['win_states']['var']
            # self.win_mu = checkpoint['win_states']['mu']
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            print(f"=> loaded checkpoint {file_path} (iter {self.global_iter})")
        else:
            print(f"=> no checkpoint found at {file_path}")
