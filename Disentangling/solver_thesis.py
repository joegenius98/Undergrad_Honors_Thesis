"""solver.py"""

# import matplotlib.pyplot as plt
# from P_PID import PIDControl
from dataset import return_data
from model import BetaVAE_H, BetaVAE_B, ContrastiveVAE_L
from utils import cuda, grid2gif
import csv

from torchvision.utils import make_grid, save_image
import torch.nn.functional as F
import torch.optim as optim

import visdom

from tqdm import tqdm
import os
import torch
# torch.cuda.set_device(0)
import warnings
warnings.filterwarnings("ignore")

from losses import reconstruction_loss, total_corr, contrastive_losses, kl_divergence


# from I_PID import PIDControl


# gathers info. about model during training loop
class DataGather(object):
    def __init__(self):
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):
        return dict(iter=[],
                    total_loss=[],
                    recon_loss=[],
                    total_corr=[],
                    total_kld=[],
                    dim_wise_kld=[],
                    mean_kld=[],
                    lambda_TC=[],
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
        # hardware
        self.use_cuda = args.cuda and torch.cuda.is_available()

        # training hyperparams.
        self.max_iter = args.max_iter
        self.global_iter = 0 # counter
        self.batch_size = args.batch_size
        self.lr = args.lr

        ## Adam optimizer beta1, beta2
        self.beta1 = args.beta1
        self.beta2 = args.beta2

        ## model
        self.z_dim = args.z_dim
        self.model = args.model
        self.objective = args.objective

        ### honors thesis by Joseph Lee and Huajie Shao

        #### original beta-vae by Higgins et al.
        self.beta = args.beta
        #### beta-TCVAE/total correlation
        self.beta_TC = args.beta_TC
        #### enforcing soft constraint on total correlation
        self.C_tc_start = args.C_tc_start
        self.C_tc_max = args.C_tc_max
        self.C_tc_step_val = args.C_tc_step_val
        self.lambda_tc = args.lambda_tc

        #### contrastive loss hyparams.
        self.num_sim_factors = args.num_sim_factors
        self.augment_factor = args.augment_factor

        ## dataset
        self.dset_dir = args.dset_dir
        self.dataset = args.dataset
        self.data_loader = return_data(args)

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
        # elif args.model == 'B':
        #     net = BetaVAE_B
        elif args.model == 'L':
            net = ContrastiveVAE_L
        else:
            raise NotImplementedError('only support model H or B')

        self.net = cuda(net(self.z_dim, self.nc), self.use_cuda)
        self.optim = optim.Adam(self.net.parameters(), lr=self.lr,
                                betas=(self.beta1, self.beta2))

        ## visualization
        self.viz_on = args.viz_on
        self.viz_name = args.viz_name
        self.viz_port = args.viz_port
        self.gather_step = args.gather_step
        self.save_output = args.save_output
        self.output_dir = os.path.join(args.output_dir, args.viz_name)
        self.save_step = args.save_step

        # `win` is short for "window"; these get instantiated later on
        self.win_recon = None
        self.win_total_loss = None
        self.win_tc = None
        self.win_lambda_tc = None
        self.win_beta = None
        self.win_kld = None
        self.win_mean_kld = None
        # self.win_mu = None
        # self.win_var = None

        # checkpoints
        self.ckpt_dir = os.path.join(args.ckpt_dir, args.viz_name)
        self.ckpt_name = args.ckpt_name

        if self.viz_on:
            self.viz = visdom.Visdom(port=self.viz_port, log_to_filename=f"./vis_logs/thesis_init_k={self.num_sim_factors}")

        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)
        if self.ckpt_name is not None:
            self.load_checkpoint(self.ckpt_name)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        # self.display_step = args.display_step

        self.gather = DataGather()

    # nopep8

    def train(self):
        """Runs training loop and provides Visdom server visualizations of images and training stats. These stats.
        include loss quanitities and beta"""

        self.net_mode(train=True)
        # self.C_max = cuda(torch.FloatTensor([self.C_max]), self.use_cuda)
        out = False

        pbar = tqdm(total=self.max_iter)
        pbar.update(self.global_iter)

        # write log to log file
        # outfile = os.path.join(self.ckpt_dir, "train.log")
        # kl_file = os.path.join(self.ckpt_dir, "train.kl")
        # fw_log = open(outfile, "w")
        # # fw_log
        # fw_kl = open(kl_file, "w")

        # newline='' prevents a blank line between every row
        log_file = open(os.path.join(f"./train_logs/train_log{self.num_sim_factors}.csv"), 'w', newline='')
        log_file_writer = csv.writer(log_file, delimiter=',')

        # header row construction 
        iter = ['iteration']
        loss_names = ['total_loss', 'recon_loss', 'total_corr', 'betaVAE_kld']
        dynamic_hyperparam_names = ['lambda_TC']
        z_dim_kld_names = [f'kld_dim{i+1}' for i in range(self.z_dim)]

        csv_row_names = iter + loss_names + dynamic_hyperparam_names + z_dim_kld_names
        log_file_writer.writerow(csv_row_names)
        # fw_kl.write('total KL\tz_dim' + '\n')

        lbd_step = 100
        alpha = 0.99
        period = 5000

        C_tc = self.C_tc_start

        while not out:
            for x in self.data_loader:
                self.global_iter += 1
                pbar.update(1)

                """Feedforward and calculating quantities for loss"""

                if torch.any(torch.isnan(x)):
                    raise ValueError("NaN minibatch")

                x = cuda(x, self.use_cuda)
                z_samples, x_recon, mu, logvar = self.net(x)

                if torch.any(torch.isnan(z_samples)):
                    raise ValueError("NaN z_samples")
                if torch.any(torch.isnan(x_recon)):
                    raise ValueError("NaN x_recon")
                if torch.any(torch.isnan(mu)):
                    raise ValueError("NaN mu")
                if torch.any(torch.isnan(z_samples)):
                    raise ValueError("NaN z_samples")


                recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

                if self.global_iter % period == 0:
                    C_tc += self.C_tc_step_val
                if C_tc > self.C_tc_max:
                    C_tc = self.C_tc_max

                tc = total_corr(z_samples, mu, logvar)
                C_tc = cuda(torch.Tensor([C_tc]), self.use_cuda)
                constrained_tc = tc - C_tc

                """Calculating loss"""
            
                if torch.any(torch.isnan(recon_loss)):
                    raise ValueError("NaN recon_loss")
                if torch.any(torch.isnan(total_kld)):
                    raise ValueError("NaN total_kld")
                if torch.any(torch.isnan(dim_wise_kld)):
                    raise ValueError("NaN dim_wise_kld")
                if torch.any(torch.isnan(mean_kld)):
                    raise ValueError("NaN mean_kld")
                if torch.any(torch.isnan(tc)):
                    raise ValueError("NaN total correlation")
                if torch.any(torch.isnan(C_tc)):
                    raise ValueError("NaN C_tc")
                if torch.any(torch.isnan(constrained_tc)):
                    raise ValueError("NaN constrained_tc")


                # honors thesis loss
                total_loss = None
                if self.objective == 'L':
                    total_loss = recon_loss + \
                        self.beta_TC * self.lambda_tc * constrained_tc + constrained_tc ** 2 + \
                        self.beta * total_kld + \
                        sum(contrastive_losses(z_samples, self.num_sim_factors))

                elif self.objective == 'H':
                    total_loss = recon_loss + self.beta * total_kld

                if torch.any(torch.isnan(sum(contrastive_losses(z_samples, self.num_sim_factors)))):
                    raise ValueError("NaN sum of contrastive losses")
                if torch.any(torch.isnan(total_loss)):
                    raise ValueError("NaN total loss")
                        

                # elif self.objective == 'B':
                #     # tricks for C
                #     C = torch.clamp(self.C_max/self.C_stop_iter * self.global_iter, 
                #                     self.C_start, self.C_max.data[0])

                #     total_loss = recon_loss + self.gamma*(total_kld-C).abs()
                

                """Backpropagation"""

                # re-initialize gradients to zero
                self.optim.zero_grad()
                # compute the gradient (i.e. partial derivs. w.r.t to neural net parameters)
                """
                note: total_loss has x_recon, which is from a forward pass of net, which is
                why calling .backward() has access to all the neural net parameters
                """
                total_loss.backward()

                grads = []
                for p in self.net.encoder.parameters():
                    if p.grad is not None:
                        grads.append(p.grad.view(-1))
                grads = torch.cat(grads)

                # examine gradient magnitudes
                print(f'Mean gradient magnitude: {grads.abs().mean().item()}')
                print(f'Max gradient magnitude: {grads.abs().max().item()}')
                print(f'Min gradient magnitude: {grads.abs().min().item()}')

                # update neural net params. based on gradient
                self.optim.step()

                with torch.no_grad():
                    if self.global_iter == 1:
                        constrain_ma = constrained_tc
                    else:
                        constrain_ma = alpha * constrain_ma.detach_() + (1 - alpha) * constrain_ma

                    if self.global_iter % lbd_step == 0 and self.global_iter > 500:
                        self.lambda_tc *= torch.clamp(torch.exp(constrain_ma), 0.9, 1.05)
                        self.lambda_tc = self.lambda_tc.item()

                """Store lots of training stats."""

                if self.viz_on and self.global_iter % self.gather_step == 0:
                    #  ['iter', 'total_loss', 'recon_loss', 'total_corr', 'total_kld', 'dim_wise_kld', 
                    # 'mean_kld', 'lambda_TC', 'mu', 'var', 'images', 'beta']
                    self.gather.insert(iter=self.global_iter,
                                        total_loss=total_loss,
                                        recon_loss=recon_loss.data, 
                                        total_corr=tc,
                                        total_kld=total_kld.data,
                                        dim_wise_kld=dim_wise_kld.data, 
                                        mean_kld=mean_kld.data, 
                                        lambda_TC=self.lambda_tc.data,
                                        mu=mu.mean(0).data, 
                                        var=logvar.exp().mean(0).data,
                                        beta=self.beta)
                
                """Log lots of training stats."""

                if self.global_iter % 20 == 0:
                    # write log to file
                    # if self.objective == 'B':
                    #     C = C.item()
                    # fw_log.write('[{}] recon_loss:{:.3f} total_kld:{:.3f} exp_kld:{:.3f} beta:{:.4f}\n'.format(
                    #     self.global_iter, recon_loss.item(), total_kld.item(), C, self.beta))
                    # # write KL to file
                    # dim_kl = dim_wise_kld.data.cpu().numpy()
                    # dim_kl = [str(k) for k in dim_kl]
                    # fw_kl.write('total_kld:{0:.3f}\t'.format(total_kld.item()))
                    # fw_kl.write('z_dim:' + ','.join(dim_kl) + '\n')
                    
                    # row names format:
                    # iteration, total_loss, recon_loss, total_corr, betaVAE_kld, lambda_TC, kld_dim0, kld_dim1, kld_dim2, ..., kld_dimn
                    row_data = [0] * len(csv_row_names)
                    row_data[0] = self.global_iter
                    row_data[1:6] = [l.item() for l in (total_loss, recon_loss, tc, total_kld)] + [self.lambda_tc]
                    row_data[7:] = list(dim_wise_kld.detach().cpu().numpy())

                    log_file_writer.writerow(row_data)

                    if self.global_iter % 500 == 0:
                        log_file.flush()
                        # fw_log.flush()
                        # fw_kl.flush()

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
        log_file.close()
        # fw_log.close()
        # fw_kl.close()


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
        # 'iter', 'total_loss', 'recon_loss', 'total_corr', 'total_kld', 'dim_wise_kld', 
        # 'klds', 'mean_kld', 'lambda_TC', 'beta'
        iters = torch.Tensor(self.gather.data['iter'])
        total_losses = torch.stack(self.gather.data['total_loss']).cpu()
        recon_losses = torch.stack(self.gather.data['recon_loss']).cpu()
        total_corrs = torch.stack(self.gather.data['total_corr']).cpu()

        total_klds = torch.stack(self.gather.data['total_kld']).cpu()
        dim_wise_klds = torch.stack(self.gather.data['dim_wise_kld']).cpu()
        klds = torch.cat([dim_wise_klds, total_klds], 1).cpu()

        mean_klds = torch.stack(self.gather.data['mean_kld']).cpu()

        lambdas_TC = torch.stack(self.gather.data['lambda_TC']).cpu()

        betas = torch.Tensor(self.gather.data['beta'])


        # legend
        legend = []
        for j in range(self.z_dim):
            legend.append(f'z_{j}')
        # legend.append('mean')
        legend.append('total')

        # "win" is short for "window"
        self.win_total_loss = self.viz.line(
            X=iters, Y=total_losses, env=self.viz_name+'_lines', win=self.win_total_loss,
            update=None if self.win_total_loss is None else 'append',
            opts=dict(width=400, height=400, xlabel='iteration', title='VAE total loss')
            )

        self.win_recon = self.viz.line(
            X=iters, Y=recon_losses, env=self.viz_name+'_lines', win=self.win_recon,
            update=None if self.win_recon is None else 'append',
            opts=dict(width=400, height=400, xlabel='iteration', title='reconsturction loss')
            )
        

        # TODO: finish creating the whole set of instance variables for mean_klds, lambda_TC
        # then check that every one has been initialized in the beginning and is saved/loaded
        # as s checkpoint
        self.win_tc = self.viz.line(
            X=iters, Y=total_corrs, env=self.viz_name+'_lines', win=self.win_tc,
            update=None if self.win_tc is None else 'append',
            opts=dict(width=400, height=400, xlabel='iteration', title='total correlation')
            )

        self.win_lambda_tc = self.viz.line(
            X=iters, Y=lambdas_TC, env=self.viz_name+'_lines', win=self.win_lambda_tc,
            update=None if self.win_lambda_tc is None else 'append',
            opts=dict(width=400, height=400, xlabel='iteration', title='total corr. lambda')
            )


        self.win_beta = self.viz.line(
            X=iters, Y=betas, env=self.viz_name+'_lines', win=self.win_beta,
            update=None if self.win_beta is None else 'append',
            opts=dict(width=400, height=400, xlabel='iteration', title='beta')
            )

        self.win_kld = self.viz.line(
            X=iters, Y=klds, env=self.viz_name+'_lines', win=self.win_kld,
            update=None if self.win_kld is None else 'append',
            opts=dict( width=400,height=400,legend=legend,xlabel='iteration', title='kl divergence')
            )

        self.win_mean_kld = self.viz.line(
            X=iters, Y=mean_klds, env=self.viz_name+'_lines', win=self.win_mean_kld,
            update=None if self.win_mean_kld is None else 'append',
            opts=dict( width=400,height=400,legend=legend,xlabel='iteration', title='mean kl div.')
            )
        # self.win_mu = self.viz.line(
        #     X=iters,
        #     Y=mus,
        #     env=self.viz_name+'_lines',
        #     win=self.win_mu,
        #     update=None if self.win_mu is None else 'append',
        #     opts=dict(
        #         width=400,
        #         height=400,
        #         legend=legend[:self.z_dim],
        #         xlabel='iteration',
        #         title='posterior mean',))

        # self.win_var = self.viz.line(
        #     X=iters,
        #     Y=vars,
        #     env=self.viz_name+'_lines',
        #     win=self.win_var,
        #     update=None if self.win_var is None else 'append',
        #     opts=dict(
        #         width=400,
        #         height=400,
        #         legend=legend[:self.z_dim],
        #         xlabel='iteration',
        #         title='posterior variance',))

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

        # 'iter', 'total_loss', 'recon_loss', 'total_corr', 'total_kld', 'dim_wise_kld', 
        # 'klds', 'mean_kld', 'lambda_TC', 'beta'
        win_states = {'recon': self.win_recon,
                      'total_loss': self.win_total_loss,
                      'tc': self.win_tc,
                      'lambda_tc': self.win_lambda_tc,
                      'beta': self.win_beta,
                      'kld': self.win_kld, # has dimension-wise and total KL div.
                      'mean_kld': self.win_mean_kld,
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

            # current training iteration
            self.global_iter = checkpoint['iter']

            # visdom window states
            win_states = checkpoint['win_states']

            self.win_recon = win_states['recon']
            self.win_total_loss = win_states['total_loss']
            self.win_tc = win_states['tc']
            self.win_lambda_tc = win_states['lambda_tc']
            self.win_beta = win_states['beta']
            self.win_kld = win_states['kld']
            self.win_mean_kld = win_states['mean_kld'] 
            # self.win_var = checkpoint['win_states']['var']
            # self.win_mu = checkpoint['win_states']['mu']

            # model and optimizer (e.g. Adam) states
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            print(f"=> loaded checkpoint {file_path} (iter {self.global_iter})")
        else:
            print(f"=> no checkpoint found at {file_path}")
