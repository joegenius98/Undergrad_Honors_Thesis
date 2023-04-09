"""
solver.py

To help you understand this code, I made many comments
on `solver_light.py` in the Disentangling folder of this GitHub repo.
"""

import os
import visdom
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
# torch.autograd.set_detect_anomaly(True)

from utils import DataGather, mkdirs, grid2gif
from ops import recon_loss, kl_divergence, permute_dims
from model import FactorVAE1, FactorVAE2, Discriminator
from dataset import return_data

# from thesis_losses import k_factor_sim_losses_params
from thesis_losses import k_factor_sim_loss_samples


class Solver(object):
    def __init__(self, args):
        # Misc
        use_cuda = args.cuda and torch.cuda.is_available()
        self.device = 'cuda' if use_cuda else 'cpu'
        self.name = args.name
        self.max_iter = int(args.max_iter)
        self.print_iter = args.print_iter
        self.global_iter = 0
        self.pbar = tqdm(total=self.max_iter)

        # Data
        self.dset_dir = args.dset_dir
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.data_loader = return_data(args)
        self.use_augment_dataloader = args.use_augment_dataloader

        # Networks & Optimizers
        self.z_dim = args.z_dim
        self.gamma = args.gamma

        self.lr_VAE = args.lr_VAE
        self.beta1_VAE = args.beta1_VAE
        self.beta2_VAE = args.beta2_VAE

        self.lr_D = args.lr_D
        self.beta1_D = args.beta1_D
        self.beta2_D = args.beta2_D

        # Honors thesis idea
        self.augment_factor = 0
        self.num_sim_factors = None

        if self.use_augment_dataloader:
            self.augment_factor = args.augment_factor 
            self.num_sim_factors = args.num_sim_factors


        if args.dataset == 'dsprites':
            self.VAE = FactorVAE1(self.z_dim).to(self.device)
            self.nc = 1
        else:
            self.VAE = FactorVAE2(self.z_dim).to(self.device)
            self.nc = 3
        self.optim_VAE = optim.Adam(self.VAE.parameters(), lr=self.lr_VAE,
                                    betas=(self.beta1_VAE, self.beta2_VAE))

        self.D = Discriminator(self.z_dim).to(self.device)
        self.optim_D = optim.Adam(self.D.parameters(), lr=self.lr_D,
                                  betas=(self.beta1_D, self.beta2_D))

        self.nets = [self.VAE, self.D]

        # Visdom
        self.viz_on = args.viz_on
        self.win_id = dict(D_z='win_D_z', recon='win_recon', kld='win_kld', D_acc='win_D_acc', 
                           vae_tc='win_vae_tc', D_loss='win_D_loss', k_sim_loss='win_k_sim_loss')

        self.win_D_z, self.win_recon, self.win_kld, self.win_D_acc, \
            self.win_vae_tc, self.win_D_loss, self.win_k_sim_loss = (None,) * len(self.win_id)

        self.line_gather = DataGather('iter', 'soft_D_z', 'soft_D_z_pperm', 'recon', 'total_kld', 'dim_wise_kld', 'mean_kld', \
                                      'D_acc', 'vae_tc', 'D_loss', 'k_sim_loss')
        self.image_gather = DataGather('true', 'recon')

        # Checkpoint
        self.ckpt_dir = os.path.join(args.ckpt_dir, args.name)
        self.ckpt_save_iter = args.ckpt_save_iter
        mkdirs(self.ckpt_dir)
        if args.ckpt_load:
            self.load_checkpoint(args.ckpt_load)

        # Visdom visualization and logging
        if self.viz_on:
            if not os.path.exists("./vis_logs"):
                os.mkdir("./vis_logs")
            self.viz_port = args.viz_port
            self.viz = visdom.Visdom(port=self.viz_port, log_to_filename=f"./vis_logs/{self.name}")

            self.viz_ll_iter = args.viz_ll_iter
            self.viz_la_iter = args.viz_la_iter
            self.viz_ra_iter = args.viz_ra_iter
            self.viz_ta_iter = args.viz_ta_iter

            # check for None or empty string (empty str. could come from checkpoint load)
            if not self.win_D_z:
                self.viz_init()
                print("Visdom line plot windows initialized")
            
            assert all([getattr(self, self.win_id[win_id]) for win_id in self.win_id])

            

        # Output(latent traverse GIF)
        self.output_dir = os.path.join(args.output_dir, args.name)
        self.output_save = args.output_save
        mkdirs(self.output_dir)


    def train(self):
        self.net_mode(train=True)

        ones, zeros = None, None
        if not self.use_augment_dataloader:
            ones = torch.ones(self.batch_size, dtype=torch.long, device=self.device)
            zeros = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
        else:
            ones = torch.ones(self.batch_size * 2, dtype=torch.long, device=self.device)
            zeros = torch.zeros(self.batch_size * 2, dtype=torch.long, device=self.device)


        out = False
        while not out:
            for x_true1, x_true2 in self.data_loader:

                self.global_iter += 1
                self.pbar.update(1)

                x_true1 = x_true1.to(self.device)
                x_recon, mu, logvar, z = self.VAE(x_true1)
                vae_recon_loss = recon_loss(x_true1, x_recon)
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

                D_z_for_vae_loss = self.D(z)
                vae_tc_loss = (D_z_for_vae_loss[:, :1] - D_z_for_vae_loss[:, 1:]).mean()

                # k_sim_loss = k_factor_sim_losses_params(mu, logvar, self.num_sim_factors)
                k_sim_loss = k_factor_sim_loss_samples(z, self.num_sim_factors)
                vae_loss = vae_recon_loss + total_kld + self.gamma*vae_tc_loss + self.augment_factor * k_sim_loss

                self.optim_VAE.zero_grad()
                vae_loss.backward(retain_graph=True)
                self.optim_VAE.step()


                # detach z to prevent updating VAE params. (would cause an err. anyway)
                D_z_for_discrim_loss = self.D(z.detach())
                x_true2 = x_true2.to(self.device)
                z_prime = self.VAE(x_true2, no_dec=True)
                # detach b/c we don't want to update VAE params.
                z_pperm = permute_dims(z_prime).detach()
                D_z_pperm = self.D(z_pperm)
                D_loss = 0.5*(F.cross_entropy(D_z_for_discrim_loss, zeros) + F.cross_entropy(D_z_pperm, ones))

                self.optim_D.zero_grad()
                D_loss.backward()
                self.optim_D.step()

                if self.global_iter%self.print_iter == 0:
                    print_str = '[{}] vae_recon_loss:{:.3f} vae_kld:{:.3f} vae_tc_loss:{:.3f} D_loss:{:.3f} k_sim_loss:{:.3f}'
                    self.pbar.write(print_str.format(
                        self.global_iter, vae_recon_loss.item(), total_kld.item(), vae_tc_loss.item(), D_loss.item(), 
                        k_sim_loss.item()))

                if self.global_iter%self.ckpt_save_iter == 0:
                    self.save_checkpoint(self.global_iter)

                if self.viz_on and (self.global_iter%self.viz_ll_iter == 0):
                    soft_D_z = F.softmax(D_z_for_vae_loss, 1)[:, :1].detach()
                    soft_D_z_pperm = F.softmax(D_z_pperm, 1)[:, :1].detach()

                    D_acc = ((soft_D_z >= 0.5).sum() + (soft_D_z_pperm < 0.5).sum()).float()
                    if self.use_augment_dataloader:
                        D_acc /= 4*self.batch_size
                    else:
                        D_acc /= 2*self.batch_size

                    self.line_gather.insert(iter=self.global_iter,
                                            soft_D_z=soft_D_z.mean().item(),
                                            soft_D_z_pperm=soft_D_z_pperm.mean().item(),
                                            recon=vae_recon_loss.item(),
                                            total_kld=total_kld.data,
                                            dim_wise_kld=dim_wise_kld.data,
                                            mean_kld=mean_kld.data,
                                            D_acc=D_acc.item(),
                                            vae_tc=vae_tc_loss.item(),
                                            D_loss=D_loss.item(),
                                            k_sim_loss=k_sim_loss.item())

                if self.viz_on and (self.global_iter%self.viz_la_iter == 0):
                    self.visualize_line()
                    self.line_gather.flush()

                if self.viz_on and (self.global_iter%self.viz_ra_iter == 0):
                    self.image_gather.insert(true=x_true1.data.cpu(),
                                             recon=torch.sigmoid(x_recon).data.cpu())
                    self.visualize_recon()
                    self.image_gather.flush()

                if self.viz_on and (self.global_iter%self.viz_ta_iter == 0):
                    if self.dataset.lower() == '3dchairs':
                        self.visualize_traverse(limit=2, inter=0.5)
                    else:
                        self.visualize_traverse(limit=3, inter=2/3)

                if self.global_iter >= self.max_iter:
                    out = True
                    break

        self.pbar.write("[Training Finished]")
        self.pbar.close()

    def visualize_recon(self):
        data = self.image_gather.data
        true_image = data['true'][0]
        recon_image = data['recon'][0]

        true_image = make_grid(true_image)
        recon_image = make_grid(recon_image)
        sample = torch.stack([true_image, recon_image], dim=0)
        self.viz.images(sample, env=self.name+'/recon_image',
                        opts=dict(title=str(self.global_iter)))

        if self.output_save:
            output_dir = os.path.join(self.output_dir, str(self.global_iter))
            os.makedirs(output_dir, exist_ok=True)
            save_image(tensor=sample, fp=os.path.join(output_dir, 'recon.jpg'), 
                       pad_value=1)


    def visualize_line(self):
        data = self.line_gather.data
        iters = torch.Tensor(data['iter'])
        recon = torch.Tensor(data['recon'])

        # kld = torch.Tensor(data['kld'])
        total_klds = torch.stack(data['total_kld'])
        dim_wise_klds = torch.stack(data['dim_wise_kld'])
        mean_klds = torch.stack(data['mean_kld'])
        klds = torch.cat([dim_wise_klds, mean_klds, total_klds], 1)

        D_acc = torch.Tensor(data['D_acc'])
        soft_D_z = torch.Tensor(data['soft_D_z'])
        soft_D_z_pperm = torch.Tensor(data['soft_D_z_pperm'])
        soft_D_zs = torch.stack([soft_D_z, soft_D_z_pperm], -1)

        vae_tcs = torch.Tensor(data['vae_tc'])
        D_losses = torch.Tensor(data['D_loss'])

        k_sim_losses = torch.Tensor(data['k_sim_loss'])

        self.win_D_z = self.viz.line(X=iters,
                      Y=soft_D_zs,
                      env=self.name+'_lines',
                      win=self.win_id['D_z'],
                      update='append',
                      opts=dict(
                        xlabel='iteration',
                        ylabel='D(.)',
                        legend=['D(z)', 'D(z_perm)']))

        self.win_recon = self.viz.line(X=iters,
                      Y=recon,
                      env=self.name+'_lines',
                      win=self.win_id['recon'],
                      update='append',
                      opts=dict(
                        xlabel='iteration',
                        ylabel='reconstruction loss',))

        self.win_D_acc = self.viz.line(X=iters,
                      Y=D_acc,
                      env=self.name+'_lines',
                      win=self.win_id['D_acc'],
                      update='append',
                      opts=dict(
                        xlabel='iteration',
                        ylabel='discriminator accuracy',))

        self.win_kld = self.viz.line(X=iters,
                      Y=klds,
                      env=self.name+'_lines',
                      win=self.win_id['kld'],
                      update='append',
                      opts=dict(
                        xlabel='iteration',
                        ylabel='kl divergence',))

        self.win_vae_tc = self.viz.line(X=iters,
                      Y=vae_tcs,
                      env=self.name+'_lines',
                      win=self.win_id['vae_tc'],
                      update='append',
                      opts=dict(
                        xlabel='iteration',
                        ylabel='Total Corr. (VAE)',))

        self.win_D_loss = self.viz.line(X=iters,
                      Y=D_losses,
                      env=self.name+'_lines',
                      win=self.win_id['D_loss'],
                      update='append',
                      opts=dict(
                        xlabel='iteration',
                        ylabel='Discriminator Loss',))
        
        self.win_k_sim_loss = self.viz.line(X=iters, Y=k_sim_losses,
                    env=f'{self.name}_lines', win=self.win_id['k_sim_loss'], update='append',
                    opts={'xlabel': 'iteration', 'ylabel': 'k-factor similarity loss'})


    def visualize_traverse(self, limit=3, inter=2/3, loc=-1):
        self.net_mode(train=False)

        decoder = self.VAE.decode
        encoder = self.VAE.encode
        interpolation = torch.arange(-limit, limit+0.1, inter)

        random_img = self.data_loader.dataset.__getitem__(0)[1]
        random_img = random_img.to(self.device).unsqueeze(0)
        random_img_z = encoder(random_img)[:, :self.z_dim]

        if self.dataset.lower() == 'dsprites':
            fixed_idx1 = 87040 # square
            fixed_idx2 = 332800 # ellipse
            fixed_idx3 = 578560 # heart

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)[0]
            fixed_img1 = fixed_img1.to(self.device).unsqueeze(0)
            fixed_img_z1 = encoder(fixed_img1)[:, :self.z_dim]

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)[0]
            fixed_img2 = fixed_img2.to(self.device).unsqueeze(0)
            fixed_img_z2 = encoder(fixed_img2)[:, :self.z_dim]

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)[0]
            fixed_img3 = fixed_img3.to(self.device).unsqueeze(0)
            fixed_img_z3 = encoder(fixed_img3)[:, :self.z_dim]

            Z = {'fixed_square':fixed_img_z1, 'fixed_ellipse':fixed_img_z2,
                 'fixed_heart':fixed_img_z3, 'random_img':random_img_z}

        elif self.dataset.lower() == 'celeba':
            fixed_idx1 = 191281 # 'CelebA/img_align_celeba/191282.jpg'
            fixed_idx2 = 143307 # 'CelebA/img_align_celeba/143308.jpg'
            fixed_idx3 = 101535 # 'CelebA/img_align_celeba/101536.jpg'
            fixed_idx4 = 70059  # 'CelebA/img_align_celeba/070060.jpg'

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)[0]
            fixed_img1 = fixed_img1.to(self.device).unsqueeze(0)
            fixed_img_z1 = encoder(fixed_img1)[:, :self.z_dim]

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)[0]
            fixed_img2 = fixed_img2.to(self.device).unsqueeze(0)
            fixed_img_z2 = encoder(fixed_img2)[:, :self.z_dim]

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)[0]
            fixed_img3 = fixed_img3.to(self.device).unsqueeze(0)
            fixed_img_z3 = encoder(fixed_img3)[:, :self.z_dim]

            fixed_img4 = self.data_loader.dataset.__getitem__(fixed_idx4)[0]
            fixed_img4 = fixed_img4.to(self.device).unsqueeze(0)
            fixed_img_z4 = encoder(fixed_img4)[:, :self.z_dim]

            Z = {'fixed_1':fixed_img_z1, 'fixed_2':fixed_img_z2,
                 'fixed_3':fixed_img_z3, 'fixed_4':fixed_img_z4,
                 'random':random_img_z}

        elif self.dataset.lower() == '3dchairs':
            fixed_idx1 = 40919 # 3DChairs/images/4682_image_052_p030_t232_r096.png
            fixed_idx2 = 5172  # 3DChairs/images/14657_image_020_p020_t232_r096.png
            fixed_idx3 = 22330 # 3DChairs/images/30099_image_052_p030_t232_r096.png

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)[0]
            fixed_img1 = fixed_img1.to(self.device).unsqueeze(0)
            fixed_img_z1 = encoder(fixed_img1)[:, :self.z_dim]

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)[0]
            fixed_img2 = fixed_img2.to(self.device).unsqueeze(0)
            fixed_img_z2 = encoder(fixed_img2)[:, :self.z_dim]

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)[0]
            fixed_img3 = fixed_img3.to(self.device).unsqueeze(0)
            fixed_img_z3 = encoder(fixed_img3)[:, :self.z_dim]

            Z = {'fixed_1':fixed_img_z1, 'fixed_2':fixed_img_z2,
                 'fixed_3':fixed_img_z3, 'random':random_img_z}
        else:
            fixed_idx = 0
            fixed_img = self.data_loader.dataset.__getitem__(fixed_idx)[0]
            fixed_img = fixed_img.to(self.device).unsqueeze(0)
            fixed_img_z = encoder(fixed_img)[:, :self.z_dim]

            random_z = torch.rand(1, self.z_dim, 1, 1, device=self.device)

            Z = {'fixed_img':fixed_img_z, 'random_img':random_img_z, 'random_z':random_z}

        gifs = []
        for key in Z:
            z_ori = Z[key]
            samples = []
            for row in range(self.z_dim):
                if loc != -1 and row != loc:
                    continue
                z = z_ori.clone()
                for val in interpolation:
                    z[:, row] = val
                    sample = torch.sigmoid(decoder(z)).data
                    samples.append(sample)
                    gifs.append(sample)
            samples = torch.cat(samples, dim=0).cpu()
            title = '{}_latent_traversal(iter:{})'.format(key, self.global_iter)
            self.viz.images(samples, env=self.name+'_traverse',
                            opts=dict(title=title), nrow=len(interpolation))

        if self.output_save:
            output_dir = os.path.join(self.output_dir, str(self.global_iter))
            mkdirs(output_dir)
            gifs = torch.cat(gifs)
            gifs = gifs.view(len(Z), self.z_dim, len(interpolation), self.nc, 64, 64).transpose(1, 2)
            for i, key in enumerate(Z.keys()):
                for j, val in enumerate(interpolation):
                    save_image(tensor=gifs[i][j].cpu(),
                               fp=os.path.join(output_dir, '{}_{}.jpg'.format(key, j)),
                               nrow=self.z_dim, pad_value=1)

                grid2gif(str(os.path.join(output_dir, key+'*.jpg')),
                         str(os.path.join(output_dir, key+'.gif')), delay=10)

        self.net_mode(train=True)

    def viz_init(self):
        print("Visdom line plot windows being instantiated")
        zero_init = torch.zeros([1])
        self.win_D_z = self.viz.line(X=zero_init,
                      Y=torch.stack([zero_init, zero_init], -1),
                      env=self.name+'_lines',
                      win=self.win_id['D_z'],
                      opts=dict(
                        xlabel='iteration',
                        ylabel='D(.)',
                        legend=['D(z)', 'D(z_perm)']))

        self.win_recon = self.viz.line(X=zero_init,
                      Y=zero_init,
                      env=self.name+'_lines',
                      win=self.win_id['recon'],
                      opts=dict(
                        xlabel='iteration',
                        ylabel='reconstruction loss',))

        self.win_D_acc = self.viz.line(X=zero_init,
                      Y=zero_init,
                      env=self.name+'_lines',
                      win=self.win_id['D_acc'],
                      opts=dict(
                        xlabel='iteration',
                        ylabel='discriminator accuracy',))

        self.win_kld = self.viz.line(X=zero_init,
                      Y=zero_init,
                      env=self.name+'_lines',
                      win=self.win_id['kld'],
                      opts=dict(
                        xlabel='iteration',
                        ylabel='kl divergence',))

        self.win_vae_tc = self.viz.line(X=zero_init,
                      Y=zero_init,
                      env=self.name+'_lines',
                      win=self.win_id['vae_tc'],
                      opts=dict(
                        xlabel='iteration',
                        ylabel='Total Corr. (VAE)',))

        self.win_D_loss = self.viz.line(X=zero_init,
                      Y=zero_init,
                      env=self.name+'_lines',
                      win=self.win_id['D_loss'],
                      opts=dict(
                        xlabel='iteration',
                        ylabel='Discriminator Loss',))

        self.win_k_sim_loss = self.viz.line(X=zero_init,
                      Y=zero_init,
                      env=self.name+'_lines',
                      win=self.win_id['k_sim_loss'],
                      opts=dict(
                        xlabel='iteration',
                        ylabel='k-factor similarity loss',))
        
    def net_mode(self, train):
        if not isinstance(train, bool):
            raise ValueError('Only bool type is supported. True|False')

        for net in self.nets:
            if train:
                net.train()
            else:
                net.eval()

    def save_checkpoint(self, ckptname='last', verbose=True):
        model_states = {'D':self.D.state_dict(),
                        'VAE':self.VAE.state_dict()}
        optim_states = {'optim_D':self.optim_D.state_dict(),
                        'optim_VAE':self.optim_VAE.state_dict()}
        # save all the Visdom line window states
        win_states={win_id: getattr(self, self.win_id[win_id]) for win_id in self.win_id}

        states = {'iter':self.global_iter,
                  'model_states':model_states,
                  'optim_states':optim_states,
                  'win_states':win_states}

        filepath = os.path.join(self.ckpt_dir, str(ckptname))

        with open(filepath, 'wb+') as f:
            torch.save(states, f)
        if verbose:
            self.pbar.write("=> saved checkpoint '{}' (iter {})".format(filepath, self.global_iter))

    def load_checkpoint(self, ckptname='last', verbose=True):
        if ckptname == 'last':
            ckpts = os.listdir(self.ckpt_dir)
            if not ckpts:
                if verbose:
                    self.pbar.write("=> no checkpoint found")
                return

            ckpts = [int(ckpt) for ckpt in ckpts]
            ckpts.sort(reverse=True)
            ckptname = str(ckpts[0])

        filepath = os.path.join(self.ckpt_dir, ckptname)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)

            self.global_iter = checkpoint['iter']
            self.VAE.load_state_dict(checkpoint['model_states']['VAE'])
            self.D.load_state_dict(checkpoint['model_states']['D'])
            self.optim_VAE.load_state_dict(checkpoint['optim_states']['optim_VAE'])
            self.optim_D.load_state_dict(checkpoint['optim_states']['optim_D'])
            self.pbar.update(self.global_iter)
            if verbose:
                self.pbar.write("=> loaded checkpoint '{} (iter {})'".format(filepath, self.global_iter))
            
            # get all Visdom window states
            for win_id in checkpoint['win_states']:
                setattr(self, f"win_{win_id}", checkpoint['win_states'][win_id])
        else:
            if verbose:
                self.pbar.write("=> no checkpoint found at '{}'".format(filepath))
