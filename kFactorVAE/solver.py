"""
Runs the training loop and produces reconstructions, latent traversals,
checkpoints, and logging data to graph training metrics. It may also
output reconstructions, latent traversals, and training metric graphs to the Visdom
server if the visualization user argument setting is enabled. 

This script builds upon the original `solver.py` found in the 
FactorVAE repo: https://github.com/1Konny/FactorVAE

In case any part of this code is confusing, I made many comments
on `solver_light.py` in the Disentangling folder of this GitHub repo.
"""

import os
import logging
import csv
import visdom
from tqdm import tqdm
from pathlib import Path
import pandas as pd

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


# Create a custom logging handler that writes messages to tqdm.write
class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg.strip())


class Solver(object):
    def __init__(self, args):
        # Misc
        use_cuda = args.cuda and torch.cuda.is_available()
        self.gpu = args.gpu

        self.seed = args.seed
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

        self.use_sort_strategy = args.use_sort_strategy
        if self.use_augment_dataloader:
            self.augment_factor = args.augment_factor 
            self.num_sim_factors = args.num_sim_factors

        self.denoise = args.denoise


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

        # graph data directory
        self.graph_data_dir_fp = Path(__file__).parent / args.graph_data_dir
        if not self.graph_data_dir_fp.exists():
            self.graph_data_dir_fp.mkdir()

        self.graph_data_subdir_fp = self.graph_data_dir_fp / f'{self.name}'
        if not self.graph_data_subdir_fp.exists(): self.graph_data_subdir_fp.mkdir()

        self.init_graph_data_loggers()


        if args.ckpt_load:
            self.load_checkpoint(args.ckpt_load)

        # Visdom visualization and logging
        if self.viz_on:
            if not os.path.exists("./vis_logs"):
                os.mkdir("./vis_logs")
            self.viz_port = args.viz_port

            # visdom.Visdom prints out "Setting up a new session ...", which makes the tqdm progress bar
            # print out twice instead of once; so I redirect that "Setting up..." logger print out
            # onto tqdm.write("Setting up a new session ...") (code assistance provided by ChatGPT)
            
            handler = TqdmLoggingHandler()
            logging.getLogger().addHandler(handler)

            self.viz = visdom.Visdom(port=self.viz_port, log_to_filename=f"./vis_logs/{self.name}")

            logging.getLogger().removeHandler(handler)


            self.viz_ll_iter = args.viz_ll_iter
            self.viz_la_iter = args.viz_la_iter
            self.viz_ra_iter = args.viz_ra_iter
            self.viz_ta_iter = args.viz_ta_iter

            # check for None or empty string (empty str. could come from checkpoint load)
            if not self.win_D_z:
                self.viz_init()
                self.pbar.write("Visdom line plot windows initialized")
            
            assert all([getattr(self, self.win_id[win_id]) for win_id in self.win_id])

            

        # Output(latent traverse GIF)
        self.output_dir = os.path.join(args.output_dir, args.name)
        self.output_save = args.output_save
        mkdirs(self.output_dir)
    
    

    def init_graph_data_loggers(self):
        # create and initialize the loggers
        self.recon_loss_csv_fp = self.graph_data_subdir_fp / 'recon_loss.csv'
        self.kl_div_csv_fp = self.graph_data_subdir_fp / 'kl_divs.csv'
        self.k_sim_loss_csv_fp = self.graph_data_subdir_fp / 'k_sim_loss.csv'
        self.discrim_acc_csv_fp = self.graph_data_subdir_fp / 'discrim_acc.csv'
        self.total_corr_csv_fp = self.graph_data_subdir_fp / 'total_corr.csv'

        # file objects from `open` method
        self.recon_loss_csv_file = open(self.recon_loss_csv_fp, 'a', newline='')
        self.kl_div_csv_file = open(self.kl_div_csv_fp, 'a', newline='')
        self.k_sim_loss_csv_file = open(self.k_sim_loss_csv_fp, 'a', newline='')
        self.discrim_acc_csv_file = open(self.discrim_acc_csv_fp, 'a', newline='')
        self.total_corr_csv_file = open(self.total_corr_csv_fp, 'a', newline='')

        # newline='' prevents a blank line between every row
        self.recon_loss_logger = csv.writer(self.recon_loss_csv_file, delimiter=',')
        self.kl_div_logger = csv.writer(self.kl_div_csv_file, delimiter=',')
        self.k_sim_loss_logger = csv.writer(self.k_sim_loss_csv_file, delimiter=',')
        self.discrim_acc_logger = csv.writer(self.discrim_acc_csv_file, delimiter=',')
        self.total_corr_logger = csv.writer(self.total_corr_csv_file, delimiter=',')

        scalar_metric_header = ['iteration', f'seed{self.seed}']
        kl_div_header = ['iteration'] + [f'z{i}_seed{self.seed}' for i in range(1, 11)] + \
            [f'mean_seed{self.seed}', f'total_seed{self.seed}']

        if self.recon_loss_csv_fp.stat().st_size == 0:
            self.recon_loss_logger.writerow(scalar_metric_header)

        if self.kl_div_csv_fp.stat().st_size == 0:
            self.kl_div_logger.writerow(kl_div_header)

        if self.k_sim_loss_csv_fp.stat().st_size == 0:
            self.k_sim_loss_logger.writerow(scalar_metric_header)

        if self.discrim_acc_csv_fp.stat().st_size == 0:
            self.discrim_acc_logger.writerow(scalar_metric_header)

        if self.total_corr_csv_fp.stat().st_size == 0:
            self.total_corr_logger.writerow(scalar_metric_header)
        
        self.graph_data_flush()

    
    def graph_data_flush(self):
        """so that CSV files are updated without delay"""
        self.recon_loss_csv_file.flush()
        self.kl_div_csv_file.flush()
        self.k_sim_loss_csv_file.flush()
        self.discrim_acc_csv_file.flush()
        self.total_corr_csv_file.flush()


    
    def log_graph_data(self, recon_loss, kl_divs, k_sim_loss, discrim_acc, total_corr):
        """Logs the graph data every self.viz_ll_iter iterations
        
        Keyword arguments:
        recon_loss (float) -- reconstruction loss
        kl_divs (list[float]) -- 
            per minibatch: averaged dimension-wise KL divergences + total average KL div. + average 
            sum-of-dimensions KL divergence
        k_sim_loss -- regularization term based on self.num_sim_factors (aka "k") and self.augment_factor 
        discrim_acc (float) -- discriminator accuracy
        total_corr (float) -- estimated total correlation by the discriminator
        """
        
        self.recon_loss_logger.writerow([f'{self.global_iter}', f'{round(recon_loss, 3)}'])
        self.kl_div_logger.writerow([f'{self.global_iter}'] + [f'{round(kld, 3)}' for kld in kl_divs])
        self.k_sim_loss_logger.writerow([f'{self.global_iter}', f'{round(k_sim_loss, 3)}'])
        self.discrim_acc_logger.writerow([f'{self.global_iter}', f'{round(discrim_acc, 3)}'])
        self.total_corr_logger.writerow([f'{self.global_iter}', f'{round(total_corr, 3)}'])

        self.graph_data_flush()



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
                vae_recon_loss = recon_loss(x_true1, x_recon, self.dataset.lower(), self.denoise)
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

                D_z_for_vae_loss = self.D(z)
                vae_tc_loss = (D_z_for_vae_loss[:, :1] - D_z_for_vae_loss[:, 1:]).mean()

                # k_sim_loss = k_factor_sim_losses_params(mu, logvar, self.num_sim_factors)
                k_sim_loss = torch.Tensor([-1.0]).to(self.device)
                precedence_idxs = None


                # if self.global_iter >= 100_000:

                # choose the first k factors of variation, from lowest to highest KL div.
                if self.use_sort_strategy:
                    precedence_idxs = torch.argsort(dim_wise_kld)
                    k_sim_loss = k_factor_sim_loss_samples(z, self.num_sim_factors, precedence_idxs)

                # no strategy; just go with selecting the first k indices
                else:
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
                    print_str = '[{}] vae_recon_loss:{:.3f} tot_kld:{:.3f} mean_kld: {:.3f} vae_tc_loss:{:.3f} D_loss:{:.3f} k_sim_loss:{:.3f}'

                    formatted_str = print_str.format(
                        self.global_iter, vae_recon_loss.item(), total_kld.item(), mean_kld.item(), 
                        vae_tc_loss.item(), D_loss.item(), k_sim_loss.item())
                    
                    if self.use_sort_strategy:
                        formatted_str += f' k idxs: {precedence_idxs[:self.num_sim_factors]}'

                    self.pbar.write(formatted_str)

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
                
                if self.global_iter % self.viz_ll_iter == 0:
                    kl_divs = [*list(dim_wise_kld.detach().cpu().numpy()), mean_kld.item(), total_kld.item()]
                    self.log_graph_data(vae_recon_loss.item(),
                                        kl_divs,
                                        k_sim_loss.item(), D_acc.item(), vae_tc_loss.item())


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
        self.pbar.write("Visdom line plot windows being instantiated")
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
                      Y=torch.stack([zero_init] * 12, -1),
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

        # save CSV files --> dataframes
        train_log = [
            pd.read_csv(self.recon_loss_csv_fp, index_col='iteration'),
            pd.read_csv(self.kl_div_csv_fp, index_col='iteration'),
            pd.read_csv(self.k_sim_loss_csv_fp, index_col='iteration'),
            pd.read_csv(self.discrim_acc_csv_fp, index_col='iteration'),
            pd.read_csv(self.total_corr_csv_fp, index_col='iteration')
        ]

        states = {'iter':self.global_iter,
                  'model_states':model_states,
                  'optim_states':optim_states,
                  'win_states':win_states,
                  'train_log':train_log}

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

            ckpts = [int(ckpt) for ckpt in ckpts if 'MIG' not in ckpt]
            ckpts.sort(reverse=True)
            ckptname = str(ckpts[0])

        filepath = os.path.join(self.ckpt_dir, ckptname)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f, map_location=torch.device(f'cuda:{self.gpu}'))

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

            train_log = checkpoint['train_log']
            train_log[0].to_csv(self.recon_loss_csv_fp)
            train_log[1].to_csv(self.kl_div_csv_fp)
            train_log[2].to_csv(self.k_sim_loss_csv_fp)
            train_log[3].to_csv(self.discrim_acc_csv_fp)
            train_log[4].to_csv(self.total_corr_csv_fp)
        else:
            if verbose:
                self.pbar.write("=> no checkpoint found at '{}'".format(filepath))
