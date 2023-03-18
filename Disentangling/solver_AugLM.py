"""Helper code for 0<KL divergence<= C constraint"""


import torch
# torch.cuda.set_device(0)
import warnings
warnings.filterwarnings("ignore")

import os
from tqdm import tqdm
import visdom
import torch.nn

import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image

from utils import cuda, grid2gif
# from model_batch import BetaVAE_H, BetaVAE_B,reparametrize
from model import BetaVAE_H, BetaVAE_B,reparametrize
from dataset import return_data
# from dataset import _data_loader

import numpy as np
import matplotlib.pyplot as plt
# from active_unit import _active_unit

def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'gaussian':
        # x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
        recon_loss = None

    return recon_loss
    

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)
    
    return total_kld, dimension_wise_kld, mean_kld


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
        print("use cuda : ", self.use_cuda)
        self.max_iter = args.max_iter
        # self.max_vae_steps = args.max_vae_steps
        self.global_iter = 0
        
        self.z_dim = args.z_dim
        self.beta = args.beta
        self.gamma = args.gamma
        self.lambd = args.lambd
        # self.Kp = args.Kp
        # self.Ki = args.Ki
        self.C_max = args.C_max
        # self.C_max_org = args.C_max
        self.C_stop_iter = args.C_stop_iter
        self.objective = args.objective
        self.model = args.model
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        # self.KL_loss = args.KL_loss
        self.step_value = args.step_val

        self.model_name = args.viz_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if args.dataset.lower() == 'dsprites':
            self.nc = 1
            self.decoder_dist = 'bernoulli'
        elif args.dataset.lower() == '3dchairs':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        elif args.dataset.lower() == 'cifar10':
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
        ## load model
        self.net = cuda(net(self.z_dim, self.nc), self.use_cuda)
        self.optim = optim.Adam(self.net.parameters(), lr=self.lr,
                                    betas=(self.beta1, self.beta2))
        
        self.viz_name = args.viz_name
        self.viz_port = args.viz_port
        self.viz_on = args.viz_on
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
        self.display_step = args.display_step
        self.save_step = args.save_step

        self.dset_dir = args.dset_dir
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        # self.data_loader = _data_loader(args)
        self.data_loader = return_data(args)
        
        self.gather = DataGather()
        self.gather2 = DataGather()
        

    def train(self):
        self.net_mode(train=True)
        self.C_max = Variable(cuda(torch.FloatTensor([self.C_max]), self.use_cuda))
        out = False
        
        pbar = tqdm(total=self.max_iter)
        pbar.update(self.global_iter)
        ## write log to log file
        outfile = os.path.join(self.ckpt_dir, "train.log")
        fw_log = open(outfile, "w")
        lbd_step = 100
        alpha = 0.99
        period = 5000
        fw_log.write("KL_divergence: {}\n".format(self.C_max.item()))
        C = 0.5

        while not out:
            for x in self.data_loader:
                # print('x and target >>', x.size(), targets, len(targets))
                self.global_iter += 1
                pbar.update(1)
                
                x = Variable(cuda(x, self.use_cuda))
                x_recon, mu, logvar, z = self.net(x)
                recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
                ## error
                if self.global_iter % period == 0:
                    C += self.step_value
                if C > self.C_max:
                    C = self.C_max
                
                C = torch.tensor([C]).to(self.device)
                constraint_kld = total_kld - C
                # constraint_kld = total_kld - self.C_max
                elbo = recon_loss.item() + total_kld.item()
                taming_loss = recon_loss + self.lambd * constraint_kld + 1*constraint_kld**2
                ## update gradient
                self.optim.zero_grad()
                taming_loss.backward()
                self.optim.step()
                ##
                with torch.no_grad():
                    if self.global_iter == 1:
                        constrain_ma = constraint_kld
                    else:
                        constrain_ma = alpha * constrain_ma.detach_() + (1 - alpha) * constraint_kld
                    if self.global_iter % lbd_step == 0 and self.global_iter > 500:
                        self.lambd *= torch.clamp(torch.exp(constrain_ma), 0.9, 1.05)
                        self.lambd = self.lambd .item()
                ## visdom
                if self.viz_on and self.global_iter%self.gather_step == 0:
                    self.gather.insert(iter=self.global_iter,
                                       mu=mu.mean(0).data, var=logvar.exp().mean(0).data,
                                       recon_loss=recon_loss.data, total_kld=total_kld.data,
                                       mean_kld=mean_kld.data, beta=self.beta)
                    self.gather2.insert(iter=self.global_iter,
                                       mu=mu.mean(0).data, var=logvar.exp().mean(0).data,
                                       recon_loss=recon_loss.data, total_kld=total_kld.data,
                                       mean_kld=mean_kld.data, beta=self.beta)
                    
                if self.global_iter % 20 == 0:
                    ## write log to file
                    elbo = recon_loss.item() + total_kld.item()
                    # fw_log.write('[{}] recon_loss:{:.3f} total_kld:{:.3f} elbo:{:.3f} lambdas:{:.4f}\n'.format(
                    #         self.global_iter, recon_loss.item(), total_kld.item(), -elbo, self.lambd))
                    fw_log.write('[{}] recon_loss:{:.3f} total_kld:{:.3f} exp_kld:{:.3f} lambdas:{:.4f}\n'.format(
                            self.global_iter, recon_loss.item(), total_kld.item(), C.item(), self.lambd))
                    fw_log.flush()
                    
                
                if self.viz_on and self.global_iter%self.save_step==0:
                    self.gather.insert(images=x.data)
                    self.gather.insert(images=F.sigmoid(x_recon).data)
                    self.viz_reconstruction()
                    self.viz_lines()
                    self.gather.flush()
                    
                if (self.viz_on or self.save_output) and self.global_iter%50000==0:
                    self.viz_traverse()

                if self.global_iter%self.save_step == 0:
                    self.save_checkpoint('last')
                    pbar.write('Saved checkpoint(iter:{})'.format(self.global_iter))
                    
                if self.global_iter%20000 == 0:
                    self.save_checkpoint(str(self.global_iter))

                if self.global_iter >= self.max_iter:
                    out = True
                    break

        pbar.write("[Training Finished]")
        pbar.close()
        fw_log.close()
        

    def viz_reconstruction(self):
        self.net_mode(train=False)
        x = self.gather.data['images'][0][:100]
        x = make_grid(x, normalize=True)
        x_recon = self.gather.data['images'][1][:100]
        x_recon = make_grid(x_recon, normalize=True)
        images = torch.stack([x, x_recon], dim=0).cpu()
        self.viz.images(images, env=self.viz_name+'_reconstruction',
                        opts=dict(title=str(self.global_iter)), nrow=10)
        if self.save_output:
            output_dir = os.path.join(self.output_dir, str(self.global_iter))
            os.makedirs(output_dir, exist_ok=True)
            save_image(tensor=images, filename=os.path.join(output_dir, 'recon.jpg'), pad_value=1)
        self.net_mode(train=True)
        

    def viz_lines(self):
        self.net_mode(train=False)
        recon_losses = torch.stack(self.gather.data['recon_loss']).cpu()
        betas = torch.Tensor(self.gather.data['beta'])

        # mus = torch.stack(self.gather.data['mu']).cpu()
        # vars = torch.stack(self.gather.data['var']).cpu()
        
        # dim_wise_klds = torch.stack(self.gather.data['dim_wise_kld'])
        mean_klds = torch.stack(self.gather.data['mean_kld'])
        total_klds = torch.stack(self.gather.data['total_kld'])
        # klds = torch.cat([dim_wise_klds, mean_klds, total_klds], 1).cpu()
        klds = torch.cat([mean_klds, total_klds], 1).cpu()
        iters = torch.Tensor(self.gather.data['iter'])
        
        recon_losses_2 = torch.stack(self.gather2.data['recon_loss']).cpu()
        betas_2 = torch.Tensor(self.gather2.data['beta'])
        
        # mus_2 = torch.stack(self.gather2.data['mu']).cpu()
        # vars_2 = torch.stack(self.gather2.data['var']).cpu()
        
        # dim_wise_klds_2 = torch.stack(self.gather2.data['dim_wise_kld'])
        mean_klds_2 = torch.stack(self.gather2.data['mean_kld'])
        total_klds_2 = torch.stack(self.gather2.data['total_kld'])
        klds_2 = torch.cat([mean_klds_2, total_klds_2], 1).cpu()
        iters_2 = torch.Tensor(self.gather2.data['iter'])
        
        legend = []
        # for z_j in range(self.z_dim):
        #     legend.append('z_{}'.format(z_j))
        legend.append('mean')
        legend.append('total')
        
        if self.win_recon is None:
            self.win_recon = self.viz.line(
                                        X=iters,
                                        Y=recon_losses,
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            xlabel='iteration',
                                            title='reconsturction loss',))
        else:
            self.win_recon = self.viz.line(
                                        X=iters,
                                        Y=recon_losses,
                                        env=self.viz_name+'_lines',
                                        win=self.win_recon,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            xlabel='iteration',
                                            title='reconsturction loss',))
        
        if self.win_beta is None:
            self.win_beta = self.viz.line(
                                        X=iters,
                                        Y=betas,
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            xlabel='iteration',
                                            title='beta',))
        else:
            self.win_beta = self.viz.line(
                                        X=iters,
                                        Y=betas,
                                        env=self.viz_name+'_lines',
                                        win=self.win_beta,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            xlabel='iteration',
                                            title='beta',))

        if self.win_kld is None:
            self.win_kld = self.viz.line(
                                        X=iters,
                                        Y=klds,
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend,
                                            xlabel='iteration',
                                            title='kl divergence',))
        else:
            self.win_kld = self.viz.line(
                                        X=iters,
                                        Y=klds,
                                        env=self.viz_name+'_lines',
                                        win=self.win_kld,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend,
                                            xlabel='iteration',
                                            title='kl divergence',))
                                            
        if self.save_output:
            output_dir = os.path.join(self.output_dir, str(self.global_iter))
            os.makedirs(output_dir, exist_ok=True)
            fig = plt.figure(figsize=(10, 10), dpi=300)
            plt.plot(iters_2, recon_losses_2)
            plt.xlabel('iteration')
            plt.title('reconsturction loss')
            fig.savefig(os.path.join(output_dir, 'graph_recon_loss.jpg'))

            fig = plt.figure(figsize=(10, 10), dpi=300)
            plt.plot(iters_2, betas_2)
            plt.xlabel('iteration')
            plt.title('beta')
            fig.savefig(os.path.join(output_dir, 'graph_beta.jpg'))

            fig = plt.figure(figsize=(10, 10), dpi=300)
            plt.plot(iters_2, klds_2)
            plt.legend(legend)
            plt.xlabel('iteration')
            plt.title('kl divergence')
            fig.savefig(os.path.join(output_dir, 'graph_kld.jpg'))
            
        self.net_mode(train=True)


    def _NLL_compute(self, repeat=100):
        """
        z: Gaussian
        repeat: input data repeat K times?
        """
        results = []
        count = 0
        self.net_mode(train=False)
        for inputs, targets in self.data_loader:
            count += 1
            inputs = Variable(cuda(inputs, self.use_cuda))
            inputs = inputs.expand(repeat,-1,-1,-1).contiguous()
            recon, zmu, zlogvar, z = self.net(inputs)
            # print("inputs shape: ", inputs.shape)
            z_eps = z - zmu
            diff = (inputs - recon)**2
            diff = diff.view(repeat,-1)
            log_p_x_z = -torch.sum(diff,1)
            #log_p_z
            log_p_z = -torch.sum(z**2/2,1)
            #log_q_z_x
            z_eps = z_eps.view(repeat,-1)
            log_q_z_x = -torch.sum(z_eps**2/2 + zlogvar/2, 1)

            weights = log_p_x_z+log_p_z-log_q_z_x
            # print('log_p_x_z: ',  log_p_x_z)
            # print('log_p_z: ', log_p_z)
            # print('log_q_z_x: ', log_q_z_x)
            
            nll_loss = -(torch.log(torch.mean(torch.exp(weights - weights.max())))+weights.max())
            results.append(nll_loss.item())
            if count % 1000 == 0:
                print("count: ", count)
            # if count >=5000:
            #     break
        print("log likelihood >>: ", np.mean(results))
        
    
    def _compute_active_unit(self):
        '''
        Fun: compute active unit
        '''
        self.net_mode(train=False)
        actUnit = _active_unit(self.net, self.data_loader,self.use_cuda)
        fileName = self.ckpt_dir.replace('checkpoints/', "")
        with open('active_unit.txt', "a") as fw_au:
            fw_au.write(fileName + ": {}\n".format(actUnit))
        print("Active unit: ", actUnit)
        
    
    def _test_elbo(self):
        fw_elbo = open("ELBO.txt","a")
        self.net_mode(train=False)
        elbo_list = []
        for x, targets in self.data_loader:
            x = Variable(cuda(x, self.use_cuda))
            x_recon, mu, logvar, z = self.net(x)
            recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            elbo = recon_loss.item() + total_kld.item()
            elbo_list.append(-elbo)
        elbo_avg = np.mean(elbo_list)
        fileName = self.ckpt_dir.replace('checkpoints/', "")
        fw_elbo.write(fileName + ": {}\n".format(elbo_avg))
        print("elbo average is: ", elbo_avg)
        fw_elbo.close()


    # def _test_model(self):
    #     print('******--testing model now--****')
    #     test_path = os.path.join('results', self.model_name)
    #     if not os.path.exists(test_path):
    #             os.makedirs(test_path)
        
    #     predict_path = os.path.join(test_path, 'testing/predict-25')
    #     ground_path = os.path.join(test_path, 'ground/ground_truth-25')
    #     image_path = [predict_path, ground_path]

    #     for path in image_path:
    #         if not os.path.exists(path):
    #             os.makedirs(path)
    #     ## evaluate the result
    #     self.net_mode(train=False)
    #     ids = 0
    #     batch = 0
    #     num_image = 5
    #     for x, targets in self.data_loader:
    #         batch += 1
    #         x = Variable(cuda(x, self.use_cuda))
    #         x_recon, _, _, _ = self.net(x)
    #         samples = F.sigmoid(x_recon).data
    #         batch_size = samples.size(0)
    #         # save_image(samples, filename=os.path.join(image_path[0], 'recontruct_{}.eps'.format(batch)),nrow=num_image,pad_value=1)
    #         # save_image(x, filename=os.path.join(image_path[1], 'recontruct_{}.eps'.format(batch)),nrow=num_image,pad_value=1)
    #         # if batch >= 10:
    #         #     break
    #         for b in range(batch_size):
    #             ids += 1
    #             save_image(samples[b,:,:,:], filename=os.path.join(image_path[0], 'predict_{}.jpg'.format(ids)))
    #             save_image(x[b,:,:,:], filename=os.path.join(image_path[1], 'ground_{}.jpg'.format(ids)))

    def viz_traverse(self, limit=3, inter=2/3, loc=-1):
        self.net_mode(train=False)
        import random

        decoder = self.net.decoder
        encoder = self.net.encoder
        interpolation = torch.arange(-limit, limit+0.1, inter)
        
        n_dsets = len(self.data_loader.dataset)
        rand_idx = random.randint(1, n_dsets-1)

        random_img = self.data_loader.dataset.__getitem__(rand_idx)
        random_img = Variable(cuda(random_img, self.use_cuda), volatile=True).unsqueeze(0)
        random_img_z = encoder(random_img)[:, :self.z_dim]

        random_z = Variable(cuda(torch.rand(1, self.z_dim), self.use_cuda), volatile=True)

        if self.dataset == 'dsprites':
            fixed_idx1 = 87040 # square
            fixed_idx2 = 332800 # ellipse
            fixed_idx3 = 578560 # heart

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)
            fixed_img1 = Variable(cuda(fixed_img1, self.use_cuda), volatile=True).unsqueeze(0)
            fixed_img_z1 = encoder(fixed_img1)[:, :self.z_dim]

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)
            fixed_img2 = Variable(cuda(fixed_img2, self.use_cuda), volatile=True).unsqueeze(0)
            fixed_img_z2 = encoder(fixed_img2)[:, :self.z_dim]

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)
            fixed_img3 = Variable(cuda(fixed_img3, self.use_cuda), volatile=True).unsqueeze(0)
            fixed_img_z3 = encoder(fixed_img3)[:, :self.z_dim]
            
            Z = {'fixed_square':fixed_img_z1, 'fixed_ellipse':fixed_img_z2,
                 'fixed_heart':fixed_img_z3, 'random_img':random_img_z}
        else:
            fixed_idx = 0
            fixed_img = self.data_loader.dataset.__getitem__(fixed_idx)
            fixed_img = Variable(cuda(fixed_img, self.use_cuda), volatile=True).unsqueeze(0)
            fixed_img_z = encoder(fixed_img)[:, :self.z_dim]
            Z = {'fixed_img':fixed_img_z, 'random_img':random_img_z, 'random_z':random_z}
            
        gifs = []
        for key in Z.keys():
            z_ori = Z[key]
            samples = []
            for row in range(self.z_dim):
                if loc != -1 and row != loc:
                    continue
                z = z_ori.clone()
                for val in interpolation:
                    z[:, row] = val  ## row is the z latent variable
                    sample = F.sigmoid(decoder(z)).data
                    samples.append(sample)
                    gifs.append(sample)
            samples = torch.cat(samples, dim=0).cpu()
            title = '{}_latent_traversal(iter:{})'.format(key, self.global_iter)
            
            if self.viz_on:
                self.viz.images(samples, env=self.viz_name+'_traverse',
                                opts=dict(title=title), nrow=len(interpolation))

        if self.save_output:
            output_dir = os.path.join(self.output_dir, str(self.global_iter))
            os.makedirs(output_dir, exist_ok=True)
            gifs = torch.cat(gifs)
            gifs = gifs.view(len(Z), self.z_dim, len(interpolation), self.nc, 64, 64).transpose(1, 2)
            for i, key in enumerate(Z.keys()):
                for j, val in enumerate(interpolation):
                    save_image(tensor=gifs[i][j].cpu(),
                               filename=os.path.join(output_dir, '{}_{}.jpg'.format(key, j)),
                               nrow=self.z_dim, pad_value=1)

                grid2gif(os.path.join(output_dir, key+'*.jpg'),
                         os.path.join(output_dir, key+'.gif'), delay=10)

        self.net_mode(train=True)
    
    def net_mode(self, train):
        if not isinstance(train, bool):
            raise('Only bool type is supported. True or False')

        if train:
            self.net.train()
        else:
            self.net.eval()

    def save_checkpoint(self, filename, silent=True):
        model_states = {'net':self.net.state_dict(),}
        optim_states = {'optim':self.optim.state_dict(),}
        win_states = {'recon':self.win_recon,
                      'beta': self.win_beta,
                      'kld':self.win_kld,
                      'mu':self.win_mu,
                      'var':self.win_var,}
        states = {'iter':self.global_iter,
                  'lambd': self.lambd,
                  'win_states':win_states,
                  'model_states':model_states,
                  'optim_states':optim_states}

        file_path = os.path.join(self.ckpt_dir, filename)
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))


    def load_checkpoint(self, filename):
        file_path = os.path.join(self.ckpt_dir, filename)
        print("checkpoint path: ", file_path)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            self.global_iter = checkpoint['iter']
            self.lambd = checkpoint['lambd']
            self.win_recon = checkpoint['win_states']['recon']
            self.win_kld = checkpoint['win_states']['kld']
            self.win_var = checkpoint['win_states']['var']
            self.win_mu = checkpoint['win_states']['mu']
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))

