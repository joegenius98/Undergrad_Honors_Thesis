"""main.py"""

import argparse

import numpy as np
import torch

# from solver_light import Solver
from solver_thesis import Solver
from utils import str2bool

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main(args):
    if args.cuda: torch.cuda.set_device(args.gpu)

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    net = Solver(args)

    if args.train:
        net.train()
    else:
        net.viz_traverse(args.limit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='toy Beta-VAE')

    # purpose
    parser.add_argument('--train', default=True,
                        type=str2bool, help='train or traverse')

    # seed
    parser.add_argument('--seed', default=1, type=int, help='random seed')

    # hardware-related
    parser.add_argument('--cuda', default=True,
                        type=str2bool, help='enable cuda')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')

    # training hyperparameters
    parser.add_argument('--max_iter', default=1e6, type=float,
                        help='maximum training iteration')
    parser.add_argument('--batch_size', default=64,
                        type=int, help='batch size')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='Adam optimizer beta1')
    parser.add_argument('--beta2', default=0.999,
                        type=float, help='Adam optimizer beta2')


    # parser.add_argument('--KL_loss', default=25,
    #                     type=float, help='KL_divergence')
    parser.add_argument('--step_val', default=0.15,
                        type=float, help='step value to increment C by in PID algorithm')
    parser.add_argument('--pid_fixed', default=False,
                        type=str2bool, help='if fixed PID or dynamic')
    parser.add_argument('--is_PID', default=True,
                        type=str2bool, help='if use pid or not')

    ## model
    parser.add_argument('--z_dim', default=10, type=int,
                        help='# of dimensions of the representation z')
    parser.add_argument('--model', default='H', type=str,
                        help='model proposed in Higgins et al. or Burgess et al. H/B')
    ## objective hyperparameters
    parser.add_argument('--objective', default='H', type=str,
                        help='beta-vae objective proposed in Higgins et al. or Burgess et al. H/B or honors thesis by Joseph')

    ### original beta-vae by Higgins et al.
    parser.add_argument('--beta', default=4, type=float,
                        help='beta parameter for KL-term in original beta-VAE')

    ### beta-TCVAE
    parser.add_argument('--beta_TC', default=4, type=float,
                        help='beta parameter for total correlation term in beta-TCVAE paper')
    #### total correlation 
    parser.add_argument('--C_tc_start', default=0, type=float,
                        help='start value of constraint term to subtract from total correlation')
    parser.add_argument('--C_tc_max', default=5, type=float,
                        help='upper bound of constraint term to subtract from total correlation')
    parser.add_argument('--C_tc_step_val', default=0.02, type=float,
                        help='increment value per 5000 iterations')

    ### understanding beta-vae by Burgess et al.
    parser.add_argument('--C_start', default=0, type=float,
                        help='start value of C for Burgess et al.\'s VAE')
    parser.add_argument('--gamma', default=1000, type=float,
                        help='gamma parameter for KL-term in understanding beta-VAE')
    
    ### ControlVAE by Shao et al.
    parser.add_argument('--C_max', default=25, type=float,
                        help='capacity parameter(C) of bottleneck channel')
    parser.add_argument('--C_stop_iter', default=1e5, type=float,
                        help='when to stop increasing the capacity')
    ## dataset
    parser.add_argument('--dset_dir', default='data',
                        type=str, help='dataset directory')
    parser.add_argument('--dataset', default='CelebA',
                        type=str, help='dataset name (choose among `dsprites`, `celeba`, or `3dchairs`)')
    parser.add_argument('--image_size', default=64, type=int,
                        help='image size. now only (64,64) is supported')
    parser.add_argument('--num_workers', default=4,
                        type=int, help='dataloader num_workers')

    ## visualziation
    parser.add_argument('--viz_on', default=True,
                        type=str2bool, help='enable visdom visualization')
    parser.add_argument('--viz_name', default='main',
                        type=str, help='visdom env name')
    
    ### traverse visualization limits
    parser.add_argument('--limit', default=3, type=float,
                        help='traverse limits')

    parser.add_argument('--viz_port', default=8097,
                        type=str, help='visdom port number')
    parser.add_argument('--gather_step', default=10000, type=int,
                        help='number of iterations after which data is gathered and displayed for visdom')
    # parser.add_argument('--display_step', default=10000, type=int,
    #                     help='number of iterations after which loss data is printed and visdom is updated')
    ### saving hyperparameter
    parser.add_argument('--save_output', default=True,
                        type=str2bool, help='whether to save traverse images and gif')
    parser.add_argument('--output_dir', default='outputs',
                        type=str, help='output directory')
    parser.add_argument('--save_step', default=10000, type=int,
                        help='number of iterations after which a checkpoint is saved')


    # checkpoints
    parser.add_argument('--ckpt_dir', default='checkpoints',
                        type=str, help='checkpoint directory')
    parser.add_argument('--ckpt_name', default='last', type=str,
                        help='checkpoint filename under `ckpt_dir` to load previous checkpoint')

    args = parser.parse_args()

    main(args)
