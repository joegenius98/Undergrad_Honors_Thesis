"""main.py"""

import argparse
import numpy as np
import torch
import random 

from solver import Solver
from utils import str2bool
from augmentations import AUGMENT_DESCRIPTIONS

# promotes as much reproducibility as possible
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def set_seed(seed, cuda):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    


def main(args):
    set_seed(args.seed, args.cuda)

    if args.use_augment_dataloader:

        assert args.num_sim_factors and args.augment_factor
        assert args.num_sim_factors <= args.z_dim
        print(f"Using augmentation(s) {AUGMENT_DESCRIPTIONS[args.augment_choice - 1]} with:")
        print(f"k = {args.num_sim_factors}, a = {args.augment_factor}")
    
    if args.use_sort_strategy:
        print("Using sorting strategy for k-factor similarity loss")

    net = Solver(args)
    net.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Factor-VAE')

    parser.add_argument('--name', default='main', type=str, help='name of the experiment')
    parser.add_argument('--seed', default=1, type=int, help='torch and numpy random generator seed of the experiment')
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--max_iter', default=1e6, type=float, help='maximum training iteration')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')

    parser.add_argument('--z_dim', default=10, type=int, help='dimension of the representation z')
    parser.add_argument('--gamma', default=6.4, type=float, help='gamma hyperparameter')
    parser.add_argument('--lr_VAE', default=1e-4, type=float, help='learning rate of the VAE')
    parser.add_argument('--beta1_VAE', default=0.9, type=float, help='beta1 parameter of the Adam optimizer for the VAE')
    parser.add_argument('--beta2_VAE', default=0.999, type=float, help='beta2 parameter of the Adam optimizer for the VAE')
    parser.add_argument('--lr_D', default=1e-4, type=float, help='learning rate of the discriminator')
    parser.add_argument('--beta1_D', default=0.5, type=float, help='beta1 parameter of the Adam optimizer for the discriminator')
    parser.add_argument('--beta2_D', default=0.9, type=float, help='beta2 parameter of the Adam optimizer for the discriminator')
    
    #### contrastive loss hyperparameters
    parser.add_argument('--use_augment_dataloader', action='store_true', help='whether to load images + their augmentations per batch')
    parser.add_argument('--augment_choice', default=1, type=int, choices=range(1, len(AUGMENT_DESCRIPTIONS)+1), help=\
                        "\n".join(f"{i+1}. {AUGMENT_DESCRIPTIONS[i]}" for i in range(len(AUGMENT_DESCRIPTIONS)))
                        )
    parser.add_argument('--num_sim_factors', default=None, type=int,
                        help='number of factors to encourage to similar in value in image representation and its augmentation representation')
    parser.add_argument('--use_sort_strategy', action='store_true', help=\
                        '''
                        instead of selecting the first k indices, where k = num_sim_factors,
                        you can choose tackle the factors that have a KL divergence above 0.5 (may put a threshold arg. later)
                        in increasing order first, and then if k is greater than the number of latent vars.
                        whose KL divergences are above 0.5, then tackle the ones below that threshold in decreasing order
                        ''')
    parser.add_argument('--augment_factor', default=None, type=float,
                        help='factor of the 2nd norm of diff. btwn. image representation and its augmentation representatoin')

    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    parser.add_argument('--dataset', default='CelebA', type=str, help='dataset name')
    parser.add_argument('--image_size', default=64, type=int, help='image size. now only (64,64) is supported')
    parser.add_argument('--num_workers', default=2, type=int, help='dataloader num_workers')

    parser.add_argument('--viz_on', default=True, type=str2bool, help='enable visdom visualization')
    parser.add_argument('--viz_port', default=8097, type=int, help='visdom port number')
    parser.add_argument('--viz_ll_iter', default=1000, type=int, help='visdom line data logging iter')
    parser.add_argument('--viz_la_iter', default=5000, type=int, help='visdom line data applying iter')
    parser.add_argument('--viz_ra_iter', default=10000, type=int, help='visdom recon image applying iter')
    parser.add_argument('--viz_ta_iter', default=10000, type=int, help='visdom traverse applying iter')

    parser.add_argument('--print_iter', default=500, type=int, help='print losses iter')

    parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory')
    parser.add_argument('--ckpt_load', default=None, type=str, help='checkpoint name to load')
    parser.add_argument('--ckpt_save_iter', default=10000, type=int, help='checkpoint save iter')

    parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')
    parser.add_argument('--output_save', default=True, type=str2bool, help='whether to save traverse results')

    args = parser.parse_args()

    main(args)
