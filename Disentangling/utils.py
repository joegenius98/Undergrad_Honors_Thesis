"""utils.py"""

import argparse
import subprocess

# import torch
# import torch.nn as nn
# from torch.autograd import Variable


def cuda(pytorch_obj, uses_cuda):
    """pytorch_obj could be a Tensor or a neural net from nn.Module, for ex."""
    return pytorch_obj.cuda() if uses_cuda else pytorch_obj


def str2bool(v):
    # codes from : stackover

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def where(cond, x, y):
    """Do same operation as np.where

    code from:
        //discuss.pytorch.org/
    """
    cond = cond.float()
    return (cond*x) + ((1-cond)*y)


def grid2gif(image_str, output_gif, delay=100):
    """Make GIF from images.

    code from:
        //stackoverflow.com/
    """
    subprocess.call(f'convert -delay {delay} -loop 0 {image_str} {output_gif}', shell = True)
    # str1 = 'convert -delay '+str(delay)+' -loop 0 ' + \
    #     image_str + ' ' + output_gif
    # subprocess.call(str1, shell=True)
