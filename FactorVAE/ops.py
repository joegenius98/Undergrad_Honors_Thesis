"""ops.py"""

import torch
import torch.nn.functional as F

def recon_loss(x, x_recon, dataset_name, denoise):
    """
    Implements the reconstruction objective/loss term

    Parameters
    ----------
    x (torch.Tensor) - a minibatch
    x_recon (torch.Tensor) - reconstruction of minibatch
    dataset_name (str) - name of training dataset (e.g., 'dsprites', 'celeba') 
    denoise (bool): whether the goal is to reconstruct the augment (False) or to reconstruct
    the denoised or de-augmented image (True) upon the augmentation input
    """
    loss = None
    # gt stands for "ground truth"
    gt_x = x[::2].repeat_interleave(2, dim=0) if denoise else x

    if dataset_name == 'dsprites':
        n = x.size(0)
        loss = F.binary_cross_entropy_with_logits(x_recon, gt_x, reduction='sum').div(n)
    else:
        x_recon = torch.sigmoid(x_recon)
        loss = F.mse_loss(x_recon, gt_x, reduction='sum').div(n)

    return loss


def kl_divergence(mu, logvar):
    klds = -0.5*(1+logvar-mu**2-logvar.exp())

    total_kld = klds.sum(1).mean(0, keepdim=True)

    # without keepdim=True, dim_wise_kld is a 1D torch obj. instead of 2D
    dim_wise_kld = klds.mean(0)

    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dim_wise_kld, mean_kld


def permute_dims(z):
    assert z.dim() == 2

    B, _ = z.size()
    perm_z = []
    # z_j is the tensor of all the latent vals. of a particular dimension
    # z.split(1, 1) gets a tuple of one slice along dimension 1 in sequence
    # each tuple containing a tensor of shape (batch_size, 1)
    for z_j in z.split(1, 1):
        # random ordering of [0, 1, 2, ..., B - 1]
        perm = torch.randperm(B).to(z.device)
        # reorder z_j accr. to perm
        perm_z_j = z_j[perm]
        # each item in list is shape (batch_size, 1) to be later concatenated
        # along axis 1
        perm_z.append(perm_z_j)

    return torch.cat(perm_z, 1)
