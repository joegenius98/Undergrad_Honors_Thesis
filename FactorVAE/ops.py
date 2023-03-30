"""ops.py"""

import torch
import torch.nn.functional as F


def recon_loss(x, x_recon):
    n = x.size(0)
    loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction='sum').div(n)
    return loss


def kl_divergence(mu, logvar):
    kld = -0.5*(1+logvar-mu**2-logvar.exp()).sum(1).mean(0, keepdim=True)
    return kld


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
