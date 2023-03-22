import torch
import torch.nn.functional as F
import math

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


def total_corr(z, z_mean, z_logvar):
    # provided by 
    # https://github.com/amir-abdi/disentanglement-pytorch/blob/master/models/betatcvae.py

    """Estimate of total correlation on a batch.
    Borrowed from https://github.com/google-research/disentanglement_lib/
    Args:
      z: [batch_size, num_latents]-tensor with sampled representation.
      z_mean: [batch_size, num_latents]-tensor with mean of the encoder.
      z_logvar: [batch_size, num_latents]-tensor with log variance of the encoder.
    Returns:
      Total correlation estimated on a batch.
    """
    # Compute log(q(z(x_j)|x_i)) for every sample in the batch, which is a
    # tensor of size [batch_size, batch_size, num_latents]. In the following
    # comments, [batch_size, batch_size, num_latents] are indexed by [j, i, l].
    log_qz_prob = _gaussian_log_density(z.unsqueeze(dim=1),
                                       z_mean.unsqueeze(dim=0),
                                       z_logvar.unsqueeze(dim=0))

    # Compute log prod_l p(z(x_j)_l) = sum_l(log(sum_i(q(z(z_j)_l|x_i)))
    # + constant) for each sample in the batch, which is a vector of size
    # [batch_size,].
    log_qz_product = log_qz_prob.exp().sum(dim=1, keepdim=False).log().sum(dim=1, keepdim=False)

    # Compute log(q(z(x_j))) as log(sum_i(q(z(x_j)|x_i))) + constant =
    # log(sum_i(prod_l q(z(x_j)_l|x_i))) + constant.
    log_qz = log_qz_prob.sum(dim=2, keepdim=False).exp().sum(dim=1, keepdim=False).log()

    return torch.abs((log_qz - log_qz_product).mean())


def _gaussian_log_density(samples, mean, log_var):
    # provided by 
    # https://github.com/amir-abdi/disentanglement-pytorch/blob/master/models/betatcvae.py

    """ Estimate the log density of a Gaussian distribution
    Borrowed from https://github.com/google-research/disentanglement_lib/
    :param samples: batched samples of the Gaussian densities with mean=mean and log of variance = log_var
    :param mean: batched means of Gaussian densities
    :param log_var: batches means of log_vars
    :return:
    """
    pi = torch.tensor(math.pi, requires_grad=False)
    normalization = torch.log(2. * pi)
    inv_sigma = torch.exp(-log_var)
    tmp = samples - mean
    return -0.5 * (tmp * tmp * inv_sigma + log_var + normalization)


def contrastive_losses(latent_samples: torch.Tensor, k):
    """
    Computes the k-factor consistency loss and contrastive loss for given latent vectors.
    
    The k-factor consistency loss encourages the latent representations of an image 
    and its augmentation to be similar for k factors. The contrastive loss encourages the latent 
    representations of an image and its augmentation to be different from the other image.
    
    Args:
        latent_samples (torch.Tensor): A PyTorch Tensor of shape (batch_size, number_of_latent_dimensions),
                                       containing the latent vectors for images, their augmentations, 
                                       and other randomly selected images.
        k: how many factors to encourage to be similar for an image's representation and the representation
        of an image's augmentation

    Returns:
        tuple: A tuple containing two elements:
            - k_factor_consistency_loss (torch.Tensor): The mean of the sum of 2nd norms for
              the difference between an image's representation and its augmentation representation.
            - contrastive_loss (torch.Tensor): The mean contrastive loss, which encourages the latent
              representations of an image and its augmentation to be different from the other image.
    """
    # Separate the vectors into groups
    # images, their respective augmentations, and their respective contrastive other images
    image_reprs = latent_samples[::3]
    aug_reprs = latent_samples[1::3]
    other_reprs = latent_samples[2::3]

    # k-factor consistency loss
    k_factor_diffs = image_reprs[:, :k] - aug_reprs[:, :k]
    k_factor_diff_norms = torch.norm(k_factor_diffs, p=2, dim=1)
    k_factor_consistency_loss = torch.mean(k_factor_diff_norms)

    # contrastive loss
    # dot product tensors, each mapping a relationship between one group to another
    # note that the entire representation vector is considered; maybe could change to just k factors later
    img_aug_scores = torch.sum(image_reprs * aug_reprs, dim=1) # shape (batch_size // 3,)
    img_other_scores = torch.sum(image_reprs * other_reprs, dim=1)
    aug_other_scores = torch.sum(aug_reprs * other_reprs, dim=1)

    # img, aug, other altogether
    logsumexp_scores_per_triplet = torch.logsumexp(torch.stack([img_aug_scores, img_other_scores, aug_other_scores]), dim=0)
    # img, aug pair
    print(torch.exp(img_aug_scores))
    logexp_scores_per_pair = torch.log(torch.exp(img_aug_scores))
    contrastive_loss = torch.mean(logsumexp_scores_per_triplet) - torch.mean(logexp_scores_per_pair)

    return k_factor_consistency_loss, contrastive_loss




def kl_divergence(mu, logvar):
    """
    Calculates the Kullback-Leibler (KL) divergence approximation between 
    Gaussian distribution of (mu, logvar) and the unit multivariate Gaussian distribution

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

