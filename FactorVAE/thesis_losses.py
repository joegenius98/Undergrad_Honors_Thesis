import torch

def k_factor_sim_loss_samples(latent_samples: torch.Tensor, k, idx_order=None):
    """
    Computes the k-factor consistency loss and contrastive loss for given latent vectors.
    
    The k-factor consistency loss encourages the latent representations of an image 
    and its augmentation to be similar for k factors. The contrastive loss encourages the latent 
    representations of an image and its augmentation to be different from the other image.
    
    Args:
        latent_samples (torch.Tensor): A PyTorch Tensor of shape (batch_size, number_of_latent_dimensions),
                                       containing the latent vectors for images, their augmentations, 
                                       and other randomly selected images. Its .size(0) must be a factor of 3

        k: how many factors to encourage to be similar for an image's representation and the representation
        of an image's augmentation

        idx_order: informs which k factors to select (as opposed to selecting just the first k indices) based
        on a strategy (e.g. sorting based on dimwise KL Div. values)


    Returns:
        tuple: A tuple containing two elements:
            - k_factor_consistency_loss (torch.Tensor): The mean of the sum of 2nd norms for
              the difference between an image's representation and its augmentation representation.

            (Not implemented for now:)
            - contrastive_loss (torch.Tensor): The mean contrastive loss, which encourages the latent
              representations of an image and its augmentation to be different from the other image.
    """
    if k is None or k == 0:
        return 0

    # assert latent_samples.size(0) % 3 == 0
    assert latent_samples.size(0) % 2 == 0

    # Separate the vectors into groups
    # images, their respective augmentations, and their respective contrastive other images
    # image_reprs = latent_samples[::3]
    # aug_reprs = latent_samples[1::3]
    # other_reprs = latent_samples[2::3]

    image_reprs_k, aug_reprs_k = None, None

    if idx_order is not None:
        image_reprs_k = latent_samples[::2, idx_order[:k]]
        aug_reprs_k = latent_samples[::2, idx_order[:k]]
    else:
        image_reprs_k = latent_samples[::2, :k]
        aug_reprs_k = latent_samples[1::2, :k]

    # k-factor consistency loss
    repr_diffs_k = image_reprs_k - aug_reprs_k 

    # 2nd norm.
    # repr_diff_norms_k = torch.norm(repr_diffs_k, p=2, dim=1)

    # squared error 
    repr_diff_norms_k = torch.sum(repr_diffs_k ** 2, dim = 1)

    # # k-factor contrastive loss
    # k_factor_contrastive_diffs = image_reprs[:, :k] - other_reprs[:, :k]
    # k_factor_contrastive_norms = torch.norm(k_factor_contrastive_diffs, p=2, dim=1)
    # k_factor_contrastive_loss = torch.mean(k_factor_contrastive_norms)
    # return k_factor_consistency_loss, k_factor_contrastive_loss

    # mean 2nd norm. or mean squared error
    return torch.mean(repr_diff_norms_k)


def k_factor_sim_losses_params(means: torch.Tensor, logvars: torch.Tensor, k):
    """Similar as above but except using the Gaussian distribution params.
    instead of the samples themselves
    
    Keyword arguments:
    means -- a (batch_size, z_dim) Tensor containing the encoder-predicted means of the inputs
    logsigmas -- a (batch_size, z_dim) Tensor containig the encoder-predicted log(std. dev.) of the inputs
    """
    if k is None or k == 0:
        return 0
    
    assert means.size(0) % 2 == 0 and logvars.size(0) % 2 == 0

    img_repr_means_k = means[::2, :k]
    aug_repr_means_k = means[1::2, :k]

    img_repr_logvars_k = logvars[::2, :k]
    aug_repr_logvars_k = logvars[1::2, :k]

    mean_diffs_k = img_repr_means_k - aug_repr_means_k
    logvar_diffs_k = img_repr_logvars_k - aug_repr_logvars_k

    mean_diff_norms_k = torch.norm(mean_diffs_k, p=2, dim=1)
    logvar_diff_norms_k = torch.norm(logvar_diffs_k, p=2, dim=1)

    return torch.mean(mean_diff_norms_k) + torch.mean(logvar_diff_norms_k)


    