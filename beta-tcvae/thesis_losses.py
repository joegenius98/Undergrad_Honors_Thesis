import torch

def k_factor_sim_loss(latent_samples: torch.Tensor, k):
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

    Returns:
        tuple: A tuple containing two elements:
            - k_factor_consistency_loss (torch.Tensor): The mean of the sum of 2nd norms for
              the difference between an image's representation and its augmentation representation.
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

    image_reprs = latent_samples[::2]
    aug_reprs = latent_samples[1::2]

    # k-factor consistency loss
    k_factor_consistency_diffs = image_reprs[:, :k] - aug_reprs[:, :k]
    k_factor_diff_consistency_norms = torch.norm(k_factor_consistency_diffs, p=2, dim=1)
    k_factor_consistency_loss = torch.mean(k_factor_diff_consistency_norms)

    # # k-factor contrastive loss
    # k_factor_contrastive_diffs = image_reprs[:, :k] - other_reprs[:, :k]
    # k_factor_contrastive_norms = torch.norm(k_factor_contrastive_diffs, p=2, dim=1)
    # k_factor_contrastive_loss = torch.mean(k_factor_contrastive_norms)

    # return k_factor_consistency_loss, k_factor_contrastive_loss
    return k_factor_consistency_loss

