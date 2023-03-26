import torch
from torchvision.transforms.functional import rotate 
# resize, pad, crop, hflip, vflip
import random 

def discrete_random_rotate(image):
    return rotate(image, random.choice([90, 180, 270]))


def augmented_batch(batch):
    """"
    Translates Joseph Lee's honors thesis idea with data augmentation --> dataloader implementation 

    Proccesses a batch of data sequentially (in the form) of a list
    to return a batch of pairs (1st image, 1st img. augmented, second image, 2nd img. augmented, etc.)
    
    Keyword arguments:
    batch -- a list of images, its length is batch_size
    Return: Tensor with shape (batch_size, 2 * num_channels, height, width)
    """

    # get image dimensions, assuming all images are the same size
    # and it is number-of-channels (nc) first
    nc, h, w = batch[0].shape[-3], batch[0].shape[-2], batch[0].shape[-1]
    batch_size = len(batch)

    # format: (batch_size, 2 for (original, augmented) * num_channels, height, width)
    # images_batch = torch.zeros((batch_size, 3, nc, h, w))
    images_batch = torch.zeros((batch_size * 2, nc, h, w))
    
    for i in range(batch_size):         
        first_image = batch[i]
        first_image_augmented = discrete_random_rotate(batch[i])

        images_batch[2*i, :, :, :] = first_image
        images_batch[2*i+1, :, :, :] = first_image_augmented

    return images_batch
