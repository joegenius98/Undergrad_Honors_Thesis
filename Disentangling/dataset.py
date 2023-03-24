"""dataset.py"""

import os
import numpy as np
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

# from augmentations import DSPRITE_AUGMENTATIONS
from augmentations import translate_shape

class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path = self.imgs[index][0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img


class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


def triplet_batch_dSprites(batch):
    """"
    Translates Joseph Lee's honors thesis idea with data augmentation --> dataloader implementation 

    Proccesses a batch of data sequentially (in the form) of a list
    to return a batch of triplets (one image, its augmentation, and another image)
    or a (one image, its augmentation) pair 
    
    Keyword arguments:
    batch -- a list of images, its length is batch_size
    Return: Tensor with shape (batch_size, 3 or 2, num_channels, height, width)
    """

    # get image dimensions, assuming all images are the same size
    # and it is number-of-channels (nc) first
    nc, h, w = batch[0].shape[-3], batch[0].shape[-2], batch[0].shape[-1]
    batch_size = len(batch)

    """to return batches of 3 (original image, augmented original, and another image)
    implies we need at least two separate images to work off of""" 
    if batch_size >= 2:
        # format: (batch_size, 3 for (original, augmented, another), num_channels, height, width)
        # images_batch = torch.zeros((batch_size, 3, nc, h, w))
        images_batch = torch.zeros((batch_size * 3, nc, h, w))
        
        for i in range(batch_size - 1):         
            first_image_index = i
            second_image_index = i + 1 # can be i and i + 1 if we toggle shuffle = True, so that it's random
            first_image = batch[first_image_index]
            second_image = batch[second_image_index]
            # first_image_augmented = random.choice(DSPRITE_AUGMENTATIONS)(first_image)
            first_image_augmented = translate_shape(first_image)

            images_batch[3*i, :, :, :] = first_image
            images_batch[3*i+1, :, :, :] = first_image_augmented
            images_batch[3*i+2, :, :, :] = second_image

            # images_batch[i, 0, :, :, :] = first_image
            # images_batch[i, 1, :, :, :] = first_image_augmented
            # images_batch[i, 2, :, :, :] = second_image
        
        # the "second_image" here is the starting image of the batch
        # This is to address the out of bounds error
        last_idx = batch_size-1
        images_batch[3*last_idx, :, :, :] = batch[batch_size-1]
        # images_batch[3*last_idx+1, :, :, :] = random.choice(DSPRITE_AUGMENTATIONS)(batch[batch_size-1])
        images_batch[3*last_idx+1, :, :, :] = translate_shape(batch[batch_size-1])
        images_batch[3*last_idx+2, :, :, :] = batch[0] 


    # otherwise, just create an (original image, augmented original) pair
    else:
        assert batch_size == 1
        # images_batch = torch.zeros((batch_size, 2, nc, h, w))
        images_batch = torch.zeros((batch_size * 2, nc, h, w))
        images_batch[0, :, :, :] = batch[0]
        # images_batch[1, :, :, :] = random.choice(DSPRITE_AUGMENTATIONS)(batch[batch_size-1])
        images_batch[1, :, :, :] = translate_shape(batch[batch_size-1])
        
        
    return images_batch
    # return images_batch.view((batch_size * 3, nc, h, w))



def return_data(args):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    assert image_size == 64, 'currently only image size of 64 is supported'

    ### TODO: implement AUGMENTATIONS for 3dchairs/celeba

    # if name.lower() == '3dchairs':
    #     root = os.path.join(dset_dir, '3DChairs')
    #     transform = transforms.Compose([
    #         transforms.Resize((image_size, image_size)),
    #         transforms.ToTensor(), ])
    #     train_kwargs = {'root': root, 'transform': transform}
    #     dset = CustomImageFolder

    # elif name.lower() == 'celeba':
    #     root = os.path.join(dset_dir, 'CelebA')
    #     transform = transforms.Compose([
    #         transforms.Resize((image_size, image_size)),
    #         transforms.ToTensor(), ])
    #     train_kwargs = {'root': root, 'transform': transform}
    #     dset = CustomImageFolder

    # elif
    if name.lower() == 'dsprites':
        root = os.path.join(
            dset_dir, 'dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        if not os.path.exists(root):
            import subprocess
            print('Now download dsprites-dataset')
            subprocess.call(['./download_dsprites.sh'])
            print('Finished')
        data = np.load(root, encoding='bytes')

        # .unsqueeze(1) allows us to have a channel size of 1
        # data.shape goes from (737280, 64, 64) to (737280, 1, 64, 64)
        data = torch.from_numpy(data['imgs']).unsqueeze(1).float()
        train_kwargs = {'data_tensor': data}
        dset = CustomTensorDataset

    else:
        raise NotImplementedError

    train_data = dset(**train_kwargs)
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                            #   collate_fn=triplet_batch_dSprites,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)

    data_loader = train_loader

    return data_loader


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(), ])

    dset = CustomImageFolder('data/CelebA', transform)
    loader = DataLoader(dset,
                        batch_size=32,
                        shuffle=True,
                        num_workers=1,
                        pin_memory=False,
                        drop_last=True)

    images1 = iter(loader).next()
    import ipdb
    ipdb.set_trace()
