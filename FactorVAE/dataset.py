"""dataset.py"""

import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
# from augmentations import discrete_random_rotate, translate_shape, horizontal_flip, vertical_flip
from augmentations import ARG_TO_AUGMENTATION

# arg_to_augmentation = {1: discrete_random_rotate, 2: translate_shape, 3: horizontal_flip, 4: vertical_flip}
chosen_augmentation = None


def is_power_of_2(num):
    return ((num & (num - 1)) == 0) and num != 0


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)
        self.indices = range(len(self))

    def __getitem__(self, index1):
        index2 = random.choice(self.indices)

        path1 = self.imgs[index1][0]
        path2 = self.imgs[index2][0]
        img1 = self.loader(path1)
        img2 = self.loader(path2)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2


class CustomTensorDataset(Dataset):
    def __init__(self, data_tensor, transform=None):
        self.data_tensor = data_tensor
        self.transform = transform
        self.indices = range(len(self))

    def __getitem__(self, index1):
        index2 = random.choice(self.indices)

        img1 = self.data_tensor[index1]
        img2 = self.data_tensor[index2]
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2

    def __len__(self):
        return self.data_tensor.size(0)


class ThesisTensorDataset(Dataset):
    """
    Helper dataset so that each minibatch of data
    will be a tuple of two items of equal length:
    one being a sequence of [img1, img1_augmented, img2, img2_augmented, ...]
    and another being [random_img1, random_img2, ...]
    """
    def __init__(self, data_tensor, transform=None):
        self.data_tensor = data_tensor
        self.transform = transform
        self.indices = range(len(self))

    def __getitem__(self, index1):
        index2 = random.choice(self.indices)
        while index2 == index1: 
            index2 = random.choice(self.indices)

        index3 = random.choice(self.indices)
        while index3 == index2 or index3 == index1: 
            index3 = random.choice(self.indices)

        img1 = self.data_tensor[index1]
        img2 = self.data_tensor[index2]
        img3 = self.data_tensor[index3]

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        return img1, img2, img3

    def __len__(self):
        return self.data_tensor.size(0)


class TensorDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


class ThesisImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super(ThesisImageFolder, self).__init__(root, transform)
        self.indices = range(len(self))

    def __getitem__(self, index1):
        index2 = random.choice(self.indices)
        while index2 == index1: 
            index2 = random.choice(self.indices)

        index3 = random.choice(self.indices)
        while index3 == index2 or index3 == index1: 
            index3 = random.choice(self.indices)


        path1 = self.imgs[index1][0]
        path2 = self.imgs[index2][0]
        path3 = self.imgs[index3][0]

        img1 = self.loader(path1)
        img2 = self.loader(path2)
        img3 = self.loader(path3)

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        return img1, img2, img3



def augmented_batch(batch):
    """"
    Proccesses a batch of data sequentially (in the form) of an iterable
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
        # first_image_augmented = discrete_random_rotate(batch[i])
        first_image_augmented = chosen_augmentation(batch[i])

        images_batch[2*i, :, :, :] = first_image
        images_batch[2*i+1, :, :, :] = first_image_augmented

    return images_batch


def augmented_batch_2tuple(lst_3_tup_data_pts):
    """
    Processes a list of tuples, each with size 2 (two randomly selected data points),
    and returns a tuple of size 2, one with the augmented batch with the first elements
    and the other with the augmented batch with the second elements.
    """
    # zip does basically does the equivalent as the below commented out lines of code
    # data_pts_to_augment = tuple(tup[0] for tup in lst_3_tup_data_pts)
    # random_datapts_1 = tuple(tup[1] for tup in lst_3_tup_data_pts])
    # random_datapts_2 = tuple(tup[2] for tup in lst_3_tup_data_pts)
    data_pts_to_augment, random_datapts_1, random_datapts_2 = zip(*lst_3_tup_data_pts)

    aug_batch = augmented_batch(data_pts_to_augment)
    rand_batch = torch.stack(random_datapts_1 + random_datapts_2)

    assert len(aug_batch) == len(rand_batch)

    return aug_batch, rand_batch


def return_data(args):
    name = args.dataset
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    assert image_size == 64, 'currently only image size of 64 is supported'

    # .toTensor() shifts (height, width, num_channels) --> (num_channels, height, width)
    # and re-scales RGB values to be [0, 1]. 
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),])

    if name.lower() == 'celeba':
        root = os.path.join(dset_dir, 'CelebA')
        train_kwargs = {'root':root, 'transform':transform}
        dset = ThesisImageFolder if args.use_augment_dataloader else CustomImageFolder
    elif name.lower() == '3dchairs':
        root = os.path.join(dset_dir, '3DChairs')
        train_kwargs = {'root':root, 'transform':transform}
        dset = ThesisImageFolder if args.use_augment_dataloader else CustomImageFolder
    elif name.lower() == 'dsprites':
        root = os.path.join(dset_dir, 'dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        data = np.load(root, encoding='latin1')
        data = torch.from_numpy(data['imgs']).unsqueeze(1).float()
        train_kwargs = {'data_tensor':data}
        dset = ThesisTensorDataset if args.use_augment_dataloader else CustomTensorDataset
    else:
        raise NotImplementedError
    
    if args.use_augment_dataloader:
        global chosen_augmentation
        assert type(args.augment_choice) == int
        chosen_augmentation = ARG_TO_AUGMENTATION[args.augment_choice]


    train_data = dset(**train_kwargs)
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              collate_fn=augmented_batch_2tuple if args.use_augment_dataloader else None,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)

    data_loader = train_loader
    return data_loader
