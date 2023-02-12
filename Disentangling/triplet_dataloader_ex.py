import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import rotate

class ImageDataset(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index]

def collate_fn(batch):
    print("Here is the batch in `collate_fn`")
    print(batch)
    batch_size = len(batch)
    print(f"Batch size: {batch_size}")
    images_batch = torch.zeros((batch_size-1, 3, 1,  3, 3))
    
    for i in range(batch_size - 1): # we'll need a batch size of at least 2 (need at least two randomly selected images)
        first_image_index = i
        second_image_index = i + 1 # can be i and i + 1 if we toggle shuffle = True, so that it's random
        first_image = batch[first_image_index]
        second_image = batch[second_image_index]
        first_image_augmented = rotate(first_image, 180)

        images_batch[i, 0, :, :, :] = first_image
        images_batch[i, 1, :, :, :] = first_image_augmented
        images_batch[i, 2, :, :, :] = second_image
        
    return images_batch

images = torch.randn(5, 1, 3, 3)
dataset = ImageDataset(images)
dataloader = DataLoader(dataset, batch_size=10, collate_fn=collate_fn, shuffle=True)

for batch in dataloader:
    print(batch)

