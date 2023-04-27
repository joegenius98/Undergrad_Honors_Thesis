import torch
from torchvision.transforms.functional import rotate, resize, pad, crop, hflip, vflip
import random

"""
Augmentations implementation, mostly designed for/inspired by dSprites

Parameters
----------
image: a PyTorch tensor defined by (# channels, height, width) shape
"""
AUGMENT_DESCRIPTIONS = ['Discrete rotation counter-clockwise (90, 180, 270 degs.)',
                        'Translate dSprite randomly in x/y positions',
                        'Horizontal flip',
                        'Vertical flip']


# 1. rotation
def continuous_random_rotate(image):
    return rotate(image, random.randrange(0, 360))

def discrete_random_rotate(image):
    return rotate(image, random.choice([90, 180, 270]))

# 2. scaling

# for any general image dataset, but note it might be good to consider specifics, like I did for dSprites
def shrink_and_pad(image):
    scale_factor = random.uniform(0.5, 0.9)  # Adjust the range according to your needs
    new_width = int(image.shape[2] * scale_factor)
    new_height = int(image.shape[1] * scale_factor)
    resized_image = resize(image, (new_height, new_width))

    # Calculate padding values to maintain the original dimensions
    pad_left_amt = (image.shape[2] - new_width) // 2
    pad_top_amt = (image.shape[1] - new_height) // 2
    pad_right_amt = image.shape[2] - new_width - pad_left_amt
    pad_bottom_amt = image.shape[1] - new_height - pad_top_amt

    # Pad the resized image with black pixels
    # default fill value is 0
    padded_image = pad(resized_image, (pad_left_amt, pad_top_amt, pad_right_amt, pad_bottom_amt))

    return padded_image

# for dSprites, we want to preserve position while scaling the shape

def get_shape_bounds(image):
    """helper method to obtain bounding box of shape sprite"""
    white_pixels = image.nonzero(as_tuple=True)
    y_min, y_max = white_pixels[1].min(), white_pixels[1].max()
    x_min, x_max = white_pixels[2].min(), white_pixels[2].max()
    return x_min, x_max, y_min, y_max

def shrink_shape_and_pad(image):
    """shrinks a shape sprite while maintaining position
    and returns the corresponding image with the same dimensions as `image`"""
    ## Calculations
    x1, x2, y1, y2 = get_shape_bounds(image)
    # bounding box dimensions of sprite/shape
    shape_width, shape_height = x2 - x1 + 1, y2 - y1 + 1

    # scale_factor = random.uniform(0.6, 1.2) <-- cannot do larger than 1.0 since I'm working with a bounding box
    scale_factor = random.uniform(0.6, 0.9)
    # print("Scale factor: " + str(scale_factor))
    while scale_factor == 1.0:
        scale_factor = random.uniform(0.6, 0.9)

    new_width, new_height = int(shape_width * scale_factor), int(shape_height * scale_factor)

    # compute padding amounts in 4 directions
    left_pad_amt = (shape_width - new_width) // 2
    top_pad_amt = (shape_height - new_height) // 2
    right_pad_amt = shape_width - left_pad_amt - new_width
    bottom_pad_amt = shape_height - top_pad_amt - new_height 


    ## Applying our calculations 
    # crop out and resize the shape sprite itself
    rescaled_shape = resize(crop(image, left=x1, top=y1, width=shape_width, height=shape_height), (new_height, new_width))

    # applying padding to preserve center spot of shape
    padded_rescaled_shape = pad(rescaled_shape, (left_pad_amt, top_pad_amt, right_pad_amt, bottom_pad_amt))

    to_ret = torch.clone(image)

    to_ret[0, y1:y2+1, x1:x2+1] = padded_rescaled_shape
    return to_ret



# 3. Translation - dSprite-specific augmentation
def get_max_shift_amts(image):
    x_min, x_max, y_min, y_max = get_shape_bounds(image)
    max_shift_left = x_min
    max_shift_right = image.size(2) - 1 - x_max
    max_shift_up = y_min
    max_shift_down = image.size(1) - 1 - y_max
    return max_shift_left, max_shift_right, max_shift_up, max_shift_down


def translate_shape(image):
    """translates a shape sprite by a random, in-bound amount in the x and y directions
    and returns the corresponding image with the same dimensions as `image`"""
    ## Calculations
    x1, x2, y1, y2 = get_shape_bounds(image)
    msl, msr, msu, msd = get_max_shift_amts(image)

    # to avoid choosing 0 as a potential shift value
    shift_x = random.randrange(-msl, msr + 1)
    while shift_x == 0:
        shift_x = random.randrange(-msl, msr + 1)
    
    shift_y = random.randrange(-msu, msd + 1)
    while shift_y == 0:
        shift_y = random.randrange(-msu, msd + 1)

    ## Applying calculations 
    to_ret = torch.clone(image)

    # get shape, clear out original spot, and place shape in translated spot

    # this way, the 0 assignment afterwards doesn't affect shape_bbox too to become all 0s
    shape_bbox = torch.clone(to_ret[0, y1:y2+1, x1:x2+1])

    to_ret[0, y1:y2+1, x1:x2+1] = 0
    to_ret[0 , y1+shift_y : y2+1+shift_y , x1+shift_x : x2+1+shift_x] = shape_bbox

    return to_ret


# 4. Flipping (horizontal/vertical) is built-into PyTorch
def horizontal_flip(image):
    return hflip(image)

def vertical_flip(image):
    return vflip(image)

# DSPRITE_AUGMENTATIONS = [discrete_random_rotate, continuous_random_rotate, shrink_shape_and_pad, translate_shape, hflip, vflip]
# TODO: include maybe more in AUGMENTATIONS, 
# e.g. Gaussian blurring and other standard self-supervised learning augmentations
# AUGMENTATIONS = [discrete_random_rotate, shrink_and_pad, hflip, vflip]
ARG_TO_AUGMENTATION = {1: discrete_random_rotate, 2: translate_shape, 3: horizontal_flip, 4: vertical_flip}
