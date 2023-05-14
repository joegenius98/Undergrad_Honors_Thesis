"""This file obtains a latent traversal based on selected latent dimension indices and
a substring found in one of the files found in a directory with latent var. pictures, each
containing a snapshot of all the latent dim

Code is borrowed from `combine_gif.py` in `plot_fig/Disentangle

Guidance: 
`python get_latent_traversal.py [dir. with latent traversal snapshot imgs.] [substr. of files] [latent index] [latent index] ...
each [latent index] must start from 1 INSTEAD OF starting from 0

e.g. `python get_latent_traversal.py ./fVAE_k1_af2/seed5/700000 fixed_square 2 4 5 1 10`

Each latent traversal snapshot img. is basically like:


            ------------------       ------------------
            |z_1 = [const. val]| ... |z_n = [const. val]|
            ------------------       ------------------

 and the idea to basically grab each index (from 1 to n inclusive) across all these images,
 and have the final result be so that each row is like:


                 ----------       ---------
                |z_x = -1.5| ... |z_x = 1.5|
                 ----------       ---------
                 ----------       ---------
                |z_y = -1.5| ... |z_y = 1.5|
                 ----------       ---------
                    .               .   
                    .               .
                    .               .
                 ----------       ---------
                |z_n = -1.5| ... |z_n = 1.5|
                 ----------       ---------

where x, y, ..., n are the chosen latent dimension indices from user arguments
"""


import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image
import re
import sys
from pathlib import Path

def atoi(text):
    return int(text) if text.isdigit() else text


# sort lexiographically, but respecting number ordering too
def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def combineImages(base_fp, name_initial, indices):
    """Obtains and saves a latent traversal of multiple latent variables as specified in `indices`
    
    Keyword arguments:
    base_fp -- the directory to look in for latent variable snapshots
    name_initial - the shared substring in several files in `base_fp`
        - e.g. in dSprites, "fixed_ellipse" is a valid substr. that represents a randomly selected
        ellipse dSprite image in which its latent representation was being traversed dimension-wise
    indices - the list of latent dimensions to capture in the final latent traversal image

    returns: the final latent traversal images
    """
    
    padding = 2
    images = [np.rollaxis(np.asarray(Image.open(f))[padding:-padding, padding:-padding, :], 2, 0) / 255.0
              for f in sorted((str(fp) for fp in base_fp.glob(f'{name_initial}_[0-9].jpg')), key=natural_keys)
                              + sorted((str(fp) for fp in base_fp.glob(f'{name_initial}_[0-9][0-9].jpg')), key=natural_keys)]
    width = images[0].shape[1] + padding
    steps = len(images)
    allt = []
    for index in indices:
        for image in images:
            sliced_img = image[:, :, (index * width):((index + 1) * width - padding)]
            # print(sliced_img.shape)
            allt.append(sliced_img)

    save_image(tensor=torch.tensor(np.stack(allt)), fp=base_fp/f'{name_initial}_combined.jpg', nrow=steps, pad_value=1)


if __name__ == '__main__':
    assert len(sys.argv[1:]) >= 3
    
    base_fp = Path(__file__).parent / sys.argv[1]
    assert base_fp.exists(), "Please insert relative directory to search in for latent traversal snapshot img. files"

    # the third user argument should be found as a substring in the name of several files,
    # depending on the latent dimensionality, in `base_fp`
    assert any(sys.argv[2] in str(fp) for fp in base_fp.glob('*')), "2nd argument should be a substring of several files"
    
    for arg in sys.argv[3:]:
        assert arg.isdigit(), "3rd argument and above should be the set of latent dim. indices, each index starting from 1"
    

    combineImages(base_fp, sys.argv[2], [int(x) - 1 for x in sys.argv[3:]])
    # combineImages('fixed_ellipse', [0,1,2,3,4,5,6,7,8,9])
    # combineImages('fixed_ellipse', [5, 1, 2, 6, 3])
