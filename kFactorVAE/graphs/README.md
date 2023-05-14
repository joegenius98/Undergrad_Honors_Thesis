# About
This directory contains the CSV files needed to plot training metrics, notably:

1. Reconstruction loss

2. KL divergences
- minibatch average sum of all the latent dimension-wise KL divergences
- minibatch average for each latent dimension
- average across latent dimensions and minibatch (global average I suppose)

3. k-Factor mean squared error similarity loss

4. Total correlation estimate from the discriminator

5. Discriminator accuracy

You can automatically generate the plots with [`make_graphs.py`](./make_graphs.py)

# How to Run `make_graphs.py`

```python make_graphs.py [folder name]```

<br><br><br><br>

The instructions below indicate a previous attempt I did with Google Sheets. It turned out to be a significant waste of time.
# ~~How to Make Graphs from the Data~~

~~1. Copy/paste the recon. loss data, discriminator accuracy, total corr. (VAE), and k-factor similarity loss Visdom window data from 5 seeds.~~
    - ~~I recommend putting all the data in Google Sheets.~~
    
~~2. Choose the median MIG-scored KL divergence graph~~

~~3. Take the average amongst the 5 seeds and put that as its own column. Get also the lower and upper bounds based on the standard deviation.~~

~~4. Create a folder indicating the hyperparameters chosen. For example, if you chose a rotation augmentation, with `num_sim_factors` as 3 and `augment_factor` as 4, then you would name the folder `rot_k3_a4`.~~

~~6. Put all the CSV files from Google Sheets into this folder.~~
