# kFactorVAE
kFactorVAE extends [FactorVAE](FactorVAE_README.md) with the k-Factor Similarity Loss
mentioned in the honors thesis, and this model is what was used in the experimental results.


## Datasets 
Instructions for the installation of datasets may be found in the [FactorVAE README](FactorVAE_README.md). 


## Reproducing Results

1. Make sure you have the Visdom server initialized. If you have not, you may refer 
to the overall [README](../README.md). 

2. Head over to the [`thesis_dsprites_scripts`](./thesis_dsprites_scripts) directory's `README` file.


If you want more details on how these subdirectories get constructed within these three folders,
carefully read through [`solver.py`](solver.py).


## Outputs of Results
For each shell script run:

1. The [`outputs`](./outputs/) will be populated with subdirectories, one per seed, for reconstructions and latent traversals.

2. The [`checkpoints`](./checkpoints/) directory will be populated with subdirectories, one per seed, of VAE model checkpoints.

3. The [`graphs`](./graphs/) directory will be populated with subdirectories, one per seed, containing information 
about training metrics of reconstruction error, KL divergence, k-factor similarity loss, discriminator accuracy, and the discriminator-estimated total correlation logged per certain amount of training iterations set by the corresponding user argument. You may make the [Matplotlib](https://matplotlib.org) plots of all these metrics
by executing [`make_graphs.py`](graphs/make_graphs.py).

4. The [`vis_logs`](./vis_logs/) directories will be populated with files that contain the data files needed
to reproduce Visdom results, one per seed. 

If you want any of these outputs to be turned off, you may either have to change the corresponding user 
argument or do it manually or create/modify a user argument to turn it off. Please refer to [`main.py`](main.py)
or run `python main.py -h` or `python main.py --help` for more info. on the user arguments. 



## Amount of Disk Space Required
Each model checkpoint in takes up around 51 or 52 MB. Each of my shell scripts in `thesis_dsprites_scripts` has 4 or 7 checkpoints per seed. That means each seed takes roughly between 204 to 364 MB
of space. There are 5 seeds total per shell script, which means that the amount of total space required
is roughly between **1.0 to 1.8 GB of checkpoints per shell script**. After the shell scripts run its 5 seeds, you will
see 5 subdirectories within the [`checkpoints`](./checkpoints/) directory, each containing 
either 4 or 7 checkpoints. The checkpoint frequency can be changed to tune the amount of space used.


## Cleaning up Outputs
If the [`outputs`](./outputs/), [`vis_logs`](./vis_logs), and/or the [`checkpoints`](./checkpoints/) directories ever grow too large, you may run the `group_seeds.sh` script inside one of those directories. 

For the [`graphs`](./graphs/) directory, run the Python script `combine_seeds.py` instead.
