# How to Make Graphs from the Data

1. Copy/paste the recon. loss data, discriminator accuracy, total corr. (VAE), and k-factor similarity loss window data from 5 seeds.
    - I recommend putting all the data in Google Sheets.
2. Choose the median MIG-scored KL divergence graph
3. Take the average amongst the 5 seeds and put that as its own column. Get also the lower and upper bounds based on the standard deviation.
4. Create a folder indicating the hyperparameters chosen. For example, if you chose a rotation augmentation, with `num_sim_factors` as 3
and `augment_factor` as 4, then you would name the folder `rot_k3_a4`. 
6. Put all the CSV files from Google Sheets into this folder.

# How to Run `make_graphs.py`

```python make_graphs.py [folder name]```