import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path
import csv
# load recon.loss, ..., data
import sys 
assert len(sys.argv) == 3,\
"""
Please input just a folder name and then a seed number for the KL div. plot.
The seed is assumed to be an integer between 1-5 inclusive, where the KL div. data enumerates 
seed 1 data first, then seed 2, all the way up until seed 5. 
If you use other seeds, please modify this code or make another script.
"""

folder_fp = Path(__file__).parent / sys.argv[1]
kl_div_seed = int(sys.argv[2])
assert 1 <= kl_div_seed <= 5

"""1. Obtain the data"""
# reconstruction losses
recon_losses = np.genfromtxt(folder_fp / 'recon_loss.csv', delimiter=',', skip_header=1)

# dimension-wise, mean, and total KL divergences
"""
z1 through z10 indicates the corresponding latent dimension's average KL divergence from a minibatch, 
"mean" indicates the globally averaged KL divergence across both the latent dimensions and the minibatch, and 
the "total" indicates the average (of the minibatch) sum of KL divergences across all the latent dimensions
"""
kl_divs = np.genfromtxt(folder_fp / 'kl_divs.csv', delimiter=',', skip_header=1)

kld_headers = None
with open(folder_fp / 'kl_divs.csv') as f:
    kld_headers = next(csv.reader(f))


# k-factor similarity losses
k_sim_losses = np.genfromtxt(folder_fp / 'k_sim_loss.csv', delimiter=',', skip_header=1)

# discriminator accuracies
discrim_accs = np.genfromtxt(folder_fp / 'discrim_acc.csv', delimiter=',', skip_header=1)

# total correlation
tc = np.genfromtxt(folder_fp / 'total_corr.csv', delimiter=',', skip_header=1)




"""2. Parse the data out"""
# kl div.
# indexing scheme for obtaining the right seed data for the KL div. csv:
"""
when there is only one seed, we could do:
# dimwise_klds = kl_divs[:, 1:11]
# mean_kld = kl_divs[:, 11]
# total_kld = kl_divs[:, 12]

When extending beyond though, we get the following sequence
[1, 10], 11, 12
[13, 22], 23, 24
[25, 34], 35, 36

12 is the factor we can multiply by to get the right indices
"""



plt.tight_layout()

"""#. Plotting the data"""
def plot_scalar_metric(save_name, data, x_label, y_label, y_limits=None, tick_interval=None):
    """Plots any non-vector-based (e.g. dimenwion-wise KL divergence) metric
    data (np.ndarray): contains iterations as the first column and 1 or more columns
    afterwards (e.g. 5 columns for 5 seeds)"""
    x = data[:, 0]
    y_avg = np.mean(data[:, 1:], axis = 1)
    error = np.std(data[:, 1:], axis = 1)
    y_lower = y_avg - error
    y_upper = y_avg + error
    
    fig, ax = plt.subplots()

    # non-zero error check (there may be cases where only one seed's value is being plotted instead)
    if error.any():
        ax.plot(x, y_avg, color='blue', label='avg.')
        ax.fill_between(x, y_lower, y_upper, color='lightblue', alpha=0.5, label='standard deviation')
        ax.legend(prop={'size': 11})
    else:
        ax.plot(x, y_avg, color='blue')

    if tick_interval:
        ax.yaxis.set_ticks(np.arange(*y_limits, tick_interval))

    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)

    # manually specified lower and upper limits for vertical axis
    if y_limits:
        ax.set_ylim(*y_limits)
    # automatically find the lower and upper limits instead of the vertical axis,
    # without consideration of outliers
    else:
        y_limits = [np.percentile(y_lower, 0.7), np.percentile(y_upper, 99.3)]
        y_range = y_limits[1] - y_limits[0]
        y_limits[0] -= 0.1 * y_range
        y_limits[1] += 0.1 * y_range
        ax.set_ylim(*y_limits)
    
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))
    ax.tick_params(labelsize=14)

    ax.grid()
    fig.savefig(folder_fp / f'{save_name}.pdf', bbox_inches='tight', dpi=600)



plot_scalar_metric(save_name='recon_loss', data=recon_losses, 
                   x_label='Training Steps', y_label='Reconstruction Loss')

plot_scalar_metric(save_name='k_sim_loss', data=k_sim_losses, 
                   x_label='Training Steps', y_label='k-Factor Similarity Loss')

plot_scalar_metric(save_name='discrim_acc', data=discrim_accs, 
                   x_label='Training Steps', y_label='Discriminator Accuracy')

plot_scalar_metric(save_name='total_corr', data=tc, 
                   x_label='Training Steps', y_label='Estimated TC')


# kl div.
def plot_kl_div(data, y_limits=None, tick_interval=None):
    x_kl = data[:, 0]

    z1_idx = 1 + 12 * (kl_div_seed - 1)
    mean_idx = 11 + 12 * (kl_div_seed - 1)
    total_idx = 12 + 12 * (kl_div_seed - 1)

    dimwise_klds = data[:, z1_idx:mean_idx]
    mean_kld = data[:, mean_idx]
    total_kld = data[:, total_idx]

    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(10)]

    fig, ax = plt.subplots()

    for i in range(10):
        dim_kld = dimwise_klds[:, i]
        ax.plot(x_kl, dim_kld, color=colors[i], label=kld_headers[z1_idx + i], lw=2.5)

    ax.plot(x_kl, mean_kld , color='black', label=kld_headers[mean_idx], lw=2.5)
    ax.plot(x_kl, total_kld , color='black', label=kld_headers[total_idx], lw=2.5)
    ax.set_xlabel('Training Steps', fontsize=15) 
    ax.set_ylabel('KL Divergence', fontsize=15)

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))
    ax.tick_params(labelsize=14)

    if y_limits:
        ax.set_ylim(*y_limits)
    else:
        ax.set_ylim(bottom=0, top=np.percentile(total_kld, 99.3))

    if tick_interval:
        ax.yaxis.set_ticks(np.arange(*y_limits, tick_interval))

    ax.legend(loc='upper left', prop={'size': 10.5})
    ax.grid()
    fig.savefig(folder_fp / 'kl_divs.pdf', bbox_inches='tight', dpi=600)

plot_kl_div(kl_divs, (0, 30), 5)

