import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path
import csv
# load recon.loss, ..., data
import sys 
assert len(sys.argv) == 2, "Please input just a folder name"

folder_fp = Path(__file__).parent / sys.argv[1]

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





"""2. Parse the data out"""

# recon.
x_recon = recon_losses[:, 0]
y_recon_avg = recon_losses[:, -3]
y_recon_lower = recon_losses[:, -2]
y_recon_upper = recon_losses[:, -1]

# kl div.
x_kl = kl_divs[:, 0]
dimwise_klds = kl_divs[:, 1:11]
mean_kld = kl_divs[:, 11]
total_kld = kl_divs[:, 12]

# k-factor similarity loss

# total corr.

# discrim. acc.


"""#. Plotting the data"""

# recon. losses
fig, ax = plt.subplots()
ax.plot(x_recon, y_recon_avg, color='blue', label='avg.')
ax.fill_between(x_recon, y_recon_lower, y_recon_upper, color='lightblue', alpha=0.5, label='standard deviation')

ax.set_xlabel('Training steps', fontsize=14)
ax.set_ylabel('Reconstruction Loss', fontsize=14)
plt.ylim(10, 150)

ax.legend(prop={'size': 11})
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))


plt.tick_params(labelsize=14)
# plt.grid(True)
plt.grid()
plt.tight_layout()
fig.savefig('recon_loss.pdf', bbox_inches='tight', dpi=600)


# kl div.
cmap = plt.get_cmap('tab10')
colors = [cmap(i) for i in range(10)]

fig, ax = plt.subplots()

for i in range(10):
    dim_kld = dimwise_klds[:, i]
    ax.plot(x_kl, dim_kld, color=colors[i], label=kld_headers[i + 1], lw=2.5)

ax.plot(x_kl, mean_kld , color='black', label='mean', lw=2.5)
ax.plot(x_kl, total_kld , color='black', label='total', lw=2.5)
ax.set_xlabel('Training steps', fontsize=15) 
ax.set_ylabel('KL Divergence', fontsize=15)

ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))
plt.ylim(0, 22)
ax.yaxis.set_ticks(np.arange(0, 22, 2))


plt.tick_params(labelsize=15)
plt.legend(loc='upper left', prop={'size': 10.5})
plt.grid()
plt.tight_layout()
fig.savefig('kl_divs.pdf', bbox_inches='tight', dpi=600)

