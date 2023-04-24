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


# k-factor similarity losses
k_sim_losses = np.genfromtxt(folder_fp / 'k_sim_loss.csv', delimiter=',', skip_header=1)

# discriminator accuracies
discrim_accs = np.genfromtxt(folder_fp / 'discrim_acc.csv', delimiter=',', skip_header=1)

# total correlation
tc = np.genfromtxt(folder_fp / 'total_corr.csv', delimiter=',', skip_header=1)




"""2. Parse the data out"""

# recon.
x_recon = recon_losses[:, 0]
y_recon_avg = np.mean(recon_losses[:, 1:], axis = 1)
error_recon = np.std(recon_losses[:, 1:], axis = 1)
y_recon_lower = y_recon_avg - error_recon
y_recon_upper = y_recon_avg + error_recon



# kl div.
x_kl = kl_divs[:, 0]
dimwise_klds = kl_divs[:, 1:11]
mean_kld = kl_divs[:, 11]
total_kld = kl_divs[:, 12]



# k-factor similarity loss
x_ksl = k_sim_losses[:, 0]
y_ksl = np.mean(k_sim_losses[:, 1:], axis = 1)
error_ksl = np.std(k_sim_losses[:, 1:], axis = 1)
y_ksl_lower = y_ksl - error_ksl
y_ksl_upper = y_ksl + error_ksl


# discrim. acc.
x_discrim = discrim_accs[:, 0]
y_discrim = np.mean(discrim_accs[:, 1:], axis = 1)
error_discrim = np.std(discrim_accs[:, 1:], axis = 1)
y_discrim_lower = y_discrim - error_discrim
y_discrim_upper = y_discrim + error_discrim

# total corr.
x_tc = tc[:, 0]
y_tc = np.mean(tc[:, 1:], axis = 1)
error_tc = np.std(tc[:, 1:], axis = 1)
y_tc_lower = y_tc - error_tc
y_tc_higher = y_tc + error_tc




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
fig.savefig(folder_fp / 'recon_loss.pdf', bbox_inches='tight', dpi=600)



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
fig.savefig(folder_fp / 'kl_divs.pdf', bbox_inches='tight', dpi=600)



# k-factor similarity loss
fig, ax = plt.subplots()
ax.plot(x_ksl, y_ksl, color='blue', label='avg.')
ax.fill_between(x_ksl, y_ksl_lower, y_ksl_upper, color='lightblue', alpha=0.5, label='standard deviation')

ax.set_xlabel('Training steps', fontsize=14)
ax.set_ylabel('K-factor Similarity Loss', fontsize=14)
ax.yaxis.set_ticks(np.arange(0, 1, 0.1))
plt.ylim(0, 1)

ax.legend(prop={'size': 11})
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))

plt.tick_params(labelsize=14)
# plt.grid(True)
plt.grid()
plt.tight_layout()
fig.savefig(folder_fp / 'k_sim_loss.pdf', bbox_inches='tight', dpi=600)


# discriminator accuracy
fig, ax = plt.subplots()
ax.plot(x_discrim, y_discrim, color='blue', label='avg.')
ax.fill_between(x_discrim, y_discrim_lower, y_discrim_upper, color='lightblue', alpha=0.5, label='standard deviation')

ax.set_xlabel('Training steps', fontsize=14)
ax.set_ylabel('Discriminator Accuracy', fontsize=14)
ax.yaxis.set_ticks(np.arange(0, 1, 0.1))
plt.ylim(0, 1)

ax.legend(prop={'size': 11})
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))

plt.tick_params(labelsize=14)
# plt.grid(True)
plt.grid()
plt.tight_layout()
fig.savefig(folder_fp / 'discrim_acc.pdf', bbox_inches='tight', dpi=600)


# total correlation
fig, ax = plt.subplots()
ax.plot(x_tc, y_tc, color='blue', label='avg.')
ax.fill_between(x_tc, y_tc_lower, y_tc_higher, color='lightblue', alpha=0.5, label='standard deviation')

ax.set_xlabel('Training steps', fontsize=14)
ax.set_ylabel('Estimated Total Corrleation', fontsize=14)
ax.yaxis.set_ticks(np.arange(0, 1, 0.1))
plt.ylim(0, 1)

ax.legend(prop={'size': 11})
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x/1000)}K'))

plt.tick_params(labelsize=14)
# plt.grid(True)
plt.grid()
plt.tight_layout()
fig.savefig(folder_fp / 'total_corr.pdf', bbox_inches='tight', dpi=600)




