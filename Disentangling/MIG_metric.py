#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

import numpy as np
from tqdm import tqdm

from model import BetaVAE_H, BetaVAE_B
# from factmodel import FactorVAE1
from dataset import CustomTensorDataset

import os
import sys
from numbers import Number
import math
import matplotlib.pyplot as plt
import time

eps = 1e-8


# In[ ]:

time_start = time.time()

file_path = sys.argv[2]
print("checkpoint file_path:>> ", file_path)

# model_map = {'betavae_h': BetaVAE_H,
#              'betavae_b': BetaVAE_B, 'factvae': FactorVAE1}

model_map = {'betavae_h': BetaVAE_H, 'betavae_b': BetaVAE_B}
model_name = sys.argv[1]

# In[2]:


def logsumexp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


# In[3]:


class Normal(nn.Module):
    def __init__(self, mu=0, sigma=1):
        super(Normal, self).__init__()
        # Precompute the normalization constant for log density calculations
        self.normalization = torch.Tensor([np.log(2 * np.pi)])

        # Set the mean and standard deviation of the distribution
        self.mu = torch.Tensor([mu])
        self.logsigma = torch.Tensor([math.log(sigma)])
        
    def _check_inputs(self, size, mu_logsigma):
        if size is None and mu_logsigma is None:
            raise ValueError(
                'Either one of size or params should be provided.')

        elif size is not None and mu_logsigma is not None:
            # Extract the mean and log standard deviation from mu_logsigma and
            # expand to match the given size
            mu = mu_logsigma.select(-1, 0).expand(size)
            logsigma = mu_logsigma.select(-1, 1).expand(size)
            return mu, logsigma

        elif size is not None:
            # Use the stored mean and log standard deviation and expand to match
            # the given size
            mu = self.mu.expand(size)
            logsigma = self.logsigma.expand(size)
            return mu, logsigma

        elif mu_logsigma is not None:
            # Use the provided mu_logsigma and extract the mean and log standard
            # deviation
            mu = mu_logsigma.select(-1, 0)
            logsigma = mu_logsigma.select(-1, 1)
            return mu, logsigma

        else:
            raise ValueError(
                'Given invalid inputs: size={}, mu_logsigma={})'.format(
                    size, mu_logsigma))


    def sample(self, size=None, params=None):
        mu, logsigma = self._check_inputs(size, params)
        # Sample from a standard normal distribution
        # std_z = torch.randn(mu.size()).type_as(mu.data)
        std_z = torch.randn(mu.size()).to(mu.data)
        # Scale and shift the sample to match the desired mean and standard deviation
        sample = std_z * torch.exp(logsigma) + mu
        return sample

    def log_density(self, sample, params=None):
        if params is not None:
            # If mu and sigma are provided, extract them
            mu, logsigma = self._check_inputs(None, params)
        else:
            # If mu and sigma are not provided, use the stored values and expand
            # to match the size of the sample
            mu, logsigma = self._check_inputs(sample.size(), None)
            # mu = mu.type_as(sample)
            mu = mu.to(sample)
            # logsigma = logsigma.type_as(sample)
            logsigma = logsigma.to(sample)

        # Compute the log density of the normal distribution
        # c = self.normalization.type_as(sample.data)
        c = self.normalization.to(sample.data)
        inv_sigma = torch.exp(-logsigma)
        tmp = (sample - mu) * inv_sigma
        return -0.5 * (tmp * tmp + 2 * logsigma + c)

    def NLL(self, params, sample_params=None):
        """
        Negative Log Likelihood of the normal distributions 
        with given params

        Analytically computes
            E_N(mu_2,sigma_2^2) [ - log N(mu_1, sigma_1^2) ]
        If mu_2, and sigma_2^2 are not provided, defaults to entropy.
        """
        mu, logsigma = self._check_inputs(None, params)
        if sample_params is not None:
            sample_mu, sample_logsigma = self._check_inputs(
                None, sample_params)
        else:
            sample_mu, sample_logsigma = mu, logsigma

        # c = self.normalization.type_as(sample_mu.data)
        c = self.normalization.to(sample_mu.data)
        nll = logsigma.mul(-2).exp() * (sample_mu - mu).pow(2) + torch.exp(
            sample_logsigma.mul(2) - logsigma.mul(2)) + 2 * logsigma + c
        return nll.mul(0.5)

    def kld(self, params):
        """Computes KL(q||p) where q is the given distribution and p
        is the standard Normal distribution.
        """
        mu, logsigma = self._check_inputs(None, params)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mean^2 - sigma^2)
        kld = logsigma.mul(2).add(1) - mu.pow(2) - logsigma.exp().pow(2)
        kld.mul_(-0.5)
        return kld

    def get_params(self):
        return torch.cat([self.mu, self.logsigma])

    @property
    def nparams(self):
        return 2

    @property
    def ndim(self):
        return 1

    @property
    def is_reparameterizable(self):
        return True

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' ({:.3f}, {:.3f})'.format(
            self.mu.data[0], self.logsigma.exp().data[0])
        return tmpstr


# In[21]:


def estimate_entropies(qz_samples, qz_params, q_dist=Normal(), n_samples=10000, weights=None):
    """Computes the term:
        E_{p(x)} E_{q(z|x)} [-log q(z)]
    and
        E_{p(x)} E_{q(z_j|x)} [-log q(z_j)]
    where q(z) = 1/N sum_n=1^N q(z|x_n).

    Assumes samples are from q(z|x) for *all* x in the dataset.
    Assumes that q(z|x) is factorial ie. q(z|x) = prod_j q(z_j|x).
    Computes numerically stable NLL:
        - log q(z) = log N - logsumexp_n=1^N log q(z|x_n)
    
    Inputs:
    -------
    qz_samples (K, N) torch.Tensor: samples from the posterior q(z|x)
    qz_params  (N, K, nparams) torch.Tensor: parameters of the posterior q(z|x)
    q_dist: an instance of a PyTorch distribution representing the prior distribution p(z). Default is Normal().
    n_samples (int): number of samples to use for the Monte Carlo integration. Default is 10000.
    weights (N) torch.Tensor: importance weights of each data point. Default is None.

    Returns:
    --------
    entropies (K torch.Tensor): 
    A K-dimensional tensor representing the estimated entropy of each dimension of the latent space.
    """

    # Only take a sample subset of the samples
    if weights is None:
        # randomly select a subset of qz_samples
        qz_samples = qz_samples.index_select(1, torch.randperm(qz_samples.size(1))[:n_samples].cuda())
    else:
        # randomly select a subset of qz_samples weighted by the importance weights
        sample_inds = torch.multinomial(weights, n_samples, replacement=True)
        qz_samples = qz_samples.index_select(1, sample_inds)

    K, S = qz_samples.size()
    N, _, nparams = qz_params.size()
    assert(nparams == q_dist.nparams)
    assert(K == qz_params.size(1))

    if weights is None:
        weights = -math.log(N)
    else:
        weights = torch.log(weights.view(N, 1, 1) / weights.sum())

    entropies = torch.zeros(K).cuda()

    # create a progress bar for visualization
    pbar = tqdm(total=S)
    k = 0
    while k < S:
        # size of the batch to process
        batch_size = min(10, S - k)
        # compute the log density of q(z|x) for the current batch
        logqz_i = q_dist.log_density(
            qz_samples.view(1, K, S).expand(N, K, S)[:, :, k:k + batch_size],
            qz_params.view(N, K, 1, nparams).expand(N, K, S, nparams)[:, :, k:k + batch_size])
        k += batch_size

        # computes - log q(z_i) summed over minibatch
        # use logsumexp to compute the logarithm of the sum of exponentials in a numerically stable way
        entropies += - logsumexp(logqz_i + weights,
                                 dim=0, keepdim=False).data.sum(1)
        pbar.update(batch_size)
    pbar.close()

    # average over the number of samples
    entropies /= S

    return entropies


# In[5]:


def MIG(mi_normed):
    """
    Computes the mutual information gap (MIG) metric given normalized mutual information scores.

    Parameters:
    -----------
    mi_normed (torch.Tensor): 
    A tensor of size (num_factors, num_latent_dims) representing the normalized mutual information scores.

    Returns:
    --------
    metric (torch.Tensor): 
    The MIG metric, defined as the mean of the difference between the largest and second-largest mutual information 
    scores for each factor.
    """
    return torch.mean(mi_normed[:, 0] - mi_normed[:, 1])


def compute_metric_shapes(marginal_entropies, cond_entropies):
    """
    Computes the MIG metric for a set of shapes.

    Parameters:
    -----------
    marginal_entropies (torch.Tensor): 
    A tensor of size (num_latent_dims,) representing the marginal entropies of the learned latent representation.

    cond_entropies (torch.Tensor): 
    A tensor of size (num_factors, num_latent_dims) representing the conditional entropies of the 
    learned latent representation.

    Returns:
    --------
    metric (torch.Tensor): 
    The MIG metric for the set of shapes, defined as the mean of the difference between 
    the largest and second-largest mutual information scores for each factor.
    """
    factor_entropies = [6, 40, 32, 32]
    # compute the mutual information scores for each factor
    mutual_infos = marginal_entropies[None] - cond_entropies
    # sort the mutual information scores in descending order
    mutual_infos = torch.sort(mutual_infos, dim=1, descending=True)[
        0].clamp(min=0)
    # normalize the mutual information scores by the factor entropies
    mi_normed = mutual_infos / torch.Tensor(factor_entropies).log()[:, None]
    # compute the MIG metric using the normalized mutual information scores
    metric = MIG(mi_normed)
    return metric


# In[6]:


root = os.path.join(
    'data', 'dsprites-dataset/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
data = np.load(root, encoding='bytes')
data = torch.from_numpy(data['imgs']).unsqueeze(1).float()
train_kwargs = {'data_tensor': data}

shapes_dataset = CustomTensorDataset(**train_kwargs)


# In[7]:

dataset_loader = DataLoader(
    shapes_dataset, batch_size=1000, shuffle=False, num_workers=0)
# In[8]:


# 10 latent variables, 1 channels (as opposed to 3 in RGB)
# for dSprites dataset
vae = model_map[model_name.lower()](10, 1)

# if model_name.lower() == 'factvae':
#     vae = FactorVAE1(10)
# else:
#     vae = model_map[model_name.lower()](10, 1)

vae.cuda()

# In[9]:


if os.path.isfile(file_path):
    print('Checkpoint loaded')
    checkpoint = torch.load(file_path)  # , map_location=torch.device('cpu'))
    if model_name.lower() == 'factvae':
        raise ValueError("FactorVAE not supported")
        # vae.load_state_dict(checkpoint['model_states']['VAE'])
    else:
        vae.load_state_dict(checkpoint['model_states']['net'])


# In[10]:


q_dist = Normal()
N = len(dataset_loader.dataset)  # number of data samples
K = vae.z_dim                    # number of latent variables
nparams = q_dist.nparams
vae.eval()


# In[11]:


print('Computing q(z|x) distributions.')
qz_params = torch.Tensor(N, K, nparams)

n = 0
for i, xs in enumerate(dataset_loader):
    print(i + 1, len(dataset_loader), end='\r')
    batch_size = xs.size(0)
    # xs = Variable(xs.view(batch_size, 1, 64, 64), volatile=True)
    if model_name == 'factvae':
        raise ValueError("FactorVAE is not supported")
        # qz_params[n:n + batch_size] = vae.encode.forward(xs.cuda()).view(
        #     batch_size, nparams, vae.z_dim).transpose(1, 2).data
    else:
        qz_params[n:n + batch_size] = vae.encoder.forward(xs.cuda()).view(
            batch_size, nparams, vae.z_dim).transpose(1, 2).data

    n += batch_size


# In[12]:


qz_params = qz_params.view(3, 6, 40, 32, 32, K, nparams).cuda()

# if model_name.lower() != 'factvae':
qz_params[:, :, :, :, :, :, 1] = qz_params[:,
                                           :, :, :, :, :, 1]/2  # added by shao

qz_samples = q_dist.sample(params=qz_params)


# In[22]:


print('Estimating marginal entropies.')
# marginal entropies
marginal_entropies = estimate_entropies(
    qz_samples.view(N, K).transpose(0, 1),
    qz_params.view(N, K, nparams),
    q_dist)

marginal_entropies = marginal_entropies.cpu()
cond_entropies = torch.zeros(4, K)

print('Estimating conditional entropies for scale.')
for i in range(6):
    qz_samples_scale = qz_samples[:, i, :, :, :, :].contiguous()
    qz_params_scale = qz_params[:, i, :, :, :, :].contiguous()

    cond_entropies_i = estimate_entropies(
        qz_samples_scale.view(N // 6, K).transpose(0, 1),
        qz_params_scale.view(N // 6, K, nparams),
        q_dist)

    cond_entropies[0] += cond_entropies_i.cpu() / 6

print('Estimating conditional entropies for orientation.')
for i in range(40):
    qz_samples_scale = qz_samples[:, :, i, :, :, :].contiguous()
    qz_params_scale = qz_params[:, :, i, :, :, :].contiguous()

    cond_entropies_i = estimate_entropies(
        qz_samples_scale.view(N // 40, K).transpose(0, 1),
        qz_params_scale.view(N // 40, K, nparams),
        q_dist)

    cond_entropies[1] += cond_entropies_i.cpu() / 40

print('Estimating conditional entropies for pos x.')
for i in range(32):
    qz_samples_scale = qz_samples[:, :, :, i, :, :].contiguous()
    qz_params_scale = qz_params[:, :, :, i, :, :].contiguous()

    cond_entropies_i = estimate_entropies(
        qz_samples_scale.view(N // 32, K).transpose(0, 1),
        qz_params_scale.view(N // 32, K, nparams),
        q_dist)

    cond_entropies[2] += cond_entropies_i.cpu() / 32

print('Estimating conditional entropies for pox y.')
for i in range(32):
    qz_samples_scale = qz_samples[:, :, :, :, i, :].contiguous()
    qz_params_scale = qz_params[:, :, :, :, i, :].contiguous()

    cond_entropies_i = estimate_entropies(
        qz_samples_scale.view(N // 32, K).transpose(0, 1),
        qz_params_scale.view(N // 32, K, nparams),
        q_dist)

    cond_entropies[3] += cond_entropies_i.cpu() / 32

metric = compute_metric_shapes(marginal_entropies, cond_entropies)
# return metric, marginal_entropies, cond_entropies


print(marginal_entropies)
print(cond_entropies)


# In[31]:
print('Metric: {}'.format(metric.cpu().numpy()))
direct = os.path.dirname(file_path)
out_file = os.path.join(direct, "MIG.txt")
print("save to path >>: ", out_file)

with open(out_file, "a") as fout:
    fout.write(file_path + ": ")
    fout.write(str(metric.cpu().numpy()) + "\n")

time_end = time.time()

print("running time: ", (time_end-time_start)/60, "mins")
