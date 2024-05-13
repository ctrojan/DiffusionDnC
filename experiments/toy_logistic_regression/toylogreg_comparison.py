#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from math import floor, ceil
import jax
from jax import numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
from dnc.metrics import d_mah, d_iad, d_skew
from dnc.transform_merges import consensus, swiss, matsqrt
from dnc.plotting import kde_contour, kde_1d, kde_contour_comparison
plotlim = [-3.5,-1.5]; M=15
experiment_dir = 'experiments/toy_logistic_regression/'
samples_exact = jnp.load(experiment_dir + 'data/samples_exact.npy')
shard_samples = jnp.load(experiment_dir + f'data/{M}shards_samples.npy')
dim = samples_exact.shape[1]; n_samp = shard_samples.shape[1]

def print_results(method, samps):
    print(method, '&',str(jnp.round(d_mah(samps, samples_exact),2))[:4],'&',str(jnp.round(d_iad(samps, samples_exact),2))[:4],'&',str(jnp.round(d_skew(samps, samples_exact),2))[:4],'\\\\')
    
print("Method & d_MAH & d_IAD & d_skew")    
    
# cmc

cmc_samps = consensus(shard_samples)
print_results('Consensus', cmc_samps)


# swiss

shard_samples_swiss = jnp.load(experiment_dir + f'data/{M}shards_samples_swiss.npy')
swiss_samps = swiss(shard_samples_swiss)
print_results('SwISS', swiss_samps)


# parametric gaussian

shard_means = jnp.mean(shard_samples, axis=1)
shard_vars_inv = [jnp.linalg.inv(jnp.cov(shard_samples[i], rowvar=False)) for i in range(M)]
V = jnp.linalg.inv(sum(shard_vars_inv))
scale_M = matsqrt(V)
mu = V@( sum([shard_vars_inv[b]@shard_means[b] for b in range(M)]) )

rng = jax.random.PRNGKey(2)
gauss_samps = jax.random.normal(rng, (n_samp, dim))@scale_M + mu
print_results('Gaussian', gauss_samps)


# semiparam DE

semikde_samps = jnp.load(experiment_dir + f'data/samples_semikde_{M}shards.npy')
print_results('Semiparametric', semikde_samps)
times = jnp.load(experiment_dir + f'data/time_semikde_{M}shards.npy')
print('time:',jnp.mean(times*60))

# GP

gp_samps = jnp.load(experiment_dir + f'data/samples_gp_{M}shards.npy')
print_results('Gaussian process', gp_samps)
times = jnp.load(experiment_dir + f'data/times_gp_{M}shards.npy')
print('times:',times)


# diffusion

diff_samps = jnp.load(experiment_dir + f'data/samples_diff_{M}shards.npy')
print_results('Diffusion', diff_samps)
times = jnp.load(experiment_dir + f'data/times_diff_{M}shards.npy')
print('times:',times)


###### plots

subpost_plotlim_x = [-7,-1.5]
subpost_plotlim_y = [-7,-0.5]
ncols = min(M,5); nrows = ceil(M/ncols)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7.8, 3))
for i in range(M):
    kde_contour(shard_samples[i], plotlim_x=subpost_plotlim_x, plotlim_y=subpost_plotlim_y, fig=axes[i%nrows,floor(i/nrows)], color=[(0,0,0)])
    kde_contour(samples_exact, plotlim_x=subpost_plotlim_x, plotlim_y=subpost_plotlim_y, fig=axes[i%nrows,floor(i/nrows)], color=[(0,0,1)])

    if floor(i/nrows)>0:
        axes[i%nrows,floor(i/nrows)].set_yticks([])
    if i%nrows < nrows-1:
        axes[i%nrows,floor(i/nrows)].set_xticks([])
fig.tight_layout()
fig.savefig(experiment_dir + f'plots/{M}_subposts_with_full.pdf',bbox_inches='tight')

#


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 4))
kde_contour_comparison(cmc_samps, samples_exact, plotlim, fig=axes[0,0])
axes[0,0].set_title('Consensus')
kde_contour_comparison(swiss_samps, samples_exact, plotlim, fig=axes[0,1])
axes[0,1].set_title('SwISS')
kde_contour_comparison(gauss_samps, samples_exact, plotlim, fig=axes[0,2])
axes[0,2].set_title('Gaussian')
kde_contour_comparison(semikde_samps, samples_exact, plotlim, fig=axes[1,0])
axes[1,0].set_title('Semiparametric')
kde_contour_comparison(gp_samps, samples_exact, plotlim, fig=axes[1,1])
axes[1,1].set_title('Gaussian Process')
kde_contour_comparison(diff_samps, samples_exact, plotlim, fig=axes[1,2])
axes[1,2].set_title('Diffusion')
fig.tight_layout()
fig.savefig(experiment_dir + f'plots/comparison_{M}shards.pdf',bbox_inches='tight')
