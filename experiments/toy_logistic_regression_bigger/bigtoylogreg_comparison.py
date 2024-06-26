#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import jax
from jax import numpy as jnp
from dnc.metrics import d_mah, d_iad, d_skew
from dnc.transform_merges import consensus, swiss, matsqrt
experiment_dir = 'experiments/toy_logistic_regression_bigger/'

seeds = [i for i in range(5)]
shard_samples = [jnp.load(experiment_dir + f'data/shard_samples_seed{i}.npy') for i in seeds]
samples_exact = [jnp.load(experiment_dir + f'data/samples_exact_seed{i}.npy') for i in seeds]
dim = samples_exact[0].shape[1]; n_samp = shard_samples[0].shape[1]; M = shard_samples[0].shape[0]

def print_results(method, samps):
    results = []
    for i in range(len(seeds)):
        results.append([d_mah(samps[i], samples_exact[i]), d_iad(samps[i], samples_exact[i]), d_skew(samps[i], samples_exact[i])])
    results = jnp.array(results)
    results_avg = jnp.mean(results, axis=0)
    results_std = jnp.std(results, axis=0)
    print(method+' & '+' & '.join([str(jnp.round(results_avg[i],2))[:4]+' ('+str(jnp.round(results_std[i],2))[:4]+')' for i in range(3)])+' \\\\')
    return results
print("Method & d_MAH & d_IAD & d_skew")    
 
# cmc

cmc_samps = [consensus(shard_samples[i]) for i in range(len(seeds))]
results_cmc = print_results('Consensus', cmc_samps)


# swiss

shard_samples_swiss = [jnp.load(experiment_dir + f'data/shard_samples_swiss_seed{i}.npy') for i in seeds]
swiss_samps = [swiss(shard_samples_swiss[i]) for i in range(len(seeds))]
results_swiss = print_results('SwISS', swiss_samps)


# parametric gaussian
gauss_samps = []
for s in range(len(seeds)):
    shard_means = jnp.mean(shard_samples[s], axis=1)
    shard_vars_inv = [jnp.linalg.inv(jnp.cov(shard_samples[s][i], rowvar=False)) for i in range(M)]
    V = jnp.linalg.inv(sum(shard_vars_inv))
    scale_M = matsqrt(V)
    mu = V@( sum([shard_vars_inv[b]@shard_means[b] for b in range(M)]) )
    
    rng = jax.random.PRNGKey(10+seeds[s])
    gauss_samps.append(jax.random.normal(rng, (n_samp, dim))@scale_M + mu)
    
results_gauss = print_results('Gaussian', gauss_samps)


# semiparam DE

semikde_samps = [jnp.load(experiment_dir + f'data/samples_semikde_seed{seed}.npy') for seed in seeds]
results_semikde = print_results('Semiparametric', semikde_samps)
times = jnp.load(experiment_dir + f'data/times_semikde.npy')
print('time:',jnp.mean(times))


# GP

gp_samps = [jnp.load(experiment_dir + f'data/samples_gp_seed{seed}.npy') for seed in seeds]
gp_results = print_results('Gaussian process', gp_samps)
times = jnp.load(experiment_dir + f'data/times_gp.npy')
print('times:',times)

# diffusion

diff_samps = [jnp.load(experiment_dir + f'data/samples_diff_seed{seed}.npy') for seed in seeds]
diff_results = print_results('Diffusion', diff_samps)
times = jnp.load(experiment_dir + f'data/times_diff.npy')
print('times:',times)