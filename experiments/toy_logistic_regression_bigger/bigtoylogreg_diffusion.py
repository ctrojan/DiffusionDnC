#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import blackjax
from matplotlib import pyplot as plt
import jax.numpy as jnp
import jax
import tensorflow as tf
import time

from dnc.diffusion_merge import EBModel, annealed_mcmc_sampling, matsqrt, normalising_transform
from dnc.mcmc import sample_nuts
from dnc.plotting import fitted_energy_subposteriors, kde_contour, kde_1d, energy_contour_t



experiment_dir = 'experiments/toy_logistic_regression_bigger/'
method = 'diff'
train_times = []
sample_times = []

seeds = [i for i in range(5)]

for seed in seeds:
    
    rng = jax.random.PRNGKey(seed+10)
    tf.random.set_seed(seed+10)
    
    shard_samples = jnp.load(experiment_dir + f'data/shard_samples_seed{seed}.npy')
    shard_evals = jnp.load(experiment_dir + f'data/shard_gradlogpost_evals_seed{seed}.npy')
    
    shard_samples = jnp.array(shard_samples) 
    shard_evals = jnp.array(shard_evals) 
    dim = shard_samples.shape[-1]
    M = shard_samples.shape[0]
    
    shift, scale = normalising_transform(shard_samples)
    
    rng, init_rng = jax.random.split(rng, 2)
    init_rng = jax.random.split(init_rng, M)
    score_models = [EBModel(1, 32, dim, 1e-3, init_rng[i]) for i in range(M)]
    
    for i in range(M):
        rng, train_rng = jax.random.split(rng, 2)
        train_steps = int(100*50000/shard_samples[i].shape[0])
        t = time.time()
        score_models[i].train_combined((shard_samples[i]-shift[i])@scale[i], shard_evals[i]@jnp.linalg.inv(scale[i]), train_steps, train_rng)
        train_times.append(time.time()-t)
    
    @jax.jit
    def potential_prod(x, t): return sum([score_models[i].potential.apply(
        score_models[i].params, (x-shift[i])@scale[i], t) for i in range(M)])
    
    #### just t=t0
    rng = jax.random.PRNGKey(seed+20)
    t0 = 0.
    burnin=100
    steps=10000 + burnin
    
    shard_vars_inv = [scale[i]@scale[i].transpose() for i in range(M)]
    V = jnp.linalg.inv(sum(shard_vars_inv))
    mu = V@( sum([shard_vars_inv[b]@shift[b] for b in range(M)]) )
    scale_M = matsqrt(V)
    
    rng, sample_rng = jax.random.split(rng, 2)
    
    def p(x):
        return potential_prod(x,jnp.full(1, t0))[0]
    
    beta_init = mu.reshape((1,dim))
    t = time.time()
    samples_approx, info = sample_nuts(sample_rng, p, 1e-2, beta_init, steps, burnin)
    sample_time = time.time() - t
    sample_times.append(sample_time)
    jnp.save(experiment_dir + f'data/samples_{method}_seed{seed}.npy', samples_approx)  
     
jnp.save(experiment_dir + f'data/times_{method}.npy', jnp.array([jnp.mean(jnp.array(train_times)),jnp.mean(jnp.array(sample_times))]))
