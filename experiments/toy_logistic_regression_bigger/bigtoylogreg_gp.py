#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
import time
import tensorflow as tf

from dnc.mcmc import sample_nuts
from dnc.gp_merge import logdensity_GP
from dnc.transform_merges import matsqrt
experiment_dir = 'experiments/toy_logistic_regression_bigger/'
method = 'gp'

train_times = []
sample_times = []
seeds = [i for i in range(5)]
for seed in seeds:    
    tf.random.set_seed(seed+10)
    rng = jax.random.PRNGKey(seed+10)
    shard_samples = jnp.load(experiment_dir + f'data/shard_samples_seed{seed}.npy')
    shard_evals = jnp.load(experiment_dir + f'data/shard_logpost_evals_seed{seed}.npy')
    
    thin_by = int(jnp.ceil(shard_samples.shape[1]/1000))
    M = shard_samples.shape[0]
    dim = shard_samples.shape[-1]
    
    fitted_logposts = []
    rng, gp_rng = jax.random.split(rng, 2)
    gp_rng = jax.random.split(gp_rng, M)
    for i in range(M):
        fitted_logposts.append(logdensity_GP(shard_samples[i][::thin_by],shard_evals[i][::thin_by]))
        t=time.time()
        fitted_logposts[i].fit(gp_rng[i])
        train_times.append(time.time()-t)
        
    shard_means = jnp.mean(shard_samples, axis=1)
    shard_vars_inv = [jnp.linalg.inv(jnp.cov(shard_samples[i], rowvar=False)) for i in range(M)]
    V = jnp.linalg.inv(sum(shard_vars_inv))
    mu = V@( sum([shard_vars_inv[b]@shard_means[b] for b in range(M)]) )
    beta_init = mu.reshape((1,dim))
    
    def p(beta):
        return sum([ fitted_logposts[i].full_post_component(beta) for i in range(M) ])
    
    rng = jax.random.PRNGKey(seed+20)
    burnin=100; steps=10000+burnin
    
    rng, sample_rng = jax.random.split(rng)
    t = time.time()
    samples_approx, info = sample_nuts(sample_rng, p, 1e-2, beta_init, steps, burnin)
    sample_time = time.time() - t

    sample_times.append(sample_time)    
    
    jnp.save(experiment_dir + f'data/samples_{method}_seed{seed}.npy', samples_approx) 

 
jnp.save(experiment_dir + f'data/times_{method}.npy', jnp.array([jnp.mean(jnp.array(train_times)),jnp.mean(jnp.array(sample_times))]))
