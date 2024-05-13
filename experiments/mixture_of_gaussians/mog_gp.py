#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
import numpy as np
import time
import tensorflow as tf

from dnc.gp_merge import logdensity_GP
from dnc.mcmc import sample_nuts
from dnc.transform_merges import matsqrt

tf.random.set_seed(1)
rng = jax.random.PRNGKey(1)

experiment_dir = 'experiments/mixture_of_gaussians/'
method = 'gp'

thin_by = 10
samples_exact = jnp.load(experiment_dir + 'data/samples_exact.npy')
shard_samples = jnp.load(experiment_dir + 'data/shard_samples.npy')[:,::thin_by]
shard_logpost_evals = jnp.load(experiment_dir + 'data/shard_logpost_evals.npy')[:,::thin_by,:]

dim = shard_samples.shape[-1]; M = shard_samples.shape[0]

fitted_logposts = []
rng, gp_rng = jax.random.split(rng, 2)
gp_rng = jax.random.split(gp_rng, M)
train_ts = []
for i in range(M):
    fitted_logposts.append(logdensity_GP(shard_samples[i],shard_logpost_evals[i]))
    t = time.time()
    fitted_logposts[i].fit(gp_rng[i])
    train_ts.append(time.time()-t)
train_ts = jnp.array(train_ts)    
    
rng = jax.random.PRNGKey(2)

shard_means = jnp.mean(shard_samples, axis=1)
shard_vars_inv = [jnp.linalg.inv(jnp.cov(shard_samples[i], rowvar=False)) for i in range(M)]
V = jnp.linalg.inv(sum(shard_vars_inv))
mu = V@( sum([shard_vars_inv[b]@shard_means[b] for b in range(M)]) )
scale_M = matsqrt(V) 

def p(beta):
    return sum([ fitted_logposts[i].full_post_component(beta) for i in range(M) ])


burnin=10; steps=1000+burnin
samples = np.zeros((10000,3))
sample_ts = []
for i in range(10):
    rng, sample_rng = jax.random.split(rng)
    sample_rng, init_rng = jax.random.split(sample_rng)
    keys = jax.random.split(sample_rng, steps)
    
    beta_init = jax.random.normal(init_rng,(1,dim))@scale_M + mu

    t = time.time()
    samples_approx, info = sample_nuts(sample_rng, p, .02, beta_init, steps, burnin)
    sample_ts.append(time.time()-t)

    samples[(i*1000):((i+1)*1000),:] = samples_approx

sample_ts = jnp.array(sample_ts)

samples_approx = samples
jnp.save(experiment_dir + f'data/samples_{method}.npy', samples_approx)  
jnp.save(experiment_dir + f'data/times_{method}.npy', jnp.array([jnp.mean(train_ts),jnp.sum(sample_ts)]))
