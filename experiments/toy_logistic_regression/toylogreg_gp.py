#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
import time
import tensorflow as tf

from dnc.mcmc import sample_nuts
from dnc.gp_merge import logdensity_GP
from dnc.transform_merges import matsqrt
experiment_dir = 'experiments/toy_logistic_regression/'
method = 'gp'

tf.random.set_seed(1)
rng = jax.random.PRNGKey(1)
M=15
thin_by = 10
samples_exact = jnp.load(experiment_dir + 'data/samples_exact.npy')
shard_samples = jnp.load(experiment_dir + f'data/{M}shards_samples.npy')[:,::thin_by]
shard_evals = jnp.load(experiment_dir + f'data/{M}shards_logpost_evals.npy')[:,::thin_by]

dim = shard_samples.shape[-1]

fitted_logposts = []
rng, gp_rng = jax.random.split(rng, 2)
gp_rng = jax.random.split(gp_rng, M)
train_times = []
for i in range(M):
    fitted_logposts.append(logdensity_GP(shard_samples[i],shard_evals[i]))
    t=time.time()
    fitted_logposts[i].fit(gp_rng[i])
    train_times.append(time.time()-t)
train_times = jnp.array(train_times)
    
shard_means = jnp.mean(shard_samples, axis=1)
shard_vars_inv = [jnp.linalg.inv(jnp.cov(shard_samples[i], rowvar=False)) for i in range(M)]
V = jnp.linalg.inv(sum(shard_vars_inv))
mu = V@( sum([shard_vars_inv[b]@shard_means[b] for b in range(M)]) )
beta_init = mu.reshape((1,dim))

def p(beta):
    return sum([ fitted_logposts[i].full_post_component(beta) for i in range(M) ])

rng = jax.random.PRNGKey(2)
burnin=10; steps=10000+burnin

rng, sample_rng = jax.random.split(rng)
t = time.time()
samples_approx, info = sample_nuts(sample_rng, p, 0.16, beta_init, steps, burnin)
sample_time = time.time() - t


jnp.save(experiment_dir + f'data/samples_{method}_{M}shards.npy', samples_approx)  
jnp.save(experiment_dir + f'data/times_{method}_{M}shards.npy', jnp.array([jnp.mean(train_times),sample_time]))
