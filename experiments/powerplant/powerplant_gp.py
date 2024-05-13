#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
import time

from dnc.gp_merge import logdensity_GP
from dnc.mcmc import sample_nuts
experiment_dir = 'experiments/powerplant/'

shard_seeds = [i for i in range(10,15)]
thin_by = 50
M = 8

train_times = []
sample_times = []
####
for shard_seed in shard_seeds:
    rng = jax.random.PRNGKey(10+shard_seed)
    
    shard_samples = jnp.load(experiment_dir + f'data/shard_samples_seed{shard_seed}_{M}shards.npy')
    shard_logpost_evals = jnp.load(experiment_dir + f'data/shard_logpost_evals_seed{shard_seed}_{M}shards.npy')  
    
    samples_exact = jnp.load(experiment_dir + 'data/samples_exact.npy')
    dim = shard_samples.shape[-1]
     
    shard_means = jnp.mean(shard_samples, axis=1)
    shard_vars_inv = [jnp.linalg.inv(jnp.cov(shard_samples[i], rowvar=False)) for i in range(M)]
    V = jnp.linalg.inv(sum(shard_vars_inv))
    mu = V@( sum([shard_vars_inv[b]@shard_means[b] for b in range(M)]) )

    
    fitted_logposts = []
    rng, gp_rng = jax.random.split(rng, 2)
    gp_rng = jax.random.split(gp_rng, M)
    for i in range(M):
        fitted_logposts.append(logdensity_GP(shard_samples[i][::thin_by],shard_logpost_evals[i][::thin_by]))
        t = time.time()
        fitted_logposts[i].fit(gp_rng[i])
        train_times.append(time.time()-t)

    #
    
    
    rng = jax.random.PRNGKey(20+shard_seed)
    def p_prod(beta):
        return sum([ fitted_logposts[i].full_post_component(beta) for i in range(M) ])
    
    beta_init = mu.reshape((1,dim))
    burnin=10
    steps=10000 + burnin; 
    shape = (steps, dim)
    rng, sample_rng = jax.random.split(rng)
    inv_mass_matrix = V

    t = time.time()
    samples_approx, info = sample_nuts(sample_rng, p_prod, 10., beta_init, steps, burnin)
    sample_times.append(time.time() - t)
    print(jnp.mean(info), time.time()-t)


    jnp.save(experiment_dir + f'data/samples_seed{shard_seed}_{M}shards_gp.npy', samples_approx)

jnp.save(experiment_dir + f'data/times_{M}shards_gp.npy', jnp.array([jnp.mean(jnp.array(train_times)),jnp.mean(jnp.array(sample_times))]))
