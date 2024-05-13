#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import tensorflow as tf
import jax
import jax.numpy as jnp
 
from dnc.diffusion_merge import EBModel, annealed_mcmc_sampling, normalising_transform, matsqrt
from dnc.mcmc import sample_nuts
experiment_dir = 'experiments/spambase/'

shard_seeds = [i for i in range(10,15)]
train_times = []
sample_times = []

####
for shard_seed in shard_seeds:
    rng = jax.random.PRNGKey(10+shard_seed)
    tf.random.set_seed(10+shard_seed)
    
    shard_samples = jnp.load(experiment_dir + f'data/shard_samples_seed{shard_seed}.npy')
    shard_evals = jnp.load(experiment_dir + f'data/shard_gradlogpost_evals_seed{shard_seed}.npy')  
    
    dim = shard_samples.shape[-1]; M = shard_samples.shape[0]

    
    # 
    
    shift, scale = normalising_transform(shard_samples)
    
    rng, init_rng = jax.random.split(rng, 2)
    init_rng = jax.random.split(init_rng, M)
    score_models = [EBModel(1, 32, dim, 1e-3, init_rng[i]) for i in range(M)]
    
    for i in range(M):
        rng, train_rng = jax.random.split(rng, 2)
        t = time.time()
        score_models[i].train_combined((shard_samples[i]-shift[i])@scale[i], shard_evals[i]@jnp.linalg.inv(scale[i]), 100, train_rng)
        train_times.append(time.time()-t)
              
    
    #
    
    
    rng = jax.random.PRNGKey(20+shard_seed)
    @jax.jit
    def potential_prod(x, t): return sum([score_models[i].potential.apply(
        score_models[i].params, (x-shift[i])@scale[i], t) for i in range(M)])

    shard_vars_inv = [scale[i]@scale[i].transpose() for i in range(M)]
    V = jnp.linalg.inv(sum(shard_vars_inv))
    mu = V@( sum([shard_vars_inv[b]@shift[b] for b in range(M)]) )
    scale_M = matsqrt(V)
    
    ##### fixed time
    
    t0=0.
    def p_prod(x):
        return potential_prod(x,jnp.full(1, t0))[0]
    
    beta_init = mu.reshape((1,dim))
    burnin=10
    steps=10000 + burnin
    shape = (steps, dim)
    
    inv_mass_matrix = V # this helps reduce number of leapfrog steps required
    rng, sample_rng = jax.random.split(rng)
    beta_init = mu.reshape((1,dim))
    t = time.time()
    samples_approx, info = sample_nuts(sample_rng, p_prod, 5e-1, beta_init, steps, burnin, inv_mass_matrix)
    sample_times.append(time.time() - t)

    jnp.save(experiment_dir + f'data/samples_seed{shard_seed}_diff.npy', samples_approx)


    # #### annealed
    
    inner_steps = 3
    outer_steps = 300
    lfrogs = 7
    t0 = 1e-3 
    shape = (10000, dim)
    
    #    
    
    rng, init_rng, sample_rng = jax.random.split(rng, 3)
    x0 = jax.random.normal(init_rng, shape)@scale_M + mu    
    t = time.time()
    samples_approx, info = annealed_mcmc_sampling(
        sample_rng, potential_prod, shape, inner_steps, outer_steps, x_0=x0, dt=5e-1, t0=t0, leapfrog_steps = lfrogs, inv_mass_matrix=V) 
    jnp.save(experiment_dir + f'data/samples_seed{shard_seed}_diff_annealed.npy', samples_approx)
    
train_times = jnp.array(train_times)
sample_times = jnp.array(sample_times)
jnp.save(experiment_dir + f'data/times_diff.npy', jnp.array([jnp.mean(train_times),jnp.mean(sample_times)]))
