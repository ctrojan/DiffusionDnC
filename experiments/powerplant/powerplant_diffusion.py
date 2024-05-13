#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import tensorflow as tf
import jax
import jax.numpy as jnp
 
from dnc.diffusion_merge import EBModel, annealed_mcmc_sampling, normalising_transform, matsqrt
from dnc.mcmc import sample_nuts
experiment_dir = 'experiments/powerplant/'
samples_exact = jnp.load(experiment_dir + 'data/samples_exact.npy')

method = 'combined'
M = 8
shard_seeds = [i for i in range(10,15)]
train_times = []
sample_times = []

####
for shard_seed in shard_seeds:
    print(shard_seed)
    rng = jax.random.PRNGKey(10+shard_seed)
    tf.random.set_seed(10+shard_seed)
    
    shard_samples = jnp.load(experiment_dir + f'data/shard_samples_seed{shard_seed}_{M}shards.npy')
    shard_evals = jnp.load(experiment_dir + f'data/shard_gradlogpost_evals_seed{shard_seed}_{M}shards.npy')  
    
    dim = shard_samples.shape[-1]     
    
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
    @jax.jit
    def p_prod(x):
        return potential_prod(x,jnp.full(1, t0))[0]
    
    beta_init = mu.reshape((1,dim))
    burnin=10
    steps=10000 + burnin
    shape = (steps, dim)
    rng, sample_rng = jax.random.split(rng)
    t = time.time()
    samples_approx, info = sample_nuts(sample_rng, p_prod, 10., beta_init, steps, burnin)
    sample_times.append(time.time() - t)
    print(jnp.mean(info), time.time()-t)
    
    jnp.save(experiment_dir + f'data/samples_seed{shard_seed}_{M}shards_diff{method}.npy', samples_approx)


    # #### annealed
    
    inner_steps = 3
    outer_steps = 300
    lfrogs = 3
    t0 = 1e-3 
    shape = (10000, dim)
    
    #    
    
    rng, init_rng, sample_rng = jax.random.split(rng, 3)
    x0 = jax.random.normal(init_rng, shape)@scale_M + mu    
    t = time.time()
    samples_approx, info = annealed_mcmc_sampling(
        sample_rng, potential_prod, shape, inner_steps, outer_steps, x_0=x0, dt=10., t0=t0, leapfrog_steps = lfrogs) 
    print(jnp.mean(info, axis=[1,2])[::-10][::-1], time.time()-t)
    print(jnp.mean(info[-30::]))
    jnp.save(experiment_dir + f'data/samples_seed{shard_seed}_{M}shards_diff{method}_annealed.npy', samples_approx)
    

jnp.save(experiment_dir + f'data/times_{M}shards_diff{method}.npy', jnp.array([jnp.mean(jnp.array(train_times)),jnp.mean(jnp.array(sample_times))]))
