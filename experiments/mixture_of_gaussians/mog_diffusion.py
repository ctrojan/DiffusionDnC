#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import jax
import jax.numpy as jnp
import tensorflow as tf
import time

from dnc.diffusion_merge import EBModel, annealed_mcmc_sampling, matsqrt, normalising_transform
from dnc.plotting import kde_contour, kde_1d

experiment_dir = 'experiments/mixture_of_gaussians/'
method = 'diff'

samples_exact = jnp.load(experiment_dir + 'data/samples_exact.npy')
shard_samples = jnp.load(experiment_dir + 'data/shard_samples.npy')
shard_evals = jnp.load(experiment_dir + 'data/shard_gradlogpost_evals.npy')

rng = jax.random.PRNGKey(1)
tf.random.set_seed(1)
shard_samples = jnp.array(shard_samples) 
shard_evals = jnp.array(shard_evals) 
dim = shard_samples.shape[-1]; M = shard_samples.shape[0]

#

shift, scale = normalising_transform(shard_samples)

rng, init_rng = jax.random.split(rng, 2)
init_rng = jax.random.split(init_rng, M)
score_models = [EBModel(1, 32, dim, 1e-3, init_rng[i]) for i in range(M)]

ts = []
for i in range(M):
    rng, train_rng = jax.random.split(rng, 2)
    t = time.time()
    score_models[i].train_combined((shard_samples[i]-shift[i])@scale[i], shard_evals[i]@jnp.linalg.inv(scale[i]), 500, train_rng)
    ts.append(time.time()-t)
ts = jnp.array(ts)


@jax.jit
def potential_prod(x, t): 
    return sum([score_models[i].potential.apply(score_models[i].params, (x-shift[i])@scale[i], t) for i in range(M)])
    
shard_vars_inv = [scale[i]@scale[i].transpose() for i in range(M)]
V = jnp.linalg.inv(sum(shard_vars_inv))
mu = V@( sum([shard_vars_inv[b]@shift[b] for b in range(M)]) )
scale_M = matsqrt(V)

# intermediate distributions
rng = jax.random.PRNGKey(2)
inner_steps = 1
outer_steps = 150
shape = (10000, dim)

rng, init_rng, sample_rng = jax.random.split(rng, 3)
x0 = jax.random.normal(init_rng, shape)@scale_M + mu
jnp.save(experiment_dir + f'data/samples_{method}_t1.npy', x0)  

t0 = 0.2
outer_steps = int((1-t0)*300)
samples_approx, info = annealed_mcmc_sampling(
    sample_rng, potential_prod, shape, inner_steps, outer_steps, x_0=x0, dt=.025, t0=t0)
jnp.save(experiment_dir + f'data/samples_{method}_t{t0}.npy', samples_approx)  

t0 = 0.12
outer_steps = int((1-t0)*300)
samples_approx, info = annealed_mcmc_sampling(
    sample_rng, potential_prod, shape, inner_steps, outer_steps, x_0=x0, dt=.025, t0=t0)
jnp.save(experiment_dir + f'data/samples_{method}_t{t0}.npy', samples_approx)  

# target distribution
rng = jax.random.PRNGKey(2)
rng, init_rng, sample_rng = jax.random.split(rng, 3)
inner_steps = 1
outer_steps = 300
t0 = 1e-3
shape = (10000, dim)

t = time.time()
x0 = jax.random.normal(init_rng, shape)@scale_M + mu
samples_approx, info = annealed_mcmc_sampling(
    sample_rng, potential_prod, shape, inner_steps, outer_steps, x_0=x0, dt=.025, t0=t0) # .25 for 2d
t = time.time() - t

jnp.save(experiment_dir + f'data/samples_{method}.npy', samples_approx)  
jnp.save(experiment_dir + f'data/times_{method}.npy', jnp.array([jnp.mean(ts),t]))
