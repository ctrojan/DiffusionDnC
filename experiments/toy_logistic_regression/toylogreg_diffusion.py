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


rng = jax.random.PRNGKey(1)
tf.random.set_seed(1)
experiment_dir = 'experiments/toy_logistic_regression/'
method = 'diff'
M = 15

samples_exact = jnp.load(experiment_dir + 'data/samples_exact.npy')
shard_samples = jnp.load(experiment_dir + f'data/{M}shards_samples.npy')
shard_evals = jnp.load(experiment_dir + f'data/{M}shards_gradlogpost_evals.npy')

shard_samples = jnp.array(shard_samples) 
shard_evals = jnp.array(shard_evals) 
dim = shard_samples.shape[-1]

shift, scale = normalising_transform(shard_samples)

rng, init_rng = jax.random.split(rng, 2)
init_rng = jax.random.split(init_rng, M)
score_models = [EBModel(1, 32, dim, 1e-3, init_rng[i]) for i in range(M)]

train_times = []
for i in range(M):
    rng, train_rng = jax.random.split(rng, 2)
    t = time.time()
    score_models[i].train_combined((shard_samples[i]-shift[i])@scale[i], shard_evals[i]@jnp.linalg.inv(scale[i]), 500, train_rng)
    train_times.append(time.time()-t)
train_times = jnp.array(train_times)

@jax.jit
def potential_prod(x, t): return sum([score_models[i].potential.apply(
    score_models[i].params, (x-shift[i])@scale[i], t) for i in range(M)])

#### just t=t0
rng = jax.random.PRNGKey(2)
t0 = 0.
burnin=10
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
samples_approx, info = sample_nuts(sample_rng, p, 0.16, beta_init, steps, burnin)
sample_time = time.time() - t

jnp.save(experiment_dir + f'data/samples_{method}_{M}shards.npy', samples_approx)   
jnp.save(experiment_dir + f'data/times_{method}_{M}shards.npy', jnp.array([jnp.mean(train_times),sample_time]))


###################################################### annealed hmc
rng = jax.random.PRNGKey(2)

inner_steps = 3
outer_steps = 300
shape = (10000, dim)
t0 = 1e-3
rng, sample_rng = jax.random.split(rng, 2)

rng, init_rng, sample_rng = jax.random.split(rng, 3)
x0 = jax.random.normal(init_rng, shape)@scale_M + mu

t = time.time()
samples_approx, info = annealed_mcmc_sampling(
    sample_rng, potential_prod, (1000, dim), inner_steps, outer_steps, dt=.16, t0=t0, x_0=x0)

jnp.save(experiment_dir + f'data/samples_{method}_annealed_{M}shards.npy', samples_approx)   
