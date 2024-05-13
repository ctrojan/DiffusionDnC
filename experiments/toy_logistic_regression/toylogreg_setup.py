#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
from dnc.mcmc import sample_subposts, sample_post
experiment_dir = 'experiments/toy_logistic_regression/'


def predict(x,beta):
    return 1/(1 + jnp.exp(-x @ beta.transpose()))

@jax.vmap
def logprior(beta):
    return - (beta@beta.transpose())/10

def loglik(x, y, beta):
    p_hat = jnp.clip((1/(1 + jnp.exp(-x @ beta.transpose())) ), 1e-7, 1-1e-7)
    return y.transpose()@jnp.log(p_hat) + (1-y.transpose())@jnp.log(1-p_hat)

@jax.jit
def logpost(x, y, beta, prior_scaling = 1., lik_scaling=1.):
    return prior_scaling*logprior(beta) + lik_scaling*loglik(x, y, beta)


def generate_data_logistic(rng, ds_size, dim, M, beta_true=None, x=None):
    if x is None:
        x = 10*jax.random.uniform(rng, (ds_size,dim))
    x = jnp.concatenate((jnp.ones((ds_size,1)), x),axis=1)
    
    if beta_true is None:
        beta_true = jnp.ones((1,dim+1))
    rng, sample_rng = jax.random.split(rng, 2)
    y = jax.random.bernoulli(sample_rng, predict(x,beta_true))

    S = ds_size//M
    
    shards_x = jnp.array([ x[(i*S):((i+1)*S), ] for i in range(M)])
    shards_y = jnp.array([ y[(i*S):((i+1)*S), ] for i in range(M)])
    
    return shards_x, shards_y


rng = jax.random.PRNGKey(2023)

ds_size = 1000; dim = 1; M = 15
beta_init = -jnp.ones((M, dim+1))
step_sizes = .6*jnp.ones((M, 1))

rng, sample_rng = jax.random.split(rng, 2)
beta_true = jnp.array([-3,-3]); x = 0.5 + jax.random.normal(sample_rng,(ds_size,dim))

rng, sample_rng = jax.random.split(rng, 2)
shards_x, shards_y = generate_data_logistic(sample_rng, ds_size, dim, M, beta_true, x)

sample_rng = jax.random.PRNGKey(2025)
samples_exact, logpost_evals, gradlogpost_evals, info = sample_post(sample_rng, jnp.concatenate(shards_x), jnp.concatenate(shards_y), beta_init[0,:].reshape((1,dim+1)), 0.22, logpost, steps=10100, burnin=100)
jnp.save(experiment_dir + 'data/samples_exact.npy', samples_exact) 

sample_rng = jax.random.PRNGKey(2024)
rng, mcmc_rng = jax.random.split(rng, 2)
shard_keys = jax.random.split(mcmc_rng, M)

shard_samples = []
shard_logpost_evals = []
shard_gradlogpost_evals = []
for i in range(M):
    samps, logpost_evals, gradlogpost_evals, info = sample_post(shard_keys[i], shards_x[i,:], shards_y[i,:], beta_init[i,:].reshape((1,dim+1)), step_sizes[i], logpost, prior_scaling = 1/M , likelihood_scaling = 1., steps=10100, burnin=100)
    shard_samples.append(samps)
    shard_logpost_evals.append(logpost_evals)
    shard_gradlogpost_evals.append(gradlogpost_evals)
shard_samples = jnp.array(shard_samples)
shard_logpost_evals = jnp.array(shard_logpost_evals)
shard_gradlogpost_evals = jnp.array(shard_gradlogpost_evals)
    
jnp.save(experiment_dir + f'data/{M}shards_samples.npy', shard_samples)   
jnp.save(experiment_dir + f'data/{M}shards_logpost_evals.npy', shard_logpost_evals)       
jnp.save(experiment_dir + f'data/{M}shards_gradlogpost_evals.npy', shard_gradlogpost_evals) 


sample_rng = jax.random.PRNGKey(2024)
step_sizes = .17*jnp.ones((M, 1))
shard_samples = []
shard_evals = []
for i in range(M):
    samps, logpost_evals, gradlogpost_evals, info = sample_post(shard_keys[i], shards_x[i,:], shards_y[i,:], beta_init[i,:].reshape((1,dim+1)), step_sizes[i], logpost, prior_scaling = 1., likelihood_scaling = M, steps=10100, burnin=100)
    shard_samples.append(samps)
    shard_evals.append(gradlogpost_evals)
shard_samples = jnp.array(shard_samples)
shard_evals = jnp.array(shard_evals)
    
jnp.save(experiment_dir + f'data/{M}shards_samples_swiss.npy', shard_samples)   
