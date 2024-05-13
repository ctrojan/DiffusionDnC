#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time, os, multiprocessing
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)
import ucimlrepo
import jax
import jax.numpy as jnp
from dnc.mcmc import opt_adam, sample_nuts, sample_subposts

experiment_dir = 'experiments/spambase/'

spamdb = ucimlrepo.fetch_ucirepo(id=94)

X = jnp.array(spamdb.data.features)[:-1, :]
y = jnp.array(spamdb.data.targets)[:-1,:]

ds_size = X.shape[0]
X = jnp.concatenate((jnp.ones((ds_size,1)), X),axis=1)
dim = X.shape[1]

def predict(x,beta):
    return jax.nn.sigmoid(x @ beta.transpose())

@jax.vmap
def logprior(beta):
    return - (beta@beta.transpose())/10

@jax.jit
def loglik(x, y, beta):
    p_hat = jnp.clip(jax.nn.sigmoid(x@beta.transpose()), 1e-7, 1-1e-7)
    return y.transpose()@jnp.log(p_hat) + (1-y.transpose())@jnp.log(1-p_hat)

@jax.jit
def logpost(x, y, beta, prior_scaling = 1., lik_scaling=1.):
    return prior_scaling*logprior(beta) + lik_scaling*loglik(x, y, beta)

def loss(beta, batch):
    return -logpost(batch[0],batch[1],beta).flatten()[0]

def p(beta):
    return logpost(X,y,beta).flatten()[0]
    

rng = jax.random.PRNGKey(2024)
beta_init = jax.random.normal(rng, (1,dim))*jnp.sqrt(5)
beta_init = opt_adam(loss, (X,y), beta_init, epochs=20, lr=1e-2)
jnp.mean(beta_init)

rng, samp_rng = jax.random.split(rng)
t = time.time()
samples_exact, a = sample_nuts(samp_rng, p, 1.5e-4, beta_init, steps=50100, burnin=100); print(a, time.time()-t)
    

jnp.save(experiment_dir + 'data/samples_exact.npy', samples_exact)


######################################################## shard inference
shard_seeds = [i for i in range(10,15)]
M = 4


for shard_seed in shard_seeds:
    print('\n', shard_seed)
    rng = jax.random.PRNGKey(shard_seed)
    
    rng, perm_rng = jax.random.split(rng)
    perm_idx = jax.random.permutation(perm_rng, jnp.array([i for i in range(ds_size)]))
    X_perm = X[perm_idx, :]
    y_perm = y[perm_idx, :]
    
    shards_x = jnp.split(X_perm, M)
    shards_y = jnp.split(y_perm,M)
    shards = [(shards_x[i], shards_y[i]) for i in range(M)]
    
    
    # finding start point
    beta_inits = jax.random.normal(rng, (M,dim))*jnp.sqrt(5)
    beta_inits = jnp.split(beta_inits, M)
    for i in range(M):
        beta_inits[i] = opt_adam(loss, shards[i], beta_inits[i], epochs=20, lr=1e-2)
    beta_inits = jnp.array(beta_inits)
    
    # tuning step sizes
    tune_steps = 5000
    rng, sample_rng = jax.random.split(rng)
    shard_keys = jax.random.split(sample_rng, M)
    test_steps = jnp.array(jnp.linspace(1e-4,6e-4,11))
    step_sizes = []
    
    
    rates = []
    for step in test_steps:
        shard_samples, logpost_evals, gradlogpost_evals, info = sample_subposts(shard_keys, jnp.array(shards_x), jnp.array(shards_y), beta_inits, jnp.array([[step]]*M), logpost, 1/M , 1., tune_steps, 50, 1)
        rates.append(info)        
    rates = jnp.array(rates).transpose()
    diffs = jnp.abs(rates - .8)
    step_sizes = jnp.array([ test_steps[jnp.where(diffs[i]==jnp.min(diffs[i]))] for i in range(M) ]).reshape((M,1))
    
    # running chains
    rng, sample_rng = jax.random.split(rng)
    shard_keys = jax.random.split(sample_rng, M)
    burnin = 100
    n_samp = 50000 + burnin
    
    t = time.time()
    shard_samples, logpost_evals, gradlogpost_evals, info = sample_subposts(shard_keys, jnp.array(shards_x), jnp.array(shards_y), beta_inits, step_sizes, logpost, 1/M , 1., n_samp, burnin, 1)
    print(info, time.time()-t)
        
        
    jnp.save(experiment_dir + f'data/shard_samples_seed{shard_seed}.npy', shard_samples)   
    jnp.save(experiment_dir + f'data/shard_logpost_evals_seed{shard_seed}.npy', logpost_evals)       
    jnp.save(experiment_dir + f'data/shard_gradlogpost_evals_seed{shard_seed}.npy', gradlogpost_evals)   
        
    
    ################################## swiss
    
    # tuning step sizes
    tune_steps = 5000
    rng, sample_rng = jax.random.split(rng)
    shard_keys = jax.random.split(sample_rng, M)
    test_steps = jnp.array(jnp.linspace(1e-4,3e-4,11))
    step_sizes = []
    
    
    rates = []
    for step in test_steps:
        shard_samples, logpost_evals, gradlogpost_evals, info = sample_subposts(shard_keys, jnp.array(shards_x), jnp.array(shards_y), beta_inits, jnp.array([[step]]*M), logpost, 1., M, tune_steps, 50, 1)
        rates.append(info)        
    rates = jnp.array(rates).transpose()
    diffs = jnp.abs(rates - .8)
    step_sizes = jnp.array([ test_steps[jnp.where(diffs[i]==jnp.min(diffs[i]))] for i in range(M) ]).reshape((M,1))
    
    # running chains
    rng, sample_rng = jax.random.split(rng)
    shard_keys = jax.random.split(sample_rng, M)
    
    t = time.time()
    shard_samples, logpost_evals, gradlogpost_evals, info = sample_subposts(shard_keys, jnp.array(shards_x), jnp.array(shards_y), beta_inits, step_sizes, logpost, 1. , M, n_samp, burnin, 1)
    print(info, time.time()-t)
        
    jnp.save(experiment_dir + f'data/shard_samples_swiss_seed{shard_seed}.npy', shard_samples)  
           
        