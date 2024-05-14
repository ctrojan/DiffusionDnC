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
from dnc.transform_merges import matsqrt
experiment_dir = 'experiments/toy_logistic_regression_bigger/'


def predict(x,beta):
    return jax.nn.sigmoid(x @ beta.transpose())

@jax.vmap
def logprior(beta):
    return - (beta@beta.transpose())/2*5

@jax.jit
def loglik(x, y, beta):
    p_hat = jnp.clip(jax.nn.sigmoid(x@beta.transpose()), 1e-7, 1-1e-7)
    return y.transpose()@jnp.log(p_hat) + (1-y.transpose())@jnp.log(1-p_hat)

@jax.jit
def logpost(x, y, beta, prior_scaling = 1., lik_scaling=1.):
    return prior_scaling*logprior(beta) + lik_scaling*loglik(x, y, beta)

def loss(beta, batch):
    return -logpost(batch[0],batch[1],beta).flatten()[0]

####

def generate_data_logistic(rng, M, beta_true, x):
    ds_size = x.shape[0]
    dim = x.shape[1] + 1
    x = jnp.concatenate((jnp.ones((ds_size,1)), x),axis=1)
    y = jax.random.bernoulli(sample_rng, predict(x,beta_true))

    S = ds_size//M
    
    shards_x = jnp.array([ x[(i*S):((i+1)*S), ] for i in range(M)])
    shards_y = jnp.array([ y[(i*S):((i+1)*S), ] for i in range(M)])
    
    return shards_x, shards_y

seeds = [i for i in range(5)]

for seed in seeds:
    rng = jax.random.PRNGKey(seed)
    ds_size = 1000; dim = 100; M = 4
    
    rng, beta_rng = jax.random.split(rng, 2)
    beta_true = jnp.concatenate([jnp.array([-5]), jax.random.normal(beta_rng, (dim-1,))]).reshape(1,dim)
    rng, sample_rng = jax.random.split(rng, 2)    
    x = jax.random.normal(sample_rng,(ds_size,dim-1))
    
    rng, sample_rng = jax.random.split(rng, 2)
    shards_x, shards_y = generate_data_logistic(sample_rng, M, beta_true, x)
    X = jnp.concatenate(shards_x)
    y = jnp.concatenate(shards_y)
    
    print('Positive rate: ', jnp.sum(shards_y)/ds_size)
    print(jnp.sum(shards_y,axis=(1,2))*M/ds_size)
    
    ###
    def p(beta):
        return logpost(X,y,beta).flatten()[0]
        
    
    rng = jax.random.PRNGKey(seed+1)
    beta_init = jnp.zeros_like(beta_true)
    
    rng, samp_rng = jax.random.split(rng)
    steps = 50000
    step = 2.5e-2
    t = time.time()
    samples_exact, a = sample_nuts(samp_rng, p, step, beta_init, steps=steps+100, burnin=100); print(a, time.time()-t)
    
    jnp.save(experiment_dir + f'data/samples_exact_seed{seed}.npy', samples_exact)
    
    
    ######################################################## shard inference


    rng = jax.random.PRNGKey(seed+2)
    beta_inits = jnp.zeros((M,1,dim))
    step_sizes = 5e-2*jnp.ones((M,1))
    
    # running chains
    rng, sample_rng = jax.random.split(rng)
    shard_keys = jax.random.split(sample_rng, M)
    burnin = 100
    n_samp = steps + burnin
    
    # paralellise over available cores
    C = len(jax.devices())
    shards_x = jnp.split(shards_x, jnp.ceil(M/C))
    shards_y = jnp.split(shards_y, jnp.ceil(M/C))
    beta_inits = jnp.split(beta_inits, jnp.ceil(M/C))
    step_sizes = jnp.split(step_sizes, jnp.ceil(M/C))
    shard_keys = jnp.split(shard_keys, jnp.ceil(M/C))     
    
    shard_samples = []
    logpost_evals = []
    gradlogpost_evals = []
    for i in range(int(jnp.ceil(M/C))):
        t = time.time()
        shard_samples_i, logpost_evals_i, gradlogpost_evals_i, info = sample_subposts(shard_keys[i], shards_x[i], shards_y[i], beta_inits[i], step_sizes[i], logpost, 1/M , 1., n_samp, burnin, 1)
        print(info)
        print(time.time()-t)
        
        shard_samples.append(shard_samples_i)
        logpost_evals.append(logpost_evals_i)
        gradlogpost_evals.append(gradlogpost_evals_i)
           
    shard_samples = jnp.array(jnp.concatenate(shard_samples))   
    logpost_evals = jnp.array(jnp.concatenate(logpost_evals))   
    gradlogpost_evals = jnp.array(jnp.concatenate(gradlogpost_evals))   
        
    jnp.save(experiment_dir + f'data/shard_samples_seed{seed}.npy', shard_samples)   
    jnp.save(experiment_dir + f'data/shard_logpost_evals_seed{seed}.npy', logpost_evals)       
    jnp.save(experiment_dir + f'data/shard_gradlogpost_evals_seed{seed}.npy', gradlogpost_evals)   
        
    
    ################################## swiss
    rng = jax.random.PRNGKey(seed+2)
    
    step_sizes = step*jnp.ones((M,1))
    # running chains
    rng, sample_rng = jax.random.split(rng)
    shard_keys = jax.random.split(sample_rng, M)
    
    step_sizes = jnp.split(step_sizes, jnp.ceil(M/C))
    shard_keys = jnp.split(shard_keys, jnp.ceil(M/C))     
    
    shard_samples = []
    for i in range(int(jnp.ceil(M/C))):
        t = time.time()
        shard_samples_i, logpost_evals_i, gradlogpost_evals_i, info = sample_subposts(shard_keys[i], shards_x[i], shards_y[i], beta_inits[i], step_sizes[i], logpost, 1. , M, n_samp, burnin, 1)
        print(info)
        print(time.time()-t)
        
        shard_samples.append(shard_samples_i)
    
               
    shard_samples = jnp.array(jnp.concatenate(shard_samples))    
        
    jnp.save(experiment_dir + f'data/shard_samples_swiss_seed{seed}.npy', shard_samples)  
           
        