#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time, os, multiprocessing
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)
import ucimlrepo
import jax
import jax.numpy as jnp
from dnc.mcmc import sample_nuts, sample_subposts

experiment_dir = 'experiments/powerplant/'
  
combined_cycle_power_plant = ucimlrepo.fetch_ucirepo(id=294) 
  
X = jnp.array(combined_cycle_power_plant.data.features )
y = jnp.array(combined_cycle_power_plant.data.targets )

ds_size = X.shape[0]
dim = X.shape[1] + 1 + 1
X = (X - jnp.mean(X, axis=0))/jnp.std(X, axis=0)
X = jnp.concatenate((jnp.ones((ds_size,1)), X), axis=1)


nu = 5.
prior_var = 100.
M = 8

@jax.vmap
def logprior(beta):
    sigma = beta[5]
    return (jnp.sum(jax.scipy.stats.norm.logpdf(beta[:5], scale=jnp.sqrt(prior_var)))
            + jax.scipy.stats.chi2.logpdf(sigma, df=1, scale=jnp.sqrt(prior_var))
            )

@jax.jit
def loglik(x, y, beta):
    means = x@beta[:,:5].transpose()
    return jax.scipy.stats.t.logpdf(y.flatten(), df=nu, loc=means.flatten(), scale=beta[:,5])

@jax.jit
def logpost(x, y, beta, prior_scaling = 1., lik_scaling=1.):
    return prior_scaling*logprior(beta) + lik_scaling*loglik(x, y, beta)

def loss(beta, batch):
    return -logpost(batch[0],batch[1],beta).flatten()[0]

def p(beta):
    return logpost(X,y,beta).flatten()[0]

burnin = 100    
n_samp = 50000 + burnin

# full posterior

rng = jax.random.PRNGKey(2024)
# prior mean 
beta_init = jnp.array([[0.,0.,0.,0.,0.,1.*jnp.sqrt(prior_var)]])

rng, samp_rng = jax.random.split(rng)
t = time.time()
samples_exact, a = sample_nuts(samp_rng, p, 10., beta_init, steps=n_samp, burnin=burnin); print(a, time.time()-t)

jnp.save(experiment_dir + 'data/samples_exact.npy', samples_exact)


######################################################## shard inference
shard_seeds = [i for i in range(10,15)]

for shard_seed in shard_seeds:
    print('\n Seed ', shard_seed)
    rng = jax.random.PRNGKey(shard_seed)
    
    rng, perm_rng = jax.random.split(rng)
    perm_idx = jax.random.permutation(perm_rng, jnp.array([i for i in range(ds_size)]))
    X_perm = X[perm_idx, :]
    y_perm = y[perm_idx, :]
    
    shards_x = jnp.array(jnp.split(X_perm, M))
    shards_y = jnp.array(jnp.split(y_perm,M))
    shards = [(shards_x[i], shards_y[i]) for i in range(M)]
    
    # prior mean    
    beta_inits = jnp.array([[0.,0.,0.,0.,0.,1.*jnp.sqrt(prior_var)]]*M).reshape((M, 1, dim))#sim_prior(init_rng, M).reshape((M, 1, dim))

    
    # paralellise over available cores
    C = len(jax.devices())
    shards_x = jnp.split(shards_x, jnp.ceil(M/C))
    shards_y = jnp.split(shards_y, jnp.ceil(M/C))
    beta_inits = jnp.split(beta_inits, jnp.ceil(M/C))
    
    # tuning step sizes
    burnin = 100
    tune_steps = 1000 + burnin
    
    rng, sample_rng = jax.random.split(rng)
    shard_keys = jax.random.split(sample_rng, M)
    shard_keys = jnp.split(shard_keys, jnp.ceil(M/C))     

    test_steps = jnp.array(jnp.exp(jnp.linspace(2,4,11)))
    step_sizes = []
    
    
    rates = []
    for step in test_steps:
        rates_i = []
        for i in range(int(jnp.ceil(M/C))):
            shard_samples_i, logpost_evals_i, gradlogpost_evals_i, info = sample_subposts(shard_keys[i], shards_x[i], shards_y[i], beta_inits[i], jnp.array([[step]]*shards_x[i].shape[0]), logpost, 1/M , 1., tune_steps, burnin, 1)
            rates_i.append(info)
        
        rates.append(jnp.concatenate(rates_i)) 
        
    rates = jnp.array(rates).transpose()
    step_sizes = jnp.array([ test_steps[jnp.min(jnp.where(rates[i]<0.775)[0])-1] for i in range(M) ]).reshape((M,1))
    
    step_sizes = jnp.split(step_sizes, jnp.ceil(M/C))

    # running chains
    rng, sample_rng = jax.random.split(rng)
    shard_keys = jax.random.split(sample_rng, M)
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
    jnp.save(experiment_dir + f'data/shard_samples_seed{shard_seed}_{M}shards.npy', shard_samples)   
    jnp.save(experiment_dir + f'data/shard_logpost_evals_seed{shard_seed}_{M}shards.npy', jnp.array(jnp.concatenate(logpost_evals)))       
    jnp.save(experiment_dir + f'data/shard_gradlogpost_evals_seed{shard_seed}_{M}shards.npy', jnp.array(jnp.concatenate(gradlogpost_evals)))   
    
    
    shard_samples = []
    logpost_evals = []
    gradlogpost_evals = []        
    
    ################################## swiss
    
    rng = jax.random.PRNGKey(shard_seed)
 
    # tuning step sizes
    rng, sample_rng = jax.random.split(rng)
    shard_keys = jax.random.split(sample_rng, M)
    shard_keys = jnp.split(shard_keys, jnp.ceil(M/C))     

    test_steps = jnp.array(jnp.exp(jnp.linspace(1,3,11)))
    step_sizes = []
    
    
    rates = []
    for step in test_steps:
        rates_i = []
        for i in range(int(jnp.ceil(M/C))):
            shard_samples_i, logpost_evals_i, gradlogpost_evals_i, info = sample_subposts(shard_keys[i], shards_x[i], shards_y[i], beta_inits[i], jnp.array([[step]]*shards_x[i].shape[0]), logpost, 1. , M, tune_steps, burnin, 1)
            rates_i.append(info)
        
        rates.append(jnp.concatenate(rates_i)) 
        
    rates = jnp.array(rates).transpose()
    step_sizes = jnp.array([ test_steps[jnp.min(jnp.where(rates[i]<0.775)[0])-1] for i in range(M) ]).reshape((M,1))
    step_sizes = jnp.split(step_sizes, jnp.ceil(M/C))

    # running chains
    rng, sample_rng = jax.random.split(rng)
    shard_keys = jax.random.split(sample_rng, M)
    shard_keys = jnp.split(shard_keys, jnp.ceil(M/C))        

    shard_samples = []
    logpost_evals = []
    gradlogpost_evals = []
    for i in range(int(jnp.ceil(M/C))):
        t = time.time()
        shard_samples_i, logpost_evals_i, gradlogpost_evals_i, info = sample_subposts(shard_keys[i], shards_x[i], shards_y[i], beta_inits[i], step_sizes[i], logpost, 1. , M, n_samp, burnin, 1)
        print(info)
        print(time.time()-t)

        
        shard_samples.append(shard_samples_i)
        logpost_evals.append(logpost_evals_i)
        gradlogpost_evals.append(gradlogpost_evals_i)
           
    shard_samples = jnp.array(jnp.concatenate(shard_samples))    
    jnp.save(experiment_dir + f'data/shard_samples_swiss_seed{shard_seed}_{M}shards.npy', shard_samples)   
    jnp.save(experiment_dir + f'data/shard_logpost_evals_swiss_seed{shard_seed}_{M}shards.npy', jnp.array(jnp.concatenate(logpost_evals)))       
    jnp.save(experiment_dir + f'data/shard_gradlogpost_evals_swiss_seed{shard_seed}_{M}shards.npy', jnp.array(jnp.concatenate(gradlogpost_evals)))   
        
    shard_samples = []
    logpost_evals = []
    gradlogpost_evals = []        
