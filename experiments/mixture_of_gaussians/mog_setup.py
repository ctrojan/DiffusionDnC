#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
import blackjax
from blackjax.mcmc.hmc import HMCState
experiment_dir = 'experiments/mixture_of_gaussians/'


s = jnp.sqrt(0.2)

def logprior(beta):
    return - (beta@beta.transpose())/(2*1.**2)

def loglik(y, beta):
    lls = jnp.log(jnp.sum( jnp.exp(-(y - beta)**2/(2*s**2))/beta.shape[1], axis = 1 ))
    return jnp.sum(lls)

def logpost(y, beta, prior_scaling = 1., lik_scaling=1.):
    return prior_scaling*logprior(beta).flatten() + lik_scaling*loglik(y, beta).flatten()


def generate_data(rng, ds_size, M, beta):
    rng = jax.random.split(rng, 2)
    means = jax.random.choice(rng[0], beta, (ds_size,), axis=1).reshape((ds_size, 1))
    y = means + jax.random.normal(rng[1],(ds_size,1))*s

    S = ds_size//M
    
    shards_y = jnp.array([ y[(i*S):((i+1)*S), ] for i in range(M)])
    
    return shards_y

def sample_post_mog(rng, y, beta_init, step_size, logpost, prior_scaling = 1., likelihood_scaling = 1., steps=10100, burnin=100):
    # MCMC sampling with label switching moves
    
    dim = beta_init.shape[1]
    inv_mass_matrix = jnp.ones((dim))  
    keys = jax.random.split(rng, steps)
    
    def p(beta):
        return logpost(y, beta, prior_scaling, likelihood_scaling).flatten()[0]
    
    sampler = blackjax.nuts(p, step_size, inv_mass_matrix)    
    initial_state= sampler.init(beta_init)
    
    @jax.jit
    def inner_step(states, rng):
        perm_rng, step_rng = jax.random.split(rng)
        states = HMCState(jax.random.permutation(perm_rng, states.position, axis=1), states.logdensity, states.logdensity_grad)
        states, info = sampler.step(step_rng, states)
        return states, (states.position, states.logdensity, states.logdensity_grad, info)
    
    state, (samples,logpost_evals,gradlogpost_evals,info) = jax.lax.scan(inner_step, initial_state, keys)
    samples = samples[burnin:,:].reshape((steps-burnin,dim))
    logpost_evals = logpost_evals[burnin:].reshape((steps-burnin, 1))
    gradlogpost_evals = gradlogpost_evals[burnin:,:].reshape((steps-burnin, dim))
    
    return samples, logpost_evals, gradlogpost_evals, jnp.mean(info.acceptance_rate[burnin:])



rng = jax.random.PRNGKey(2023)
ds_size = 2000; M = 4
rng, ds_rng = jax.random.split(rng, 2) 
beta = jnp.array([[-.4,.4,0.]])     
shards_y = generate_data(ds_rng, ds_size, M, beta)
y = jnp.concatenate(shards_y)

# sample full posterior
rng = jax.random.PRNGKey(2024)
rng, sample_rng = jax.random.split(rng, 2) 
samples_exact, logpost_evals, gradlogpost_evals, accept_rate  = sample_post_mog(sample_rng, y, step_size=0.02, logpost=logpost, beta_init=beta)

jnp.save(experiment_dir + 'data/samples_exact.npy', samples_exact)  

# sample subposteriors
rng = jax.random.PRNGKey(2025)
rng, sample_rng = jax.random.split(rng, 2) 
shard_samples = []
shard_gradlogpost_evals = []
shard_logpost_evals = []
for i in range(M):
    samples, logpost_evals, gradlogpost_evals, accept_rate  = sample_post_mog(sample_rng, shards_y[i], step_size=0.03, logpost=logpost, beta_init=beta)
    
    shard_samples.append(samples)
    shard_gradlogpost_evals.append(gradlogpost_evals)
    shard_logpost_evals.append(logpost_evals)

jnp.save(experiment_dir + 'data/shard_samples.npy', shard_samples)   
jnp.save(experiment_dir + 'data/shard_logpost_evals.npy', shard_logpost_evals)       
jnp.save(experiment_dir + 'data/shard_gradlogpost_evals.npy', shard_gradlogpost_evals)  
 

# sample subposteriors (SwISS formulation)
rng = jax.random.PRNGKey(2025)
rng, sample_rng = jax.random.split(rng, 2) 
shard_samples = []
shard_gradlogpost_evals = []
shard_logpost_evals = []
for i in range(M):
    samples, logpost_evals, gradlogpost_evals, accept_rate  = sample_post_mog(sample_rng, shards_y[i], step_size=0.02, logpost=logpost, beta_init=beta, prior_scaling = 1., likelihood_scaling = M)
    
    shard_samples.append(samples)
    shard_gradlogpost_evals.append(gradlogpost_evals)
    shard_logpost_evals.append(logpost_evals)


jnp.save(experiment_dir + 'data/shard_samples_swiss.npy', shard_samples)   
