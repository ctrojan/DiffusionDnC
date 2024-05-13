#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 13:42:19 2024

@author: trojanc1
"""

import jax
from jax import numpy as jnp


def d_mah(samps_approx, samps_exact):
    # Mahalanobis distance
    dim = samps_approx.shape[1]
    
    u_a = jnp.mean(samps_approx, axis=0).reshape(1,dim)
    u_f = jnp.mean(samps_exact, axis=0).reshape(1,dim)
    V_f = jnp.cov(samps_exact, rowvar=False)
    
    return jnp.sqrt( (u_a - u_f)@jnp.linalg.inv(V_f)@(u_a - u_f).transpose() ).flatten()[0]


def d_skew(samps_approx, samps_exact):
    # Mean absolute skew deviation
    dim = samps_approx.shape[1]

    u_a = jnp.mean(samps_approx, axis=0).reshape(1,dim)
    s_a = jnp.var(samps_approx, axis=0)
    g_a = jnp.mean((samps_approx - u_a)**3, axis=0)/s_a**(3/2)
    
    u_f = jnp.mean(samps_exact, axis=0).reshape(1,dim)    
    s_f = jnp.var(samps_exact, axis=0)
    g_f = jnp.mean((samps_exact - u_f)**3, axis=0)/s_f**(3/2)
    
    return jnp.mean(jnp.abs(g_a - g_f))


def d_iad(samps_approx, samps_exact, res=128, sigmas=5):
    # Integrated absolute difference
    dim = samps_approx.shape[1]
    
    iad = 0
    for j in range(dim):
        lower_endpt = min( [jnp.mean(samps_approx[:,j]) - sigmas*jnp.sqrt(jnp.var(samps_approx[:,j])),
                               jnp.mean(samps_exact[:,j]) - sigmas*jnp.sqrt(jnp.var(samps_exact[:,j]))]
                              )
        upper_endpt = max( [jnp.mean(samps_approx[:,j]) + sigmas*jnp.sqrt(jnp.var(samps_approx[:,j])),
                               jnp.mean(samps_exact[:,j]) + sigmas*jnp.sqrt(jnp.var(samps_exact[:,j]))]
                              )
        
        eval_pts = jnp.linspace(lower_endpt, upper_endpt, res)
        
        pi_a = jax.scipy.stats.gaussian_kde(samps_approx[:,j].transpose())
        pi_a_evals = pi_a(eval_pts)
        
        pi_f = jax.scipy.stats.gaussian_kde(samps_exact[:,j].transpose())
        pi_f_evals = pi_f(eval_pts)
        
        ad_evals = jnp.abs(pi_a_evals - pi_f_evals)
        
        iad += jax.scipy.integrate.trapezoid(ad_evals, eval_pts)
        
    return iad/(2*dim)
        
        
        
        
        
        
        
