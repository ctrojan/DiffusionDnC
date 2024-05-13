import functools
import jax
import jax.numpy as jnp
import blackjax
import optax
from tqdm import trange
import tensorflow as tf

def opt_adam(loss, data, beta_init, epochs, batch_size=32, lr=1e-3):
    tf.random.set_seed(1)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(beta_init)
    
    dataset = tf.data.Dataset.from_tensor_slices(data).shuffle(1024).batch(batch_size, drop_remainder=True)

    
    # @jax.jit
    def update_step(beta, batch, opt_state):
        val, grads = jax.value_and_grad(loss, has_aux=False)(beta, batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        beta = optax.apply_updates(beta, updates)
        return val, beta, opt_state
    
    beta = beta_init
    mean_losses = []
    for i in (pbar := trange(epochs)):

        total_loss = 0.  
        n_batches = 0          
        for batch in dataset.as_numpy_iterator(): 
            loss_eval, beta, opt_state = update_step(beta, batch, opt_state)
            total_loss += loss_eval
            n_batches += 1
            
        mean_loss = total_loss/n_batches
        mean_losses.append(mean_loss)
        pbar.set_postfix({"Loss":"{:.2f} ".format(mean_loss)})
        
    return beta


def sample_nuts(rng, p, step_size, beta_init, steps=10000, burnin=100, inv_mass_matrix=None):
    # Sample from given log density
    dim = beta_init.shape[1]
    if inv_mass_matrix is None:
        inv_mass_matrix = jnp.ones((dim))  
    keys = jax.random.split(rng, steps)
    
    sampler = blackjax.nuts(p, step_size, inv_mass_matrix)
    initial_state= sampler.init(beta_init)
    
    @jax.jit
    def inner_step(states, rng):
        states, info = sampler.step(rng, states)
        return states, (states.position, states.logdensity, info)
    
    state, (samples,potential,info) = jax.lax.scan(inner_step, initial_state, keys)
    samples = samples[burnin:,:].reshape((steps-burnin,dim))
    
    return samples, jnp.mean(info.acceptance_rate[burnin:])


def sample_post(rng, x, y, beta_init, step_size, logpost, prior_scaling = 1., likelihood_scaling = 1., steps=10000, burnin=100):
    # Single posterior sampling & tuning
    
    dim = beta_init.shape[1]
    inv_mass_matrix = jnp.ones((dim))  
    keys = jax.random.split(rng, steps)
    
    def p(beta):
        return logpost(x, y, beta, prior_scaling, likelihood_scaling).flatten()[0]
    
    sampler = blackjax.nuts(p, step_size, inv_mass_matrix)    
    initial_state= sampler.init(beta_init)
    
    @jax.jit
    def inner_step(states, rng):
        states, info = sampler.step(rng, states)
        return states, (states.position, states.logdensity, states.logdensity_grad, info)
    state, (samples,logpost_evals,gradlogpost_evals,info) = jax.lax.scan(inner_step, initial_state, keys)
    samples = samples[burnin:,:].reshape((steps-burnin,dim))
    logpost_evals = logpost_evals[burnin:].reshape((steps-burnin, 1))
    gradlogpost_evals = gradlogpost_evals[burnin:,:].reshape((steps-burnin, dim))
    
    return samples, logpost_evals, gradlogpost_evals, jnp.mean(info.acceptance_rate[burnin:])


@functools.partial(jax.pmap, in_axes=(0,0,0,0,0,None,None,None,None,None,None), static_broadcasted_argnums=(5,8,9,10))
def sample_subposts(rng, x, y, beta_init, step_size, logpost, prior_scaling, likelihood_scaling, steps, burnin, thin_by):

    dim = beta_init.shape[1]
    inv_mass_matrix = jnp.ones((dim))
    keys = jax.random.split(rng, steps)

    def p(beta):
        return logpost(x, y, beta, prior_scaling, likelihood_scaling).flatten()[0]

    sampler = blackjax.nuts(p, step_size, inv_mass_matrix)

    initial_state = sampler.init(beta_init.reshape(1, dim))

    def inner_step(states, rng):
        states, info = sampler.step(rng, states)
        return states, (states.position, states.logdensity, states.logdensity_grad, info)

    state, (samples, logpost_evals, gradlogpost_evals, info) = jax.lax.scan(
        inner_step, initial_state, keys)
    samples = samples[burnin:, :].reshape(
        (steps-burnin, dim))[0:-1:thin_by, :]
    logpost_evals = logpost_evals[burnin:].reshape(
        (steps-burnin, 1))[0:-1:thin_by, :]
    gradlogpost_evals = gradlogpost_evals[burnin:, :].reshape(
        (steps-burnin, dim))[0:-1:thin_by, :]

    return samples, logpost_evals, gradlogpost_evals, jnp.mean(info.acceptance_rate[burnin:])