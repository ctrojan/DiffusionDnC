import jax
import functools
import jax.numpy as jnp
from tqdm import trange
import blackjax

from flax import linen as nn
import optax
import tensorflow as tf
from dnc.transform_merges import matsqrt

###############################################################################

def normalising_transform(shard_samples, matsqrt=matsqrt):
    M = shard_samples.shape[0]
    scale = [jnp.linalg.inv(matsqrt(jnp.cov(shard_samples[i], rowvar=False))) for i in range(M)]
    shift = jnp.mean(shard_samples, axis=1)

    return shift, scale

class VP_SDE():
    def __init__(self, beta_min=0.1, beta_max=20.):
        self.beta_min = beta_min
        self.beta_max = beta_max
        
    def beta(self, t):
        return self.beta_min + (self.beta_max - self.beta_min)*t
    
    def drift_coeff(self, t):
        return -0.5*self.beta(t)
    
    def mean(self, t):
        return jnp.exp(-0.5*t *self.beta_min - 0.25*t**2*(self.beta_max - self.beta_min))
        
    def diffusion_coeff(self, t):
        return jnp.sqrt(self.beta(t))
    
    def variance(self, t):
        return 1 - jnp.exp(-t *self.beta_min - 0.5*t**2*(self.beta_max - self.beta_min))
    
sde = VP_SDE()


def reverse_sde_solver(rng, shape, score, n_steps=1000, t0=0.001):
    rng, init_rng = jax.random.split(rng, 2)
    n_samples = shape[0]
    dim = shape[1]
    x_0 = jax.random.normal(init_rng,shape)*jnp.sqrt(sde.variance(1.))
    dt = (1-t0)/n_steps
    
    @jax.jit
    def step(carry, t):
        x_t, rng = carry
        rng, step_rng = jax.random.split(rng)
        x_t += ( - sde.drift_coeff(t)*x_t + score(x_t,jnp.full(n_samples,t))*sde.diffusion_coeff(t)**2 )*dt + sde.diffusion_coeff(t)*jnp.sqrt(dt)*jax.random.normal(step_rng,(n_samples,dim))
        return (x_t, rng), None
    
    ts = jnp.linspace(1,t0,n_steps)
    (x_t, rng), _ = jax.lax.scan(step, (x_0, rng), ts)

    return x_t

class EnergyResMLP(nn.Module):
    num_blocks: int
    width: int
    out_dim: int

    @nn.compact
    def __call__(self, x_in, t):
        # x: array shape (batch,dim), t: array shape (batch,1) 
        t = t.reshape((t.shape[0],1))
        st = jnp.sqrt(sde.variance(t))
        at = sde.mean(t)
        
        x = x_in 
        x = nn.Dense(self.width)(jnp.concatenate([x,st], axis=1))
        x = nn.activation.swish(x)
        
        for block in range(self.num_blocks):
            x_r = x
            x = nn.Dense(self.width)(jnp.concatenate([x,st], axis=1))
            x = nn.activation.swish(x)
            x = nn.Dense(self.width)(jnp.concatenate([x,st], axis=1))
            x = nn.activation.swish(x) 
            x += x_r
            
        x = nn.Dense(self.out_dim)(jnp.concatenate([x,st], axis=1))
        
        out_scale = st**2 + at**2
      
        return -jnp.linalg.norm(x - x_in, axis=1)**2 / 2*out_scale.flatten()


class ScoreFunction():
    def __init__(self, potential):
        self.potential = potential

    @functools.partial(jax.vmap, in_axes=(None, None, 0, 0))
    @functools.partial(jax.grad, argnums=2)    
    def __call__(self, params, x, t):
        # overall: input is (n,d)
        # apply receives (d,)
        # potential net requires (n,d) ie (1,d)
        return self.potential.apply(params, x.reshape(1, x.shape[0]), t.reshape(1, 1)).reshape(())
    
   
class EBModel(nn.Module):
    
    def __init__(self, depth, width, out_dim, lr, rng):   
            
        self.potential = EnergyResMLP(depth, width, out_dim) # depth is number of blocks. One block = 2 layers and a skip connection.
        self.score = ScoreFunction(self.potential)
        self.dim = out_dim        
        self.params = self.potential.init(rng, jnp.zeros((32, self.dim)), jnp.ones((32,1)))
        
        self.optimizer = optax.adam(lr) 
        
        self.opt_state = self.optimizer.init(self.params)        

    def train_DSM(self, samples, epochs, rng, batch_size=32):        
        
        n_batches = samples.shape[0] // batch_size
        dataset = tf.data.Dataset.from_tensor_slices(samples).shuffle(1024).batch(batch_size, drop_remainder=True)
        
        
        def loss(params, rng, batch):
            x0 = batch
            
            t_rng, z_rng = jax.random.split(rng)
            ts = jax.random.uniform(t_rng, (x0.shape[0],1), minval=1e-5, maxval=1-1e-5)
            z = jax.random.normal(z_rng, x0.shape)
            xt = sde.mean(ts)*x0 + jnp.sqrt(sde.variance(ts))*z
            
            wt = sde.diffusion_coeff(ts)**2
            return jnp.mean( wt*(self.score(params, xt, ts) + z/jnp.sqrt(sde.variance(ts)))**2 )
        
        @jax.jit
        def update_step(params, rng, batch, opt_state):
            val, grads = jax.value_and_grad(loss, has_aux=False)(params, rng, batch)
            updates, opt_state = self.optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return val, params, opt_state
        
        mean_losses = []
        for i in (pbar := trange(epochs)):

            total_loss = 0.            
            for batch in dataset.as_numpy_iterator():
                rng, step_rng = jax.random.split(rng)
                loss_eval, self.params, self.opt_state = update_step(self.params, step_rng, batch, self.opt_state)
                total_loss += loss_eval
                
            mean_loss = total_loss/n_batches
            mean_losses.append(mean_loss)
            pbar.set_postfix({"Loss":"{:.2f} ".format(mean_loss)})
                
        return mean_losses
        
    def train_TSM(self, samples, evals, epochs, rng, batch_size=32):
        
        n_batches = samples.shape[0] // batch_size
        dataset = tf.data.Dataset.from_tensor_slices((samples, evals)).shuffle(1024).batch(batch_size, drop_remainder=True)
        
        
        def loss(params, rng, batch):
            x0 = batch[0]
            gradlogpx = batch[1]
            
            t_rng, z_rng = jax.random.split(rng)
            ts = jax.random.uniform(t_rng, (x0.shape[0],1), minval=1e-5, maxval=1-1e-5)
            z = jax.random.normal(z_rng, x0.shape)
            st = jnp.sqrt(sde.variance(ts))
            at = sde.mean(ts)
            xt = at*x0 + st*z
            
            wt = at**2/st**2 * (st**2 + at**2)
            return jnp.mean( wt*(self.score(params, xt, ts) - gradlogpx/at )**2 )
        
        @jax.jit
        def update_step(params, rng, batch, opt_state):
            val, grads = jax.value_and_grad(loss, has_aux=False)(params, rng, batch)
            updates, opt_state = self.optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return val, params, opt_state
        
        mean_losses = []
        for i in (pbar := trange(epochs)):

            total_loss = 0.            
            for batch in dataset.as_numpy_iterator():
                rng, step_rng = jax.random.split(rng)
                loss_eval, self.params, self.opt_state = update_step(self.params, step_rng, batch, self.opt_state)
                total_loss += loss_eval
                
            mean_loss = total_loss/n_batches
            mean_losses.append(mean_loss)
            pbar.set_postfix({"Loss":"{:.2f} ".format(mean_loss)})
        
        
        return mean_losses
    
    def train_combined(self, samples, evals, epochs, rng, batch_size=32):
        
        n_batches = samples.shape[0] // batch_size
        dataset = tf.data.Dataset.from_tensor_slices((samples, evals)).shuffle(1024).batch(batch_size, drop_remainder=True)
        
        
        def loss(params, rng, batch):
            x0 = batch[0]
            gradlogpx = batch[1]
            
            t_rng, z_rng = jax.random.split(rng)
            ts = jax.random.uniform(t_rng, (x0.shape[0],1), minval=1e-5, maxval=1-1e-5)
            st = jnp.sqrt(sde.variance(ts))
            at = sde.mean(ts)
            z = jax.random.normal(z_rng, x0.shape)
            
            xt = at*x0 + st*z            

            wt = st**2 / (st**2 + at**2)
            
            phi = self.score(params, xt, ts)
            L_TSM = gradlogpx/at
            L_DSM = - z/st
        
            return jnp.mean( (phi - ((1-wt)*L_TSM + wt*L_DSM))**2 )
        
        @jax.jit
        def update_step(params, rng, batch, opt_state):
            val, grads = jax.value_and_grad(loss, has_aux=False)(params, rng, batch)
            updates, opt_state = self.optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return val, params, opt_state
        
        mean_losses = []
        for i in (pbar := trange(epochs)):

            total_loss = 0.            
            for batch in dataset.as_numpy_iterator():
                rng, step_rng = jax.random.split(rng)
                loss_eval, self.params, self.opt_state = update_step(self.params, step_rng, batch, self.opt_state)
                total_loss += loss_eval
                
            mean_loss = total_loss/n_batches
            mean_losses.append(mean_loss)
            pbar.set_postfix({"Loss":"{:.2f} ".format(mean_loss)})
        
        
        return mean_losses    


    def __call__(self, x, t):
        return self.model(x,t)     
            
    def sample(self,sample_rng, N):
        return reverse_sde_solver(sample_rng, (N,self.dim), jax.jit(lambda x,t: self.score(self.params, x, t)))



def annealed_mcmc_sampling(rng, potential, shape, inner_steps, outer_steps=None, dt=.1, t0=0.001, x_0 = None, leapfrog_steps=3, inv_mass_matrix=None):
    rng, init_rng = jax.random.split(rng, 2)
    n_samples = shape[0]
    dim = shape[1]
    if x_0 is None:
        x_0 = jax.random.normal(init_rng,shape)*jnp.sqrt(sde.variance(1.))
    if inv_mass_matrix is None:
        inv_mass_matrix = jnp.ones((dim))
        
    def inner_loop(carry, t):
        x, rng = carry
        rng, step_rng = jax.random.split(rng,2)
        keys = jax.random.split(step_rng, inner_steps)
        
        def p_t(x):
            return potential(x,jnp.full(1, t))[0]
        
        
        step_size = dt
        #(nuts updates take variable length of time/number of leapfrog steps so vmap is slow)
        sampler = blackjax.hmc(p_t, step_size, inv_mass_matrix, leapfrog_steps)
        initial_state = jax.vmap(sampler.init, in_axes=(0))(x.reshape((n_samples,1,dim))) 

        @jax.jit
        def inner_step(states, rng_key):
            keys = jax.random.split(rng_key, n_samples)
            states, info = jax.vmap(sampler.step)(keys, states)
            return states, info.acceptance_rate

        state, info = jax.lax.scan(inner_step, initial_state, keys)         
        x = state.position
        
        return (x.reshape(shape), rng), info    
    

    @jax.jit
    def outer_step(carry, t):
        x, rng = carry

        rng, step_rng = jax.random.split(rng,2)     
        (x, rng), info = inner_loop((x, rng), t)
        
        return (x, rng), info   
            
    if outer_steps:
        ts = jnp.linspace(1, t0, outer_steps).reshape(-1, 1)
        (samples, rng), info = jax.lax.scan(outer_step, (x_0, rng), ts)
        return samples, info
    else:
        (samples, rng), info = inner_loop((x_0, rng), jnp.array([t0]))
        return samples, info
    
