import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt

def kde_contour(samples, plotlim_x, plotlim_y=None, res=32, cmap='magma', color=None, lty='solid', fig=plt):
    b_x = jnp.linspace(plotlim_x[0],plotlim_x[1],res)
    plotlim_y = plotlim_x if plotlim_y is None else plotlim_y
    b_y = jnp.linspace(plotlim_y[0],plotlim_y[1],res)

    b_0, b_1 = jnp.meshgrid(b_x, b_y)
    grid = jnp.stack([b_0.flatten(), b_1.flatten()])
    kernel = jax.scipy.stats.gaussian_kde(samples[:,:2].transpose())
    z = kernel(grid).reshape(b_0.shape)
    if color is None:
        fig.contour(b_0,b_1,z, cmap=cmap, linestyles=lty)
    else:
        fig.contour(b_0,b_1,z, colors=color, linestyles=lty)
        
    
def kde_contour_comparison(samples, samples_exact, plotlim_x, plotlim_y=None, res=64, approx_col=[(0.20, 0.69 , 0.84 )], exact_col=[(0,0,0)], fig=plt):
    b_x = jnp.linspace(plotlim_x[0],plotlim_x[1],res)
    plotlim_y = plotlim_x if plotlim_y is None else plotlim_y
    b_y = jnp.linspace(plotlim_y[0],plotlim_y[1],res)

    b_0, b_1 = jnp.meshgrid(b_x, b_y)
    grid = jnp.stack([b_0.flatten(), b_1.flatten()])
    
    kernel = jax.scipy.stats.gaussian_kde(samples_exact[:,:2].transpose())
    z = kernel(grid).reshape(b_0.shape)    
    fig.contour(b_0,b_1,z, colors=exact_col)
    
    kernel = jax.scipy.stats.gaussian_kde(samples[:,:2].transpose())
    z = kernel(grid).reshape(b_0.shape) 
    fig.contour(b_0,b_1,z,colors=approx_col)
    
    if fig!=plt:
        fig.set_xticks([])
        fig.set_yticks([])
    
    
def kde_1d(samples, plotlim, res=32, color='black'):
    b = jnp.linspace(plotlim[0],plotlim[1],res)
    kernel = jax.scipy.stats.gaussian_kde(samples.transpose())
    z = kernel(b)
    plt.plot(b,z, color=color)
    plt.xlim(plotlim[0], plotlim[1])    
    
def energy_contour(energy_fn, plotlim_x, plotlim_y=None, res=32, cmap='magma', lty='solid'):
    b_x = jnp.linspace(plotlim_x[0],plotlim_x[1],res)
    plotlim_y = plotlim_x if plotlim_y is None else plotlim_y
    b_y = jnp.linspace(plotlim_y[0],plotlim_y[1],res)
    b_0, b_1 = jnp.meshgrid(b_x, b_y)
    grid = jnp.stack([b_0.flatten(), b_1.flatten()])
    
    z = energy_fn(grid.transpose()).reshape(b_0.shape)
    z -= jnp.mean(z)
    z = jnp.exp(z)
    
    plt.contour(b_0,b_1,z, cmap=cmap, linestyles=lty)
    plt.xlim(plotlim_x[0], plotlim_x[1])
    plt.ylim(plotlim_y[0], plotlim_y[1])
    
    
def energy_contour_t(energy_fn, t, plotlim_x, shift=None, scale=None, plotlim_y=None, res=32, cmap='viridis', lty='dashed'):
    b_x = jnp.linspace(plotlim_x[0],plotlim_x[1],res)
    plotlim_y = plotlim_x if plotlim_y is None else plotlim_y
    b_y = jnp.linspace(plotlim_y[0],plotlim_y[1],res)
    
    b_0, b_1 = jnp.meshgrid(b_x, b_y)
    grid = jnp.stack([b_0.flatten(), b_1.flatten()])
    
    if shift is not None and scale is not None:
        z = jnp.exp(energy_fn((grid.transpose()-shift)@scale,jnp.full(grid.shape[1],t)).reshape(b_0.shape))
    else:
        z = jnp.exp(energy_fn(grid.transpose(),jnp.full(grid.shape[1],t)).reshape(b_0.shape))

    plt.contour(b_0,b_1,z, cmap=cmap, linestyles=lty)
    plt.xlim(plotlim_x[0], plotlim_x[1])
    plt.ylim(plotlim_y[0], plotlim_y[1])

def fitted_energy_subposteriors(shard_samples, score_models, t0, plotlim, shift, scale):  
    M = len(shard_samples)
    fig = plt.figure()
    for i in range(M):
        plt.subplot(int(jnp.ceil(M/4)),4,i+1)
        kde_contour(shard_samples[i], plotlim)
        energy_contour_t(lambda x,t: score_models[i].potential.apply(score_models[i].params, x, t), t0, plotlim, shift[i], scale[i], cmap='viridis', lty='dashed')
        plt.axis('off')
    
    