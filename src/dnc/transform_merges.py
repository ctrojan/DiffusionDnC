import jax.numpy as jnp

# ================= consensus monte carlo =================

def consensus(shard_samples):
    M = len(shard_samples)

    shard_vars = [jnp.cov(shard.transpose()) for shard in shard_samples]
    shard_vars_inv = [jnp.linalg.inv(V_b) for V_b in shard_vars]
    
    cmc_samps = (jnp.linalg.inv(sum(shard_vars_inv))@sum([shard_vars_inv[b]@shard_samples[b].transpose() for b in range(M)])).transpose()
    
    return cmc_samps 


# ================= swiss =================
def matsqrt(A):
    # Square root by diagonalization
    evalues, evectors = jnp.linalg.eigh(A)
    assert (evalues >= 0).all() # Check positive semi-definite
    return evectors * jnp.sqrt(evalues) @ jnp.transpose(evectors)

def swiss(shard_samples, matsqrt=matsqrt):
    M = len(shard_samples)
    J = shard_samples[0].shape[0]
    dim = shard_samples[0].shape[1]
    
    shard_means = [jnp.mean(shard, axis=0) for shard in shard_samples]
    shard_vars = [jnp.cov(shard, rowvar=False) for shard in shard_samples]
    shard_vars_inv = [jnp.linalg.inv(V_b) for V_b in shard_vars]
    
    V = jnp.linalg.inv(1/M*sum(shard_vars_inv))
    mu = V@( 1/M*sum([shard_vars_inv[b]@shard_means[b] for b in range(M)]) )
    
    scale_M = matsqrt(V)
    scale_M_inv = jnp.linalg.inv(scale_M)
    
    swiss_samps = jnp.zeros((J*M,dim))
    for b in range(M):
        V_b = scale_M_inv@shard_vars[b]@scale_M_inv
        M_b = matsqrt(V_b)
        A_b = scale_M@jnp.linalg.inv(M_b)@scale_M_inv
        swiss_samps = swiss_samps.at[(b*J):((b+1)*J),:].set((A_b@(shard_samples[b].transpose() - shard_means[b][:,None]) + mu[:,None]).transpose())
    
    return swiss_samps


def swissprod(shard_samples):
    M = len(shard_samples)
    J = shard_samples[0].shape[0]
    dim = shard_samples[0].shape[1]
    
    shard_means = [jnp.mean(shard, axis=0) for shard in shard_samples]
    shard_vars = [jnp.cov(shard, rowvar=False) for shard in shard_samples]
    shard_vars_inv = [jnp.linalg.inv(V_b) for V_b in shard_vars]
    
    V = jnp.linalg.inv(sum(shard_vars_inv))
    mu = V@( sum([shard_vars_inv[b]@shard_means[b] for b in range(M)]) )
    
    scale_M = matsqrt(V)
    scale_M_inv = jnp.linalg.inv(scale_M)
    
    swiss_samps = jnp.zeros((J*M,dim))
    for b in range(M):
        V_b = scale_M_inv@shard_vars[b]@scale_M_inv
        M_b = matsqrt(V_b)
        A_b = scale_M@jnp.linalg.inv(M_b)@scale_M_inv
        swiss_samps = swiss_samps.at[(b*J):((b+1)*J),:].set((A_b@(shard_samples[b].transpose() - shard_means[b][:,None]) + mu[:,None]).transpose())
    
    return swiss_samps
 
