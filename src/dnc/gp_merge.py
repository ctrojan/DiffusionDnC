
import jax
import functools
import jax.numpy as jnp
import optax

import gpjax
import dataclasses
import jaxtyping
import cola
from cola.ops import (
    Identity,
    LinearOperator,
)


class ConjugateMMLL(gpjax.objectives.AbstractObjective):
    def step(
        self,
        posterior: "gpjax.gps.ConjugatePosterior",  # noqa: F821
        train_data: gpjax.Dataset,  # noqa: F821
    ) -> gpjax.typing.ScalarFloat:
        r"""marginal y-log-likelihood of the Gaussian process, also marginalising out some mean fn params

            Prior's mean function should have a h(x) method applying the basis functions in the mean 
            whose coefficients have been analytically marginalised out.

        """
        x, y = train_data.X, train_data.y

        # Observation noise o²
        obs_noise = posterior.likelihood.obs_stddev**2
        mx = posterior.prior.mean_function(x)

        # Σ = (Kxx + Io²) = LLᵀ
        Kxx = posterior.prior.kernel.gram(x)
        Kxx += cola.ops.I_like(Kxx) * posterior.prior.jitter
        
        
        Ky = Kxx + cola.ops.I_like(Kxx) * obs_noise     
        Ky = cola.PSD(Ky)
        Ky_inv = cola.inverse(Ky)
        
        H = cola.ops.Dense(posterior.prior.mean_function.h(x))
        A = H @ Ky_inv @ cola.ops.Transpose(H)
        C = Ky_inv @ cola.ops.Transpose(H) @ cola.inverse(A) @ H @ Ky_inv
        
        log_prob = 0.5*(jnp.transpose(jnp.atleast_1d((y - mx).squeeze())) @ ( C - Ky_inv ) @ jnp.atleast_1d((y - mx).squeeze())
                        - cola.linalg.logdet(Ky) - cola.linalg.logdet(A) 
                                                                             
                        )


        return self.constant * log_prob
    
    
def predictive_dist_corrected(predict_pts, posterior, train_data):  
    r"""
    Specifies predictive distribution of the GP, with some mean function coefficients
    analytically marginalised.
    """
     
    # Σ = (Kxx + Io²) = LLᵀ
    K = posterior.prior.kernel.gram(train_data.X)
    Kxx = K + cola.ops.I_like(K) * posterior.prior.jitter 
    mx = posterior.prior.mean_function(train_data.X)
    
    # Observation noise o²
    obs_noise = posterior.likelihood.obs_stddev**2
    
    Ky = Kxx + cola.ops.I_like(Kxx) * obs_noise     
    Ky = cola.PSD(Ky)
    Ky_inv = cola.inverse(Ky)
    
    H = cola.ops.Dense(posterior.prior.mean_function.h(train_data.X))
    Hstar = cola.ops.Dense(posterior.prior.mean_function.h(predict_pts))
    
    Kstar = cola.ops.Dense(posterior.prior.kernel.cross_covariance(train_data.X, predict_pts))
    
    beta_bar = cola.inverse(H@Ky_inv@cola.ops.Transpose(H)) @ H@Ky_inv@(train_data.y - mx)
    R = Hstar - H@Ky_inv@Kstar
    
    predict_means = cola.ops.Transpose(Kstar) @ Ky_inv @ (train_data.y - mx)
    # correction for prior mean and beta
    predict_means +=  posterior.prior.mean_function(predict_pts) + cola.ops.Transpose(R)@beta_bar
    
    predict_cov = posterior.prior.kernel.gram(predict_pts) - cola.ops.Transpose(Kstar) @ Ky_inv @ Kstar
    # correction for beta
    predict_cov += cola.ops.Transpose(R) @ cola.inverse(H@Ky_inv@cola.ops.Transpose(H)) @ R
    
    return (predict_means, predict_cov.to_dense()) #gpjax.distributions.GaussianDistribution


class logdensity_GP():
    def __init__(self, x, y):
        n = x.shape[0]
        self.dim = x.shape[1]
        V_inv = jnp.linalg.inv(jnp.cov(x, rowvar=False).reshape(self.dim,self.dim))
        self.train_data = gpjax.Dataset(X=x, y=y)
        
        @dataclasses.dataclass
        class Quadratic(gpjax.mean_functions.AbstractMeanFunction):
            r"""
            
            """
            V_inv: jaxtyping.Float[gpjax.typing.Array, "1"] = gpjax.base.static_field(jnp.identity(self.dim))
            beta_2: jaxtyping.Float[gpjax.typing.Array, "1"] = gpjax.base.param_field(jnp.ones((1,1)))
            
            @functools.partial(jax.vmap, in_axes=(None, 0))
            def mx(self,x):
                return x@self.V_inv@x.transpose()*-jnp.abs(self.beta_2)
            
            def h(self,x):
                # basis functions whose coefficients are being analytically marginalised
                # beta_0 + x@beta_1 -> (1, x)@beta
                return (jnp.hstack((jnp.ones((x.shape[0],1)), x))).transpose()

            def __call__(self, x: jaxtyping.Num[gpjax.typing.Array, "N D"]) -> jaxtyping.Float[gpjax.typing.Array, "N 1"]:
                return self.mx(x).reshape((x.shape[0],1))
        
        meanf = Quadratic(V_inv=V_inv)
        kernel = gpjax.kernels.PoweredExponential()
        prior = gpjax.gps.Prior(mean_function=meanf, kernel=kernel)
        likelihood = gpjax.likelihoods.Gaussian(num_datapoints = n)
        self.posterior = prior * likelihood
        
        
    
    def fit(self, rng, opt_lr=1e-3):
        objective=jax.jit(ConjugateMMLL(negative=True))
        optimiser = optax.adam(learning_rate=opt_lr)
        # Obtain Type 2 MLEs of the hyperparameters
        self.opt_posterior, history = gpjax.fit(
            model=self.posterior,
            objective=objective,
            train_data=self.train_data,
            optim=optimiser,
            num_iters=200,
            safe=True,
            key=rng,
        )
        
    @functools.partial(jax.jit, static_argnums=[0])
    def __call__(self, eval_points):
        return predictive_dist_corrected(eval_points, self.opt_posterior, self.train_data)
    
    # @functools.partial(jax.vmap, in_axes=(None,0))
    def full_post_component(self, theta):
        mu_c, sigma_c = self(theta.reshape((1,self.dim)))
        return (mu_c + 1/2*sigma_c).reshape(())
    

