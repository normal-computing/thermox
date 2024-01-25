import jax
import jax.numpy as jnp
from thermox.sampler import collect_samples, collect_samples_full_diffusion_matrix
from jax.lax import fori_loop

def solve(
    A,
    b,
    num_samples: int=10000,
    dt: float=1., 
    burnin: int=0,
    seed: int=0,
) -> jax.Array:
    """
    Obtain the solution of the linear system

    Ax = b

    by collecting samples from an Ornstein-Uhlenbeck
    process and calculating the mean over the samples.

    Args:
        - A: drift matrix.
        - b: mean displacement vector.
        - num_samples: float, number of samples to be collected.
        - dt: float, time step.
        - burnin: burn-in, time before which samples are not collected.
        - seed: random seed
    
    Returns:
        - x: approximate solution of the linear system.
    """
    key = jax.random.PRNGKey(seed)
    samples = collect_samples(key, A, b, num_samples=num_samples, dt=dt, burnin=burnin)
    return jnp.mean(samples, axis=0)

def inv(
    A,
    num_samples: int=10000,
    dt: float=1., 
    burnin: int=0,
    seed: int=0,
) -> jax.Array:
    """
    Obtain the inverse of a matrix A by 
    collecting samples from an Ornstein-Uhlenbeck
    process and calculating the covariance of the samples.

    Args:
        - A: drift matrix.
        - b: mean displacement vector.
        - num_samples: float, number of samples to be collected.
        - dt: float, time step.
        - burnin: burn-in, time before which samples are not collected.
        - seed: random seed
    
    Returns:
        - inv: approximate inverse of A.
    """
    key = jax.random.PRNGKey(seed)
    b = jnp.zeros(A.shape[0])
    samples = collect_samples(key, A, b, num_samples=num_samples, dt=dt, burnin=burnin)
    return jnp.cov(samples.T)

def negexpm(
    A,
    num_samples: int=10000,
    dt: float=1., 
    burnin: int=0,
    seed: int=0,
    alpha: float=0.,
    rtol: float=1e-3,
) -> jax.Array:
    """
    Obtain the exponential of a matrix A by 
    collecting samples from an Ornstein-Uhlenbeck
    process and calculating the covariance of the samples.

    Args:
        - A: drift matrix.
        - b: mean displacement vector.
        - num_samples: float, number of samples to be collected.
        - dt: float, time step.
        - burnin: burn-in, time before which samples are not collected.
        - seed: random seed
    
    Returns:
        - inv: approximate inverse of A.
    """
    A = (A + alpha * jnp.eye(A.shape[0])) / dt
    B = 0.5 * (A + A.T)

    key = jax.random.PRNGKey(seed)
    b, x0 = jnp.zeros(A.shape[0]), jnp.zeros(A.shape[0])
    # samples = collect_samples_from_device_ODL_full_diffusion_matrix(key, x0, A, b, D=B, num_samples=num_samples, dt=dt, burnin=burnin)
    samples = collect_samples_full_diffusion_matrix(key, A, b, D, num_samples=num_samples, dt=dt, burnin=burnin)
    return autocovariance(samples) * jnp.exp(-alpha)

def expm(
    A,
    num_samples: int=10000,
    dt: float=1., 
    burnin: int=0,
    seed: int=0,
    alpha: float=1000,
    rtol: float=1e-3,
) -> jax.Array:
    """
    Obtain the exponential of a matrix A by 
    collecting samples from an Ornstein-Uhlenbeck
    process and calculating the covariance of the samples.

    Args:
        - A: drift matrix.
        - b: mean displacement vector.
        - num_samples: float, number of samples to be collected.
        - dt: float, time step.
        - burnin: burn-in, time before which samples are not collected.
        - seed: random seed
    
    Returns:
        - inv: approximate inverse of A.
    """

    A_shifted = (-A + alpha * jnp.eye(A.shape[0])) / dt
    B = 0.5 * (A_shifted + A_shifted.T)
    key = jax.random.PRNGKey(seed)
    b = jnp.zeros(A.shape[0])
    samples = collect_samples(key, A, b, D=B, num_samples=num_samples, dt=dt, burnin=burnin)
    return jnp.cov(samples.T) * jnp.exp(alpha)

def autocovariance(samples):
    return fori_loop(
        0,
        len(samples) - 1,
        lambda i, x: x + jnp.outer(samples[i + 1], samples[i]) / (len(samples) - 1),
        jnp.zeros((samples.shape[1], samples.shape[1])),
    )
