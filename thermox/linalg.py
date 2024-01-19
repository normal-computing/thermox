import jax
import jax.numpy as jnp
from thermox.sampler import collect_samples

def solve(
    A,
    b,
    num_samples: int=10000,
    dt: float=0.1, 
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
    dt: float=0.1, 
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

def expm(
    A,
    num_samples: int=10000,
    dt: float=0.1, 
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
    samples = collect_samples(key, A, D=A, num_samples=num_samples, dt=dt, burnin=burnin)
    return jnp.cov(samples)