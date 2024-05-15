import jax
import jax.numpy as jnp
from thermox.sampler import sample, sample_identity_diffusion
from jax.lax import fori_loop
from jax import Array


def solve(
    A,
    b,
    num_samples: int = 10000,
    dt: float = 1.0,
    burnin: int = 0,
    key: Array = None,
) -> Array:
    """
    Obtain the solution of the linear system

    Ax = b

    by collecting samples from an Ornstein-Uhlenbeck
    process and calculating the mean over the samples.

    Args:
        A: Linear system matrix.
        b: Linear system vector.
        num_samples: Number of samples to be collected.
        dt: Time step.
        burnin: Time-step index corresponding to the end of the burn-in period.
            Samples before this step are not collected.
        key: JAX random key

    Returns:
        Approximate solution, x, of the linear system.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    ts = jnp.arange(burnin, burnin + num_samples) * dt
    x0 = jnp.zeros_like(b)
    samples = sample_identity_diffusion(key, ts, x0, A, jnp.linalg.solve(A, b))
    return jnp.mean(samples, axis=0)


def inv(
    A,
    num_samples: int = 10000,
    dt: float = 1.0,
    burnin: int = 0,
    key: Array = None,
) -> Array:
    """
    Obtain the inverse of a matrix A by
    collecting samples from an Ornstein-Uhlenbeck
    process and calculating the covariance of the samples.

    Args:
        A: Matrix to invert (must be symmetric positive definite).
        num_samples: Number of samples to be collected.
        dt: Time step.
        burnin: Time-step index corresponding to the end of the burn-in period.
            Samples before this step are not collected.
        key: JAX random key

    Returns:
        Approximate inverse of A.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    ts = jnp.arange(burnin, burnin + num_samples) * dt
    b = jnp.zeros(A.shape[0])
    x0 = jnp.zeros_like(b)
    samples = sample(key, ts, x0, A, b, 2 * jnp.eye(A.shape[0]))
    return jnp.cov(samples.T)


def expnegm(
    A,
    num_samples: int = 10000,
    dt: float = 1.0,
    burnin: int = 0,
    key: Array = None,
    alpha: float = 0.0,
) -> Array:
    """
    Obtain the negative exponential of a matrix A by
    collecting samples from an Ornstein-Uhlenbeck
    process and calculating the covariance of the samples.

    Args:
        A: Matrix to exponentiate.
        num_samples: Number of samples to be collected.
        dt: Time step.
        burnin: Time-step index corresponding to the end of the burn-in period.
            Samples before this step are not collected.
        key: JAX random key
        alpha: Regularization parameter to ensure diffusion matrix
            is symmetric positive definite.

    Returns:
        Approximate negative matrix exponential, exp(-A).
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    A_shifted = (A + alpha * jnp.eye(A.shape[0])) / dt
    B = A_shifted + A_shifted.T

    ts = jnp.arange(burnin, burnin + num_samples) * dt
    b = jnp.zeros(A.shape[0])
    x0 = jnp.zeros_like(b)
    samples = sample(key, ts, x0, A_shifted, b, B)
    return autocovariance(samples) * jnp.exp(alpha)


def expm(
    A,
    num_samples: int = 10000,
    dt: float = 1.0,
    burnin: int = 0,
    key: Array = None,
    alpha: float = 1.0,
) -> Array:
    """
    Obtain the exponential of a matrix A by
    collecting samples from an Ornstein-Uhlenbeck
    process and calculating the covariance of the samples.

    Args:
        A: Matrix to exponentiate.
        num_samples: Number of samples to be collected.
        dt: Time step.
        burnin: Time-step index corresponding to the end of the burn-in period.
            Samples before this step are not collected.
        key: JAX random key
        alpha: Regularization parameter to ensure diffusion matrix
            is symmetric positive definite.

    Returns:
        Approximate matrix exponential, exp(A).
    """
    return expnegm(-A, num_samples, dt, burnin, key, alpha)


def autocovariance(samples: Array) -> Array:
    """
    Calculate the autocovariance of a set of samples.

    Args:
        samples: Samples from a stochastic process, as an array of shape (num_samples, dimension).

    Returns:
        Autocovariance of the samples.
    """
    return fori_loop(
        0,
        len(samples) - 1,
        lambda i, x: x + jnp.outer(samples[i + 1], samples[i]) / (len(samples) - 1),
        jnp.zeros((samples.shape[1], samples.shape[1])),
    )
