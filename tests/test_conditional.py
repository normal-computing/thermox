import jax
from jax import numpy as jnp

import thermox


def van_loan_covariance(A, D, t):
    """int_0^t exp(-A s) D exp(-A^T s) ds via Van Loan (1978)."""
    d = A.shape[0]
    M = jnp.block([[-A, D], [jnp.zeros((d, d)), A.T]]) * t
    F = jax.scipy.linalg.expm(M)
    return F[:d, d:] @ jax.scipy.linalg.expm(-A.T * t)


def test_mean_and_cov():
    jax.config.update("jax_enable_x64", True)
    dim = 2
    t = 1.0

    A = jnp.array([[3, 2.5], [2, 4.0]])  # not symmetric, not normal
    b = jax.random.normal(jax.random.PRNGKey(1), (dim,))
    x0 = jax.random.normal(jax.random.PRNGKey(2), (dim,))
    D = 2 * jnp.eye(dim)

    # References independent of thermox
    mean_ref = b + jax.scipy.linalg.expm(-A * t) @ (x0 - b)
    cov_ref = van_loan_covariance(A, D, t)

    mean = thermox.conditional.mean(t, x0, A, b, D)
    assert mean.shape == (dim,)
    assert jnp.allclose(mean, mean_ref, atol=1e-10)

    cov = thermox.conditional.covariance(t, A, D)
    assert cov.shape == (dim, dim)
    assert jnp.allclose(cov, cov_ref, atol=1e-10)

    samples = jax.vmap(
        lambda k: thermox.sample(k, jnp.array([0.0, t]), x0, A, b, D)[-1]
    )(jax.random.split(jax.random.PRNGKey(0), 1000000))
    assert jnp.allclose(mean_ref, jnp.mean(samples, axis=0), atol=1e-2)
    assert jnp.allclose(cov_ref, jnp.cov(samples.T), atol=1e-3)

    mean_and_cov = thermox.conditional.mean_and_covariance(t, x0, A, b, D)
    assert mean_and_cov[0].shape == (dim,)
    assert mean_and_cov[1].shape == (dim, dim)
    assert jnp.allclose(mean_and_cov[0], mean, atol=1e-5)
    assert jnp.allclose(mean_and_cov[1], cov, atol=1e-5)
