import jax
from jax import numpy as jnp

import thermox


def test_mean_and_cov():
    jax.config.update("jax_enable_x64", True)
    dim = 2
    t = 1.0

    A = jnp.array([[3, 2.5], [2, 4.0]])
    b = jax.random.normal(jax.random.PRNGKey(1), (dim,))
    x0 = jax.random.normal(jax.random.PRNGKey(2), (dim,))
    D = 2 * jnp.eye(dim)

    mean = thermox.conditional.mean(t, x0, A, b, D)
    samples = jax.vmap(
        lambda k: thermox.sample(k, jnp.array([0.0, t]), x0, A, b, D)[-1]
    )(jax.random.split(jax.random.PRNGKey(0), 1000000))
    assert mean.shape == (dim,)
    assert jnp.allclose(mean, jnp.mean(samples, axis=0), atol=1e-2)

    cov = thermox.conditional.covariance(t, A, D)
    assert cov.shape == (dim, dim)
    assert jnp.allclose(cov, jnp.cov(samples.T), atol=1e-3)

    mean_and_cov = thermox.conditional.mean_and_covariance(t, x0, A, b, D)
    assert mean_and_cov[0].shape == (dim,)
    assert mean_and_cov[1].shape == (dim, dim)
    assert jnp.allclose(mean_and_cov[0], mean, atol=1e-5)
    assert jnp.allclose(mean_and_cov[1], cov, atol=1e-5)
