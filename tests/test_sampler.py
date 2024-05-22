import jax
from jax import numpy as jnp

import thermox


def test_sample_array_input():
    key = jax.random.PRNGKey(0)
    dim = 2
    dt = 0.1
    ts = jnp.arange(0, 10_000, dt)

    A = jnp.array([[3, 2], [2, 4.0]])
    b, x0 = jnp.zeros(dim), jnp.zeros(dim)
    D = 2 * jnp.eye(dim)

    samples = thermox.sample(key, ts, x0, A, b, D)

    samp_cov = jnp.cov(samples.T)
    samp_mean = jnp.mean(samples.T, axis=1)
    assert jnp.allclose(A @ samp_cov, jnp.eye(2), atol=1e-1)
    assert jnp.allclose(samp_mean, b, atol=1e-1)
