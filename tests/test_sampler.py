import jax
from jax import numpy as jnp

import thermox


def test_sample_array_input():
    jax.config.update("jax_enable_x64", True)
    key = jax.random.PRNGKey(0)
    dim = 2
    dt = 0.1
    ts = jnp.arange(0, 10_000, dt)

    # Add some noise to the time points to make the timesteps different
    ts += jax.random.uniform(key, (ts.shape[0],)) * dt
    ts = ts.sort()

    A = jnp.array([[3, 2.5], [2, 4.0]])
    b = jax.random.normal(jax.random.PRNGKey(1), (dim,))
    x0 = jax.random.normal(jax.random.PRNGKey(2), (dim,))
    D = 2 * jnp.eye(dim)

    samples = thermox.sample(key, ts, x0, A, b, D, associative_scan=False)

    samp_cov = jnp.cov(samples.T)
    samp_mean = jnp.mean(samples.T, axis=1)
    assert jnp.allclose(A @ samp_cov, jnp.eye(2), atol=1e-1)
    assert jnp.allclose(samp_mean, b, atol=1e-1)

    samples_as = thermox.sample(key, ts, x0, A, b, D, associative_scan=True)
    assert jnp.allclose(samples, samples_as, atol=1e-6)
