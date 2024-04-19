import jax
from jax import numpy as jnp

import thermox


def test_linear_system():
    A = jnp.array([[3, 2], [2, 4.0]])
    b = jnp.array([1, 2.0])

    x = thermox.linalg.solve(A, b, num_samples=10000, dt=0.1, burnin=0)

    assert jnp.allclose(A @ x, b, atol=1e-1)


def test_inv():
    A = jnp.array([[3, 2], [2, 4.0]])

    A_inv = thermox.linalg.inv(A, num_samples=10000, dt=0.1, burnin=0)

    assert jnp.allclose(A @ A_inv, jnp.eye(2), atol=1e-1)


def test_expnegm():
    A = jnp.array([[3, 2], [2, 4.0]])

    expnegA = thermox.linalg.expnegm(A, num_samples=10000, dt=0.1, burnin=0)

    assert jnp.allclose(expnegA, jax.scipy.linalg.expm(-A), atol=1e-1)


def test_expm():
    A = jnp.array([[-0.4, 0.1], [0.5, -0.3]])

    expA = thermox.linalg.expm(A, num_samples=100000, dt=0.1, burnin=0, alpha=1.0)

    assert jnp.allclose(expA, jax.scipy.linalg.expm(A), atol=1e-1)
