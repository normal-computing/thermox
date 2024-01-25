import jax
from jax import numpy as jnp
from jax.lax import fori_loop

from thermox.sampler import collect_samples_from_device_ODL_full_diffusion_matrix

A = jnp.array([[3, 2], [3, 4.0]])
def test_thermo_matexp(A):
    

    B = 0.5 * (A + A.T)

    dt = 0.1

    samps = collect_samples_from_device_ODL_full_diffusion_matrix(
        jax.random.PRNGKey(42), jnp.array([0, 0.0]), A, 0, B, 10000, dt
    )

    assert jnp.allclose(samps.mean(axis=0), jnp.array([0, 0]), atol=1e-1)
    assert jnp.allclose(jnp.cov(samps.T), jnp.eye(2), atol=1e-1)

    emp_mat_exp = fori_loop(
        0,
        len(samps) - 1,
        lambda i, x: x + jnp.outer(samps[i + 1], samps[i]) / (len(samps) - 1),
        jnp.zeros((2, 2)),
    )

    print(jnp.linalg.norm(emp_mat_exp - jax.scipy.linalg.expm(-A * dt)))
    assert jnp.allclose(emp_mat_exp, jax.scipy.linalg.expm(-A * dt), atol=1e-1)