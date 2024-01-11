import jax
import jax.numpy as jnp
from jax.lax import fori_loop, scan

def collect_samples(
    key: jax.random.KeyArray, 
    x0: jax.Array,
    A: jax.Array,
    b: jax.Array, 
    num_samples: int,
    D: jax.Array=None, 
    dt: float=0.1, 
    gamma: float=1., 
    k0: float=1., 
    burnin: int=0,
) -> jax.Array:
    """
    Collects samples from the overdamped Langevin (ODL)
    process, defined as:
    
    dx = - A/γ * (x - b) dt + sqrt(2 * κ_0 * D/γ) dW
    
    by using exact diagonalization. 

    Args:
        - key: jax PRNGKey.
        - x0: initial state of the Langevin process.
        - A: drift matrix.
        - b: mean displacement vector.
        - num_samples: float, number of samples to be collected.
        - dt: float, time step.
        - gamma: float, damping constant.
        - k0: float, noise variance.
        - burnin: burn-in, time before which samples are not collected.

    Returns:
        - samples: array-like, desired samples.
    """

    if D is not None:
        D_sqrt = jnp.linalg.cholesky(2 * D)
    else:
        D_sqrt = jnp.eye(A.shape[0])

    drift = A / gamma
    diffusion_scalar = jnp.sqrt(k0 / gamma)

    expm = jax.scipy.linalg.expm(-drift * dt)
    exp2m = expm @ expm
    transition_mean = lambda x: b + expm @ (x - b)
    transition_cov = (
        diffusion_scalar * jnp.linalg.inv(drift) @ (jnp.eye(A.shape[0]) - exp2m)
    )
    transition_cov_sqrt = jnp.linalg.cholesky(transition_cov)

    def next_x(x, tkey):
        return transition_mean(x) + transition_cov_sqrt @ jax.random.normal(
            tkey, shape=x.shape
        )

    def scan_body(x_and_key, i):
        x, rk = x_and_key
        rk, rk_use = jax.random.split(rk)
        x = next_x(x, rk_use)
        return (x, rk), x

    xs = scan(scan_body, (x0, key), jnp.arange(num_samples + burnin))[1]

    return xs[burnin:]

def collect_samples_diffusion_matrix(
    key, x0, A, b, D, num_samples, dt=0.1, burnin=0, solver_steps_per_dt=10
):
    """
    Collects samples from an overdamped Langevin (ODL)
    process via an Euler discretisation. The times are
    in units of the time constant RC.

    dx = - A(x - b) dt + sqrt(2 * D) dW

    Args:
        - key: jax PRNGKey.
        - x0: vector, initial state of the Langevin process.
        - A: array-like, matrix to be inverted or on the right hand side
        of the linear system to be solved.
        - b: vector, on the left hand side of the linear system,
        corresponding to a DC voltage bias.
        - D: diffusion matrix
        - num_samples: float, number of samples to be collected.
        - dt: float, time step.
        - burnin: int, time before which samples are not collected.

    Returns:
        - samples: array-like, desired samples.
    """

    if D is not None:
        D_sqrt = jnp.linalg.cholesky(2 * D)
    else:
        D_sqrt = jnp.eye(A.shape[0])

    def next_x(x_in, t_diff, tkey, steps_per_dt):
        ts = jnp.linspace(0, t_diff, steps_per_dt + 1)
        keys = jax.random.split(tkey, steps_per_dt)

        def body_fun(i, x):
            stepsize = ts[i + 1] - ts[i]
            return (
                x
                - stepsize * (A @ (x - b))
                + jnp.sqrt(stepsize)
                * D_sqrt
                @ jax.random.normal(keys[i], shape=x.shape)
            )

        return fori_loop(0, steps_per_dt, body_fun, x_in)

    def scan_body(x_and_key, i):
        x, rk = x_and_key
        rk, rk_use = jax.random.split(rk)
        x = next_x(x, dt, rk_use, solver_steps_per_dt)
        return (x, rk), x

    xs = scan(scan_body, (x0, key), jnp.arange(num_samples + burnin))[1]

    return xs[burnin:]