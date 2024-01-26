import jax
import jax.numpy as jnp
from jax.lax import scan
from jax import Array


def collect_samples_identity_diffusion(
    key: Array,
    ts: Array,
    x0: Array,
    A: Array,
    b: Array,
) -> Array:
    """
    Collects samples from the Ornstein-Uhlenbeck process, defined as:

    dx = - A * (x - b) dt + dW

    by using exact diagonalization.

    Args:
        - key: jax PRNGKey.
        - ts: array-like, times at which samples are collected. Includes time for x0.
        - x0: initial state of the process.
        - A: drift matrix.
        - b: drift displacement vector.

    Returns:
        - samples: array-like, desired samples.
            shape: (len(ts), ) + x0.shape
    """

    eigvals, eigvecs = jnp.linalg.eig(A)
    eigvecs_inv = jnp.linalg.inv(eigvecs)

    def expm_vp(v, dt):
        out = eigvecs_inv @ v
        out = jnp.exp(-eigvals * dt) * out
        out = eigvecs @ out
        return out.real

    def transition_mean(x, dt):
        return b + expm_vp(x - b, dt)

    symA = 0.5 * (A + A.T)
    symA_eigvals, symA_eigvecs = jnp.linalg.eig(symA)

    def transition_cov_sqrt_vp(v, dt):
        diag = ((1 - jnp.exp(-2 * symA_eigvals * dt)) / (2 * symA_eigvals)) ** 0.5
        out = diag * v
        out = symA_eigvecs @ out
        return out.real

    def next_x(x, dt, tkey):
        randv = jax.random.normal(tkey, shape=x.shape)
        return transition_mean(x, dt) + transition_cov_sqrt_vp(randv, dt)

    def scan_body(x_and_key, dt):
        x, rk = x_and_key
        rk, rk_use = jax.random.split(rk)
        x = next_x(x, dt, rk_use)
        return (x, rk), x

    dts = jnp.diff(ts)

    xs = scan(scan_body, (x0, key), dts)[1]
    xs = jnp.concatenate([jnp.expand_dims(x0, axis=0), xs], axis=0)
    return xs


def collect_samples(
    key: Array, ts: Array, x0: Array, A: Array, b: Array, D: Array
) -> Array:
    """
    Collects samples from the Ornstein-Uhlenbeck process, defined as:

    dx = - A * (x - b) dt + sqrt(D) dW

    by using exact diagonalization.

    Args:
        - key: jax PRNGKey.
        - ts: array-like, times at which samples are collected. Includes time for x0.
        - x0: initial state of the process.
        - A: drift matrix.
        - b: drift displacement vector.
        - D: diffusion matrix.

    Returns:
        - samples: array-like, desired samples.
            shape: (len(ts), ) + x0.shape
    """
    D_sqrt = jnp.linalg.cholesky(D)
    D_sqrt_inv = jnp.linalg.inv(D_sqrt)
    y0 = D_sqrt_inv @ x0
    A_y = D_sqrt_inv @ A @ D_sqrt
    b_y = D_sqrt_inv @ b
    ys = collect_samples_identity_diffusion(key, ts, y0, A_y, b_y)
    return jax.vmap(jnp.matmul, in_axes=(None, 0))(D_sqrt, ys)

