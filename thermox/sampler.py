import jax
import jax.numpy as jnp
from jax.lax import scan
from jax import Array

from thermox.utils import (
    preprocess,
    preprocess_drift_matrix,
    ProcessedDriftMatrix,
    ProcessedDiffusionMatrix,
)


def sample_identity_diffusion(
    key: Array,
    ts: Array,
    x0: Array,
    A: Array | ProcessedDriftMatrix,
    b: Array,
) -> Array:
    """Collects samples from the Ornstein-Uhlenbeck process, defined as:

    dx = - A * (x - b) dt + dW

    by using exact diagonalization.

    Preprocessing (diagonalisation) costs O(d^3) and sampling costs O(T * d^2)
    where T=len(ts).

    Args:
        - key: jax PRNGKey.
        - ts: array-like, times at which samples are collected. Includes time for x0.
        - x0: initial state of the process.
        - A: drift matrix (Array or thermox.ProcessedDriftMatrix).
        - b: drift displacement vector.

    Returns:
        - samples: array-like, desired samples.
            shape: (len(ts), ) + x0.shape
    """

    if isinstance(A, Array):
        A = preprocess_drift_matrix(A)

    def expm_vp(v, dt):
        out = A.eigvecs_inv @ v
        out = jnp.exp(-A.eigvals * dt) * out
        out = A.eigvecs @ out
        return out.real

    def transition_mean(x, dt):
        return b + expm_vp(x - b, dt)

    def transition_cov_sqrt_vp(v, dt):
        diag = ((1 - jnp.exp(-2 * A.sym_eigvals * dt)) / (2 * A.sym_eigvals)) ** 0.5
        out = diag * v
        out = A.sym_eigvecs @ out
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


def sample(
    key: Array,
    ts: Array,
    x0: Array,
    A: Array | ProcessedDriftMatrix,
    b: Array,
    D: Array | ProcessedDiffusionMatrix,
    A_spd: bool = False,
) -> Array:
    """Collects samples from the Ornstein-Uhlenbeck process, defined as:

    dx = - A * (x - b) dt + sqrt(D) dW

    by using exact diagonalization.

    Preprocessing (diagonalisation) costs O(d^3) and sampling costs O(T * d^2)
    where T=len(ts).

    Args:
        - key: jax PRNGKey.
        - ts: array-like, times at which samples are collected. Includes time for x0.
        - x0: initial state of the process.
        - A: drift matrix (Array or thermox.ProcessedDriftMatrix).
        - b: drift displacement vector.
        - D: diffusion matrix (Array or thermox.ProcessedDiffusionMatrix).
        - A_spd: if true uses jax.linalg.eigh to calculate eigendecomposition of A.
            If false uses jax.scipy.linalg.eig.
            jax.linalg.eigh supports gradients but assumes A is Hermitian
            (i.e. real symmetric).
            See https://github.com/google/jax/issues/2748

    Returns:
        - samples: array-like, desired samples.
            shape: (len(ts), ) + x0.shape
    """
    if isinstance(A, Array) and isinstance(D, Array):
        A_y, D = preprocess(A, D, A_spd)

    assert isinstance(A_y, ProcessedDriftMatrix)
    assert isinstance(D, ProcessedDiffusionMatrix)

    y0 = D.sqrt_inv @ x0
    b_y = D.sqrt_inv @ b
    ys = sample_identity_diffusion(key, ts, y0, A_y, b_y)
    return jax.vmap(jnp.matmul, in_axes=(None, 0))(D.sqrt, ys)
