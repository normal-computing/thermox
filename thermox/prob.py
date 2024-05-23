import jax.numpy as jnp
from jax.lax import fori_loop
from jax import Array, vmap

from thermox.utils import (
    handle_matrix_inputs,
    preprocess_drift_matrix,
    ProcessedDriftMatrix,
    ProcessedDiffusionMatrix,
)


def log_prob_identity_diffusion(
    ts: Array,
    xs: Array,
    A: Array | ProcessedDriftMatrix,
    b: Array,
) -> float:
    """Calculates log probability of samples from the Ornstein-Uhlenbeck process,
    defined as:

    dx = - A * (x - b) dt + dW

    by using exact diagonalization.

    Assumes x(t_0) is given deterministically.

    Preprocessing (diagonalisation) costs O(d^3) and evaluation then costs O(T * d^2).

    Args:
        ts: Times at which samples are collected. Includes time for x0.
        xs: Initial state of the process.
        A: Drift matrix (Array or thermox.ProcessedDriftMatrix).
        b: Drift displacement vector.
    Returns:
        Scalar log probability of given xs.
    """
    if isinstance(A, Array):
        A = preprocess_drift_matrix(A)

    def expm_vp(v, dt):
        out = A.eigvecs_inv @ v
        out = jnp.exp(-A.eigvals * dt) * out
        out = A.eigvecs @ out
        return out.real

    def transition_mean(y, dt):
        return b + expm_vp(y - b, dt)

    def transition_cov_sqrt_inv_vp(v, dt):
        diag = ((1 - jnp.exp(-2 * A.sym_eigvals * dt)) / (2 * A.sym_eigvals)) ** 0.5
        diag = jnp.where(diag < 1e-20, 1e-20, diag)
        out = A.sym_eigvecs.T @ v
        out = out / diag
        return out.real

    def transition_cov_log_det(dt):
        diag = (1 - jnp.exp(-2 * A.sym_eigvals * dt)) / (2 * A.sym_eigvals)
        diag = jnp.where(diag < 1e-20, 1e-20, diag)
        return jnp.sum(jnp.log(diag))

    def logpt(yt, y0, dt):
        mean = transition_mean(y0, dt)
        diff_val = transition_cov_sqrt_inv_vp(yt - mean, dt)
        return (
            -jnp.dot(diff_val, diff_val) / 2
            - transition_cov_log_det(dt) / 2
            - jnp.log(2 * jnp.pi) * (yt.shape[0] / 2)
        )

    log_prob_val = fori_loop(
        1,
        len(ts),
        lambda i, val: val + logpt(xs[i], xs[i - 1], ts[i] - ts[i - 1]),
        0.0,
    )

    return log_prob_val.real


def log_prob(
    ts: Array,
    xs: Array,
    A: Array | ProcessedDriftMatrix,
    b: Array,
    D: Array | ProcessedDiffusionMatrix,
) -> Array:
    """Calculates log probability of samples from the Ornstein-Uhlenbeck process,
    defined as:

    dx = - A * (x - b) dt + sqrt(D) dW

    by using exact diagonalization.

    Assumes x(t_0) is given deterministically.

    Preprocessing (diagonalisation) costs O(d^3) and evaluation then costs O(T * d^2),
    where T=len(ts).

    By default, this function does the preprocessing on A and D before the evaluation.
    However, the preprocessing can be done externally using thermox.preprocess
    the output of which can be used as A and D here, this will skip the preprocessing.

    Args:
        ts: Times at which samples are collected. Includes time for x0.
        xs: Initial state of the process.
        A: Drift matrix (Array or thermox.ProcessedDriftMatrix).
            Note : If a thermox.ProcessedDriftMatrix instance is used as input,
            must be the transformed drift matrix, A_y, given by thermox.preprocess,
            not thermox.utils.preprocess_drift_matrix.
        b: Drift displacement vector.
        D: Diffusion matrix (Array or thermox.ProcessedDiffusionMatrix).

    Returns:
        Scalar log probability of given xs.
    """
    A_y, D = handle_matrix_inputs(A, D)

    ys = vmap(jnp.matmul, in_axes=(None, 0))(D.sqrt_inv, xs)
    b_y = D.sqrt_inv @ b
    log_prob_ys = log_prob_identity_diffusion(ts, ys, A_y, b_y)

    D_sqrt_inv_log_det = jnp.log(jnp.linalg.det(D.sqrt_inv))
    return log_prob_ys + D_sqrt_inv_log_det * (len(ts) - 1)
