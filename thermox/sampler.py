from functools import partial
import jax
import jax.numpy as jnp
from jax import Array

from thermox.utils import (
    handle_matrix_inputs,
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
        key: Jax PRNGKey.
        ts: Times at which samples are collected. Includes time for x0.
        x0: Initial state of the process.
        A: Drift matrix (Array or thermox.ProcessedDriftMatrix).
        b: Drift displacement vector.

    Returns:
        Array-like, desired samples.
            shape: (len(ts), ) + x0.shape
    """

    if isinstance(A, Array):
        A = preprocess_drift_matrix(A)

    def expm_vp(v, dt):
        out = A.eigvecs_inv @ v
        out = jnp.exp(-A.eigvals * dt) * out
        out = A.eigvecs @ out
        return out.real

    def transition_cov_sqrt_vp(v, dt):
        diag = ((1 - jnp.exp(-2 * A.sym_eigvals * dt)) / (2 * A.sym_eigvals)) ** 0.5
        out = diag * v
        out = A.sym_eigvecs @ out
        return out.real

    dts = jnp.diff(ts)

    # transition_mean(x, dt) = b + expm_vp(x - b, dt)
    def position_indep_mean_component(dt):
        return b - expm_vp(b, dt)

    def position_dep_mean_component(x, dt):
        return expm_vp(x, dt)

    gauss_samps = jax.random.normal(key, (len(dts),) + x0.shape)
    position_indep_terms = jax.vmap(transition_cov_sqrt_vp)(
        gauss_samps, dts
    ) + jax.vmap(position_indep_mean_component)(dts)

    @partial(jax.vmap, in_axes=(0, 0))
    def binary_associative_operator(elem_a, elem_b):
        t_a, x_a = elem_a
        t_b, x_b = elem_b
        return t_a + t_b, position_dep_mean_component(x_a, t_b) + x_b

    scan_times = jnp.concatenate([ts[:1], dts], dtype=float)  # [t0, dt1, dt2, ...]
    scan_input_values = jnp.concatenate([x0[None], position_indep_terms], axis=0)
    scan_elems = (scan_times, scan_input_values)

    scan_output = jax.lax.associative_scan(binary_associative_operator, scan_elems)
    return scan_output[1]


def sample(
    key: Array,
    ts: Array,
    x0: Array,
    A: Array | ProcessedDriftMatrix,
    b: Array,
    D: Array | ProcessedDiffusionMatrix,
) -> Array:
    """Collects samples from the Ornstein-Uhlenbeck process, defined as:

    dx = - A * (x - b) dt + sqrt(D) dW

    by using exact diagonalization.

    Preprocessing (diagonalisation) costs O(d^3) and sampling costs O(T * d^2),
    where T=len(ts).

    By default, this function does the preprocessing on A and D before the evaluation.
    However, the preprocessing can be done externally using thermox.preprocess
    the output of which can be used as A and D here, this will skip the preprocessing.

    Args:
        key: Jax PRNGKey.
        ts: Times at which samples are collected. Includes time for x0.
        x0: Initial state of the process.
        A: Drift matrix (Array or thermox.ProcessedDriftMatrix).
            Note : If a thermox.ProcessedDriftMatrix instance is used as input,
            must be the transformed drift matrix, A_y, given by thermox.preprocess,
            not thermox.utils.preprocess_drift_matrix.
        b: Drift displacement vector.
        D: Diffusion matrix (Array or thermox.ProcessedDiffusionMatrix).

    Returns:
        Array-like, desired samples.
            shape: (len(ts), ) + x0.shape
    """
    A_y, D = handle_matrix_inputs(A, D)

    y0 = D.sqrt_inv @ x0
    b_y = D.sqrt_inv @ b
    ys = sample_identity_diffusion(key, ts, y0, A_y, b_y)
    return jax.vmap(jnp.matmul, in_axes=(None, 0))(D.sqrt, ys)
