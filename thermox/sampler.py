import jax
import jax.numpy as jnp
from jax.lax import scan
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

    def transition_mean(x, dt):
        return b + expm_vp(x - b, dt)

    def transition_cov_sqrt_vp(v, dt):
        diag = ((1 - jnp.exp(-2 * A.sym_eigvals * dt)) / (2 * A.sym_eigvals)) ** 0.5
        out = diag * v
        out = A.sym_eigvecs @ out
        return out.real

    # def next_x(x, dt, tkey):
    #     randv = jax.random.normal(tkey, shape=x.shape)
    #     return transition_mean(x, dt) + transition_cov_sqrt_vp(randv, dt)

    # def scan_body(x_and_key, dt):
    #     x, rk = x_and_key
    #     rk, rk_use = jax.random.split(rk)
    #     x = next_x(x, dt, rk_use)
    #     return (x, rk), x

    dts = jnp.diff(ts)

    noise_terms = jax.vmap(transition_cov_sqrt_vp)(
        jax.random.normal(key, (len(dts),) + x0.shape), dts
    )

    def binary_associative_operator(elem_a, elem_b):
        t_a, x_a = elem_a
        t_b, x_b = elem_b
        return t_a + t_b, transition_mean(x_a, t_b) + x_b

    scan_times = jnp.concatenate([ts[:1], dts], dtype=float)  # [t0, dt1, dt2, ...]
    scan_input_values = jnp.concatenate([x0[None], noise_terms], axis=0)
    scan_elems = (scan_times, scan_input_values)

    # elem_a = (scan_elems[0][0], scan_elems[1][0])
    # elem_b = (scan_elems[0][1], scan_elems[1][1])
    # elem_c = (scan_elems[0][2], scan_elems[1][2])

    # binary_associative_operator(elem_a, binary_associative_operator(elem_b, elem_c))
    # binary_associative_operator(binary_associative_operator(elem_a, elem_b), elem_c)

    scan_output = jax.lax.associative_scan(binary_associative_operator, scan_elems)

    vals = [(scan_elems[0][0], scan_elems[1][0])]
    for i in range(len(scan_elems[0])):
        vals.append(
            binary_associative_operator(vals[-1], (scan_elems[0][i], scan_elems[1][i]))
        )

    # xs = scan(scan_body, (x0, key), dts)[1]
    # xs = jnp.concatenate([jnp.expand_dims(x0, axis=0), xs], axis=0)
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
