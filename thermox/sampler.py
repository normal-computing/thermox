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


def sample(
    key: Array,
    ts: Array,
    x0: Array,
    A: Array | ProcessedDriftMatrix,
    b: Array,
    D: Array | ProcessedDiffusionMatrix,
    associative_scan: bool = True,
) -> Array:
    """Collects samples from the Ornstein-Uhlenbeck process, defined as:

    dx = - A * (x - b) dt + sqrt(D) dW

    by using exact diagonalization.

    Preprocessing (diagonalization) costs O(d^3) and sampling costs O(T * d^2),
    where T=len(ts); when D^-0.5 @ A @ D^0.5 is not a normal matrix,
    O(d^3 + T * d^2) on any time grid.

    If associative_scan=True then jax.lax.associative_scan is used which will run in
    time O((T/p + log(T)) * d^2) on a GPU/TPU with p cores, still with
    O(d^3) preprocessing.

    By default, this function does the preprocessing on A and D before the evaluation.
    However, the preprocessing can be done externally using thermox.preprocess
    the output of which can be used as A and D here, this will skip the preprocessing.

    Args:
        key: Jax PRNGKey.
        ts: Times at which samples are collected. Includes time for x0.
        x0: Initial state of the process.
        A: Drift matrix (Array or thermox.ProcessedDriftMatrix).
            Note: If a thermox.ProcessedDriftMatrix instance is used as input,
            must be the transformed drift matrix, A_y, given by thermox.preprocess,
            not thermox.utils.preprocess_drift_matrix.
        b: Drift displacement vector.
        D: Diffusion matrix (Array or thermox.ProcessedDiffusionMatrix).
        associative_scan: If True, uses jax.lax.associative_scan.

    Returns:
        Array-like, desired samples.
            shape: (len(ts), ) + x0.shape
    """
    A_y, D = handle_matrix_inputs(A, D)

    y0 = D.sqrt_inv @ x0
    b_y = D.sqrt_inv @ b
    ys = sample_identity_diffusion(key, ts, y0, A_y, b_y, associative_scan)
    return jax.vmap(jnp.matmul, in_axes=(None, 0))(D.sqrt, ys)


def sample_identity_diffusion(
    key: Array,
    ts: Array,
    x0: Array,
    A: Array | ProcessedDriftMatrix,
    b: Array,
    associative_scan: bool = True,
) -> Array:
    if isinstance(A, Array):
        A = preprocess_drift_matrix(A)
    if associative_scan:
        stepwise = _sample_identity_diffusion_associative_scan
    else:
        stepwise = _sample_identity_diffusion_scan
    if len(ts) < 3:
        return stepwise(key, ts, x0, A, b)
    # A non-normal A: build the transition operator once on a uniform grid, or one
    # per binary digit of the gaps (the ladder) on any other grid.
    is_uniform, dt = uniform_dt(ts)
    index = jnp.where(A.is_normal, 0, jnp.where(is_uniform, 1, 2)).astype(jnp.int32)
    return jax.lax.switch(
        index,
        [
            stepwise,
            lambda *args: _sample_identity_diffusion_uniform(
                *args, dt, associative_scan
            ),
            lambda *args: _sample_identity_diffusion_ladder(*args, associative_scan),
        ],
        key,
        ts,
        x0,
        A,
        b,
    )


def expm_vp(A, v, dt):
    out = A.eigvecs_inv @ v
    out = jnp.exp(-A.eigvals * dt) * out
    out = A.eigvecs @ out
    return out.real


def transition_expm_and_cov(A, dt, n_doublings=12):
    """exp(-A dt) and int_0^dt exp(-A s) exp(-A^T s) ds for a d x d matrix A,
    without an eigendecomposition: Van Loan's block exponential at
    h = dt / 2**n_doublings, then n_doublings steps of E(2h) = E(h)^2 and
    cov(2h) = cov(h) + E(h) cov(h) E(h)^T. Exact for any stable A, including
    dt = 0, for ||A|| dt up to about 1e5 with the default n_doublings.
    """
    d = A.shape[0]
    h = dt / 2**n_doublings
    zeros, eye = jnp.zeros((d, d), dtype=A.dtype), jnp.eye(d, dtype=A.dtype)
    # h is small, so expm needs few squarings; its loop always runs max_squarings.
    F = jax.scipy.linalg.expm(jnp.block([[-A, eye], [zeros, A.T]]) * h, max_squarings=4)
    E = F[:d, :d]
    cov = F[:d, d:] @ E.T
    for _ in range(n_doublings):
        cov = cov + E @ cov @ E.T
        E = E @ E
    return E, 0.5 * (cov + cov.T)


def transition_cov(A, dt):
    """Covariance of x_dt given x_0 for dx = -A x dt + dW, i.e.
    int_0^dt exp(-A s) exp(-A^T s) ds. Exact for any stable A.
    """
    return transition_expm_and_cov(A.val, dt)[1]


def transition_cov_eigh(A, dt, apply=lambda w, U: (w, U)):
    """Spectral factorization transition_cov(A, dt) = U diag(w) U^T, returned as
    apply(w, U).

    Branches on A.is_normal. Normal A: U = A.sym_eigvecs is precomputed and w is
    a closed-form function of dt, O(d^2) per step. Otherwise transition_cov(A, dt)
    is eigendecomposed at each step, O(d^3). apply is evaluated inside the branch
    so that only its result, not a d x d matrix per step, leaves the lax.cond.

    Gradients with respect to A follow the branch taken: at an exactly normal A
    they are those of the normal-branch formula, which depends on A only through
    (A + A^T)/2.
    """

    def normal(A, dt):
        w = (1 - jnp.exp(-2 * A.sym_eigvals * dt)) / (2 * A.sym_eigvals)
        return apply(w, A.sym_eigvecs)

    def general(A, dt):
        # eigh rather than Cholesky: stays well defined at dt = 0 (zero covariance).
        w, U = jnp.linalg.eigh(transition_cov(A, dt))
        return apply(w, U)

    # apply is evaluated inside the branches so the cond returns a small result;
    # returning (w, U) and applying it outside made the associative-scan path
    # measurably slower (vmap's cond batching rule broadcasts the d x d U over all
    # steps).
    return jax.lax.cond(A.is_normal, normal, general, A, dt)


def transition_cov_sqrt_vp(A, v, dt):
    return transition_cov_eigh(
        A, dt, lambda w, U: U @ (jnp.sqrt(jnp.maximum(w, 0.0)) * v)
    )


def uniform_dt(ts):
    """Whether the time grid is uniform after its first gap, up to floating-point
    rounding of the time stamps, and that step. The first gap is free so that
    the grids built by thermox.linalg ([0, burnin * dt, dt, ...]) qualify.
    """
    n = len(ts) - 2
    dt = (ts[-1] - ts[1]) / n
    fitted = ts[1] + dt * jnp.arange(n + 1)
    eps = jnp.finfo(jnp.result_type(ts, float)).eps
    is_uniform = jnp.max(jnp.abs(ts[1:] - fitted)) <= 1e3 * eps * jnp.max(jnp.abs(ts))
    return is_uniform, dt


def _scan_linear_recurrence(E, y0, u):
    """y_k = E y_{k-1} + u_k for k = 1, ..., n, computed like
    jax.lax.associative_scan but with the level's power of E passed down: a
    combine at depth j applies E ** (2 ** j), one matmul per level, so the
    scan costs O(d^3 log n + n d^2) time and O(d^2 log n) memory. Carrying
    the power inside the scanned elements instead would store one d x d
    matrix per step, O(n d^2) memory.
    """

    def scan(elems, M):
        # The recursion of jax.lax.associative_scan: combine adjacent pairs,
        # recurse on the pairs, fill in the even positions, interleave.
        m = elems.shape[0]
        if m < 2:
            return elems
        reduced = elems[0:-1:2] @ M.T + elems[1::2]
        odd = scan(reduced, M @ M)
        even = jnp.concatenate(
            [elems[:1], (odd[:-1] if m % 2 == 0 else odd) @ M.T + elems[2::2]]
        )
        # Interleave [even0, odd0, even1, odd1, ...]; len(even) is len(odd) or len(odd) + 1.
        same = even.shape[0] == odd.shape[0]
        zero, rest = jnp.zeros((), even.dtype), [(0, 0, 0)] * (even.ndim - 1)
        return jax.lax.pad(even, zero, [(0, int(same), 1)] + rest) + jax.lax.pad(
            odd, zero, [(1, int(not same), 1)] + rest
        )

    return scan(jnp.concatenate([y0[None], u]), E)[1:]


def _sample_identity_diffusion_uniform(key, ts, x0, A, b, dt, associative_scan):
    # One transition operator for the first gap, one for dt, applied to the
    # same draws as the per-step engines.
    E1, cov1 = transition_expm_and_cov(A.val, ts[1] - ts[0])
    E, cov = transition_expm_and_cov(A.val, dt)
    # The one-sided factor U sqrt(w) of the per-step path, once per operator.
    w1, U1 = jnp.linalg.eigh(cov1)
    w, U = jnp.linalg.eigh(cov)
    z = jax.random.normal(key, (len(ts) - 1,) + x0.shape)
    y1 = E1 @ (x0 - b) + U1 @ (jnp.sqrt(jnp.maximum(w1, 0.0)) * z[0])
    u = (z[1:] * jnp.sqrt(jnp.maximum(w, 0.0))) @ U.T
    if associative_scan:
        ys = _scan_linear_recurrence(E, y1, u)
    else:
        _, ys = jax.lax.scan(lambda y, u_k: (E @ y + u_k,) * 2, y1, u)
    return jnp.concatenate([x0[None], y1[None] + b, ys + b])


def _ladder_lattice(ts):
    """Gaps of ts as integers on the lattice delta = 2^(e - M), where 2^e is the
    power of two just above the largest gap and M the number of mantissa bits
    of ts (52 in float64, 23 in float32): M + 1 binary digits, and every gap is
    reproduced to the rounding of the time stamps themselves. A power of two
    keeps dts / delta exact, so the digits do not depend on how it is compiled.
    """
    dts = jnp.diff(ts)
    M = jnp.finfo(ts.dtype).nmant
    delta = jnp.ldexp(jnp.ones((), ts.dtype), jnp.frexp(jnp.max(dts))[1] - M)
    n = jnp.round(dts / delta).astype(jnp.int64 if M > 23 else jnp.int32)
    return delta, n, M


def _ladder_noise(A, ts, key):
    """Noise terms of every step, shape (len(ts) - 1, d), from a dyadic ladder of
    transition operators: for level j the pair E_j = exp(-A delta 2^j),
    L_j L_j^T = cov(delta 2^j) is built once and applied to the steps whose gap
    has bit j set, e <- E_j e + L_j z_j, which composes the covariances exactly
    (cov(a + b) = cov(b) + E(b) cov(a) E(b)^T). M + 1 operators once, two
    masked products per level and step.
    """
    delta, n, M = _ladder_lattice(ts)
    d = A.val.shape[0]

    def level(e, j):
        E, cov = transition_expm_and_cov(A.val, jnp.ldexp(delta, j))
        # Cholesky, not eigh: every level's gap is positive, and the factor is unique,
        # so draws at a fixed key do not depend on how the level was compiled.
        L = jnp.linalg.cholesky(cov)
        z = jax.random.normal(jax.random.fold_in(key, j), (len(n), d))
        bit = ((n >> j.astype(n.dtype)) & 1).astype(bool)
        return jnp.where(bit[:, None], e @ E.T + z @ L.T, e), None

    noise, _ = jax.lax.scan(level, jnp.zeros((len(n), d)), jnp.arange(M + 1))
    return noise


def _sample_identity_diffusion_ladder(key, ts, x0, A, b, associative_scan):
    # Noise from the ladder, mean through the eigenbasis as on the per-step paths.
    dts = jnp.diff(ts)
    noise = _ladder_noise(A, ts, key)
    if associative_scan:

        @partial(jax.vmap, in_axes=(0, 0))
        def binary_associative_operator(elem_a, elem_b):
            t_a, x_a = elem_a
            t_b, x_b = elem_b
            return t_a + t_b, expm_vp(A, x_a, t_b) + x_b

        scan_times = jnp.concatenate([ts[:1], dts])
        scan_values = jnp.concatenate([x0[None] - b, noise])
        return (
            jax.lax.associative_scan(
                binary_associative_operator, (scan_times, scan_values)
            )[1]
            + b
        )

    def step(x, dt_and_u):
        dt, u = dt_and_u
        x = b + expm_vp(A, x - b, dt) + u
        return x, x

    _, xs = jax.lax.scan(step, x0.astype(noise.dtype), (dts, noise))
    return jnp.concatenate([x0[None], xs])


def _sample_identity_diffusion_scan(
    key: Array,
    ts: Array,
    x0: Array,
    A: Array | ProcessedDriftMatrix,
    b: Array,
) -> Array:
    if isinstance(A, Array):
        A = preprocess_drift_matrix(A)

    def transition_mean(x, dt):
        return b + expm_vp(A, x - b, dt)

    def next_x(x, dt, rv):
        return transition_mean(x, dt) + transition_cov_sqrt_vp(A, rv, dt)

    def scan_body(carry, dt_and_rv):
        x = carry
        dt, rv = dt_and_rv
        new_x = next_x(x, dt, rv)
        return new_x, new_x

    dts = jnp.diff(ts)
    gauss_samps = jax.random.normal(key, (len(dts),) + x0.shape)

    # Stack dts and gauss_samps along a new axis
    dt_and_rv = (dts, gauss_samps)

    _, xs = jax.lax.scan(scan_body, x0, dt_and_rv)
    xs = jnp.concatenate([jnp.expand_dims(x0, axis=0), xs], axis=0)
    return xs


def _sample_identity_diffusion_associative_scan(
    key: Array,
    ts: Array,
    x0: Array,
    A: Array | ProcessedDriftMatrix,
    b: Array,
) -> Array:
    if isinstance(A, Array):
        A = preprocess_drift_matrix(A)

    dts = jnp.diff(ts)

    # transition_mean(x, dt) = b + expm_vp(A, x - b, dt)

    gauss_samps = jax.random.normal(key, (len(dts),) + x0.shape)
    noise_terms = jax.vmap(lambda v, dt: transition_cov_sqrt_vp(A, v, dt))(
        gauss_samps, dts
    )

    @partial(jax.vmap, in_axes=(0, 0))
    def binary_associative_operator(elem_a, elem_b):
        t_a, x_a = elem_a
        t_b, x_b = elem_b
        return t_a + t_b, expm_vp(A, x_a, t_b) + x_b

    scan_times = jnp.concatenate([ts[:1], dts], dtype=float)  # [t0, dt1, dt2, ...]
    scan_input_values = jnp.concatenate(
        [x0[None] - b, noise_terms], axis=0
    )  # Shift input by b
    scan_elems = (scan_times, scan_input_values)

    scan_output = jax.lax.associative_scan(binary_associative_operator, scan_elems)
    return scan_output[1] + b  # Shift back by b
