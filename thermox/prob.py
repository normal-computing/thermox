import jax.numpy as jnp
from jax.lax import cond, fori_loop, scan, switch
from jax import Array, checkpoint, vmap

from thermox.utils import (
    handle_matrix_inputs,
    preprocess_drift_matrix,
    ProcessedDriftMatrix,
    ProcessedDiffusionMatrix,
)
from thermox.sampler import (
    expm_vp,
    transition_cov_eigh,
    transition_expm_and_cov,
    uniform_dt,
)


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
    where T=len(ts); when D^-0.5 @ A @ D^0.5 is not a normal matrix,
    O(d^3 + T * d^2) on any time grid, or O(T * d^3) when the run-time accuracy
    check of the interpolation across the gaps fails.

    By default, this function does the preprocessing on A and D before the evaluation.
    However, the preprocessing can be done externally using thermox.preprocess
    the output of which can be used as A and D here, this will skip the preprocessing.

    Args:
        ts: Times at which samples are collected. Includes time for x0.
        xs: States of the process.
        A: Drift matrix (Array or thermox.ProcessedDriftMatrix).
            Note: If a thermox.ProcessedDriftMatrix instance is used as input,
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


def log_prob_identity_diffusion(
    ts: Array,
    xs: Array,
    A: Array | ProcessedDriftMatrix,
    b: Array,
) -> float:
    if isinstance(A, Array):
        A = preprocess_drift_matrix(A)
    if len(ts) < 3:
        return _log_prob_identity_diffusion_stepwise(ts, xs, A, b)
    # A non-normal A: build the transition operator once on a uniform grid, or
    # interpolate the transition covariance's log-determinant and inverse across
    # the gaps on any other grid, falling back to the per-step path when that
    # interpolation fails its run-time check.
    is_uniform, dt = uniform_dt(ts)
    index = jnp.where(A.is_normal, 0, jnp.where(is_uniform, 1, 2)).astype(jnp.int32)
    return switch(
        index,
        [
            _log_prob_identity_diffusion_stepwise,
            lambda ts, xs, A, b: _log_prob_identity_diffusion_uniform(ts, xs, A, b, dt),
            _log_prob_identity_diffusion_panels,
        ],
        ts,
        xs,
        A,
        b,
    )


# cond and switch save every branch's residuals for the backward pass; with
# checkpoint this branch recomputes its per-step intermediates instead of storing them.
@checkpoint
def _log_prob_identity_diffusion_stepwise(ts, xs, A, b):
    def transition_mean(y, dt):
        return b + expm_vp(A, y - b, dt)

    def logpt(yt, y0, dt):
        diff = yt - transition_mean(y0, dt)

        def mahalanobis_and_log_det(w, U):
            w = jnp.where(w < 1e-20, 1e-20, w)
            diff_val = (U.T @ diff) / jnp.sqrt(w)
            return jnp.dot(diff_val, diff_val), jnp.sum(jnp.log(w))

        # One factorization of the transition covariance per step serves both terms.
        quad, log_det = transition_cov_eigh(A, dt, mahalanobis_and_log_det)
        return -quad / 2 - log_det / 2 - jnp.log(2 * jnp.pi) * (yt.shape[0] / 2)

    log_prob_val = fori_loop(
        1,
        len(ts),
        lambda i, val: val + logpt(xs[i], xs[i - 1], ts[i] - ts[i - 1]),
        0.0,
    )

    return log_prob_val.real


def _log_prob_identity_diffusion_uniform(ts, xs, A, b, dt):
    """log_prob_identity_diffusion on a uniform grid: the residuals of all
    steps with gap dt at once, one eigendecomposition of their common
    covariance, and one term for the first gap."""
    E1, cov1 = transition_expm_and_cov(A.val, ts[1] - ts[0])
    E, cov = transition_expm_and_cov(A.val, dt)
    residuals1 = xs[1] - b - E1 @ (xs[0] - b)
    residuals = xs[2:] - b - (xs[1:-1] - b) @ E.T

    def log_density(cov, r):
        w, U = jnp.linalg.eigh(cov)
        w = jnp.where(w < 1e-20, 1e-20, w)
        z = (r @ U) / jnp.sqrt(w)
        n, d = r.shape
        return (
            -jnp.sum(z * z) / 2
            - n * (jnp.sum(jnp.log(w)) + d * jnp.log(2 * jnp.pi)) / 2
        )

    return log_density(cov1, residuals1[None]) + log_density(cov, residuals)


def _chebyshev_matrix(N):
    """(N + 1) x (N + 1) matrix taking values at the Chebyshev points cos(m pi / N)
    to the coefficients of the interpolating polynomial in T_0, ..., T_N.
    """
    m = jnp.arange(N + 1)
    h = jnp.where((m == 0) | (m == N), 0.5, 1.0)
    return (2.0 / N) * h[:, None] * jnp.cos(jnp.pi * jnp.outer(m, m) / N) * h


def _log_prob_panels(ts, xs, A, b):
    """Log-density of the trajectory from Chebyshev panels: for each octave of the
    gaps that some step falls in, N + 1 = 17 nodes, one eigh per node, the
    Chebyshev coefficients of log det cov(dt) (scalars) and of cov(dt)^-1
    (matrices) by one cosine transform, and the interpolants at every step in
    the panel, O(d^2) per step. Returns (value, ok); ok is false when a panel's
    last two coefficients are not below the tolerance, a gap is zero, or the
    gaps span more than P = 53 octaves.
    """
    # tol is the accuracy of transition_expm_and_cov in this dtype; N is the
    # Chebyshev degree per panel and P the number of octaves of gaps handled.
    dtype = ts.dtype
    tol = 1e-10 if dtype == jnp.float64 else 3e-4
    N, P = 16, 53
    dts = jnp.diff(ts)
    positive = dts > 0
    tau_min = jnp.min(jnp.where(positive, dts, jnp.inf))
    # frexp places every gap in its octave panel p, with edges tau_min 2^p and
    # tau_min 2^(p + 1) exactly, and x is the gap's coordinate in [-1, 1] there.
    m, e = jnp.frexp(jnp.where(positive, dts, tau_min) / tau_min)
    p, x = e - 1, 4.0 * m - 3.0
    x_nodes = jnp.cos(jnp.pi * jnp.arange(N + 1) / N).astype(dtype)
    C_N = _chebyshev_matrix(N).astype(dtype)
    # T_n(x) at every step by the three-term recurrence rather than cos(n arccos x),
    # whose derivative is infinite at x = -1 (the gradient with respect to ts needs it).
    W = [jnp.ones_like(x), x]
    for _ in range(N - 1):
        W.append(2 * x * W[-1] - W[-2])
    W = jnp.stack(W)
    # Residuals of every step, as on the per-step path.
    residuals = xs[1:] - b - vmap(lambda y, dt: expm_vp(A, y - b, dt))(xs[:-1], dts)

    # Reverse mode recomputes each panel instead of storing its node covariances.
    @checkpoint
    def panel(carry, j):
        terms, ok = carry
        mask = p == j

        def body(terms, ok):
            taus = jnp.ldexp(tau_min, j) * (1.5 + 0.5 * x_nodes)
            covs = vmap(lambda t: transition_expm_and_cov(A.val, t)[1])(taus)
            w, U = jnp.linalg.eigh(covs)
            w = jnp.where(w < 1e-20, 1e-20, w)
            # Chebyshev coefficients of log det cov (c) and of cov^-1 (C_inv).
            c = C_N @ jnp.sum(jnp.log(w), axis=1)
            C_inv = jnp.einsum("nm,mij,mj,mkj->nik", C_N, U, 1.0 / w, U)
            q = jnp.einsum(
                "tn,nt->t", jnp.einsum("ti,nij,tj->tn", residuals, C_inv, residuals), W
            )
            norms = jnp.linalg.norm(C_inv, axis=(1, 2))
            # Both series have converged: the last two coefficients are below tol.
            tail_ok = (jnp.max(jnp.abs(c[-2:])) <= tol * jnp.max(jnp.abs(c))) & (
                jnp.max(norms[-2:]) <= tol * jnp.max(norms)
            )
            return jnp.where(mask, q + c @ W, terms), ok & tail_ok

        return cond(jnp.any(mask), body, lambda t, o: (t, o), terms, ok), None

    init = (jnp.zeros(len(dts), dtype), jnp.array(True))
    (terms, ok), _ = scan(panel, init, jnp.arange(P))
    value = -0.5 * jnp.sum(terms + xs.shape[1] * jnp.log(2 * jnp.pi))
    return value, ok & jnp.all(positive) & (jnp.max(e) <= P)


def _log_prob_identity_diffusion_panels(ts, xs, A, b):
    # The panels' value when their run-time check passes, the per-step path's otherwise.
    value, ok = _log_prob_panels(ts, xs, A, b)
    return cond(
        ok, lambda: value, lambda: _log_prob_identity_diffusion_stepwise(ts, xs, A, b)
    )
