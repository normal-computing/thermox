"""Exactness checks against references (Van Loan, Lyapunov) that do not go
through thermox: drifts whose transformed form D^{-1/2} A D^{1/2} is not normal,
plus normal cases whose results must not change.
"""

import jax
import jax.numpy as jnp
import pytest

import thermox
from thermox.sampler import _scan_linear_recurrence, uniform_dt
from thermox.utils import preprocess_drift_matrix

jax.config.update("jax_enable_x64", True)

A_SYM = jnp.array([[3.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 5.0]])
A_TRI = jnp.array([[2.0, 1.5, 0.0], [0.0, 3.0, 1.5], [0.0, 0.0, 4.0]])
A_ROT = jnp.array([[1.0, 2.0], [-2.0, 1.0]])
D_DIAG = jnp.diag(jnp.array([1.0, 4.0, 9.0]))
D_DENSE = jnp.array([[1.0, 0.3, -0.1], [0.3, 1.0, 0.2], [-0.1, 0.2, 1.0]])

NONNORMAL_CASES = [
    pytest.param(A_SYM, D_DIAG, id="symmetric-A-diagonal-D"),
    pytest.param(A_SYM, D_DENSE, id="symmetric-A-dense-D"),
    pytest.param(A_TRI, jnp.eye(3), id="triangular-A-identity-D"),
]
NORMAL_CASES = [
    pytest.param(A_SYM, jnp.eye(3), id="symmetric-A-identity-D"),
    pytest.param(A_ROT, jnp.eye(2), id="rotation-A-identity-D"),
]


def van_loan_covariance(A, D, t):
    """int_0^t exp(-A s) D exp(-A^T s) ds via Van Loan (1978)."""
    d = A.shape[0]
    M = jnp.block([[-A, D], [jnp.zeros((d, d)), A.T]]) * t
    F = jax.scipy.linalg.expm(M)
    return F[:d, d:] @ jax.scipy.linalg.expm(-A.T * t)


def lyapunov_covariance(A, D):
    """Solve A S + S A^T = D by Kronecker vectorization (small d)."""
    d = A.shape[0]
    eye = jnp.eye(d)
    K = jnp.kron(eye, A) + jnp.kron(A, eye)
    return jnp.linalg.solve(K, D.reshape(-1, order="F")).reshape(d, d, order="F")


def reference_covariance(A, D, t):
    """Sigma_t = Sigma_inf - exp(-A t) Sigma_inf exp(-A^T t), valid for stable A.

    Numerically benign for large t, unlike the block exponential.
    """
    S_inf = lyapunov_covariance(A, D)
    E = jax.scipy.linalg.expm(-A * t)
    return S_inf - E @ S_inf @ E.T


def relerr(x, y):
    return jnp.linalg.norm(x - y) / jnp.linalg.norm(y)


@pytest.mark.parametrize("A,D", NONNORMAL_CASES + NORMAL_CASES)
def test_references_agree(A, D):
    # Guard the references themselves: two independent formulas for Sigma_t.
    assert (
        relerr(van_loan_covariance(A, D, 0.7), reference_covariance(A, D, 0.7)) < 1e-10
    )


@pytest.mark.parametrize("A,D", NONNORMAL_CASES + NORMAL_CASES)
@pytest.mark.parametrize("t", [0.05, 0.7, 5.0])
def test_conditional_covariance_matches_reference(A, D, t):
    cov = thermox.conditional.covariance(t, A, D)
    assert relerr(cov, reference_covariance(A, D, t)) < 1e-8


@pytest.mark.parametrize("A,D", NONNORMAL_CASES + NORMAL_CASES)
def test_stationary_covariance_matches_lyapunov(A, D):
    lam_min = jnp.min(jnp.linalg.eigvals(A).real)
    cov = thermox.conditional.covariance(60.0 / lam_min, A, D)
    assert relerr(cov, lyapunov_covariance(A, D)) < 1e-8


def reference_log_prob(ts, xs, A, b, D):
    """Sum of transition log-densities built from the Van Loan covariance.

    Independent of thermox and differentiable (jax.scipy.linalg.expm has a JVP).
    """

    def transition_logpdf(x1, x0, dt):
        mean = b + jax.scipy.linalg.expm(-A * dt) @ (x0 - b)
        cov = van_loan_covariance(A, D, dt)
        return jax.scipy.stats.multivariate_normal.logpdf(x1, mean, cov)

    return sum(
        transition_logpdf(xs[i], xs[i - 1], ts[i] - ts[i - 1])
        for i in range(1, len(ts))
    )


GRIDS = [
    pytest.param(
        jnp.array([0.0, 0.1, 0.5, 0.6, 1.4]), id="non-uniform"
    ),  # per-step path
    pytest.param(jnp.arange(0.0, 1.5, 0.1), id="uniform"),  # factor-once path
]


@pytest.mark.parametrize("ts", GRIDS)
@pytest.mark.parametrize("A,D", NONNORMAL_CASES)
def test_log_prob_matches_reference_gaussian(A, D, ts):
    d = A.shape[0]
    b = jnp.arange(1.0, d + 1.0)
    xs = jax.random.normal(jax.random.PRNGKey(3), (len(ts), d))
    ref = reference_log_prob(ts, xs, A, b, D)
    assert jnp.isclose(thermox.log_prob(ts, xs, A, b, D), ref, rtol=1e-8)


@pytest.mark.parametrize("ts", GRIDS)
@pytest.mark.parametrize("A,D", NONNORMAL_CASES)
def test_log_prob_grad_wrt_drift_matches_reference(A, D, ts):
    d = A.shape[0]
    b = jnp.arange(1.0, d + 1.0)
    xs = jax.random.normal(jax.random.PRNGKey(3), (len(ts), d))
    g = jax.grad(lambda A: thermox.log_prob(ts, xs, A, b, D))(A)
    g_ref = jax.grad(lambda A: reference_log_prob(ts, xs, A, b, D))(A)
    assert relerr(g, g_ref) < 1e-8


def test_log_prob_grad_symmetric_parametrization_matches_reference():
    # A = B B^T with D = I stays on the normal branch; gradients w.r.t. B are exact
    # there. (At an exactly normal A the derivative w.r.t. A itself in directions
    # that break normality is that of the symmetric-part formula -- see
    # thermox.sampler.transition_cov_eigh.)
    A, D = A_SYM, jnp.eye(3)
    B = jnp.linalg.cholesky(A)
    b = jnp.arange(1.0, 4.0)
    ts = jnp.array([0.0, 0.1, 0.5, 0.6, 1.4])
    xs = jax.random.normal(jax.random.PRNGKey(3), (len(ts), 3))
    g = jax.grad(lambda B: thermox.log_prob(ts, xs, B @ B.T, b, D))(B)
    g_ref = jax.grad(lambda B: reference_log_prob(ts, xs, B @ B.T, b, D))(B)
    assert relerr(g, g_ref) < 1e-8


@pytest.mark.parametrize("jitter", [0.0, 0.5], ids=["uniform-grid", "jittered-grid"])
def test_sample_covariance_matches_lyapunov_for_anisotropic_noise(jitter):
    A, D = A_SYM, D_DIAG
    ts = jnp.arange(0.0, 20000.0, 0.5)
    ts = jnp.sort(ts + jitter * jax.random.uniform(jax.random.PRNGKey(1), ts.shape))
    xs = thermox.sample(jax.random.PRNGKey(0), ts, jnp.zeros(3), A, jnp.zeros(3), D)
    emp = jnp.cov(xs[2000:].T)
    # Monte Carlo error of the sample covariance is ~1e-2 here; the symmetric-part
    # formula gives 0.13.
    assert relerr(emp, lyapunov_covariance(A, D)) < 0.05


@pytest.mark.parametrize("A,D", NORMAL_CASES)
def test_normal_case_equals_symmetric_part_formula(A, D):
    # For a normal transformed drift the symmetric-part formula is exact; the
    # general formula must reproduce it, so nothing changes for existing users.
    t = 0.7
    A_y, PD = thermox.preprocess(A, D)
    sym_eigvals, sym_eigvecs = jnp.linalg.eigh(0.5 * (A_y.val + A_y.val.T))
    old = sym_eigvecs @ jnp.diag(
        (1 - jnp.exp(-2 * sym_eigvals * t)) / (2 * sym_eigvals)
    )
    old = PD.sqrt @ old @ sym_eigvecs.T @ PD.sqrt.T
    assert relerr(thermox.conditional.covariance(t, A, D), old) < 1e-12


@pytest.mark.parametrize("A,D", NONNORMAL_CASES + NORMAL_CASES)
@pytest.mark.parametrize("t", [0.0, 0.7])
def test_transition_cov_eigh_factorizes_transition_cov(A, D, t):
    # (w, U) is the one object sample and log_prob read the covariance through;
    # both branches must return an orthonormal spectral factorization of
    # transition_cov, including at t = 0 (the linalg grids contain a zero step).
    from thermox.sampler import transition_cov, transition_cov_eigh

    A_y, _ = thermox.preprocess(A, D)
    cov = transition_cov(A_y, t)
    w, U = transition_cov_eigh(A_y, t)
    assert jnp.all(w > -1e-12)
    assert jnp.linalg.norm(U.T @ U - jnp.eye(len(w))) < 1e-10
    scale = max(1.0, float(jnp.linalg.norm(cov)))
    assert jnp.linalg.norm(U @ jnp.diag(w) @ U.T - cov) < 1e-10 * scale
    if bool(A_y.is_normal):
        # normal branch: precomputed eigenbasis of (A_y + A_y^T)/2, no per-step eigh
        assert jnp.array_equal(U, A_y.sym_eigvecs)
    # apply is evaluated inside the branch (so the normal branch never
    # materializes U per step)
    trace = transition_cov_eigh(A_y, t, lambda w, U: jnp.sum(w))
    assert jnp.isclose(trace, jnp.trace(cov), rtol=1e-10, atol=1e-14)


def test_preprocess_flags_normality():
    assert bool(preprocess_drift_matrix(A_SYM).is_normal)
    assert bool(preprocess_drift_matrix(A_ROT).is_normal)
    assert not bool(preprocess_drift_matrix(A_TRI).is_normal)
    # symmetric A becomes non-normal after transforming with anisotropic D
    A_y, _ = thermox.preprocess(A_SYM, D_DIAG)
    assert not bool(A_y.is_normal)


def ill_conditioned_eigenvectors_case():
    # d = 12, diag(1..3) plus 6 x a strictly upper triangular Gaussian: eigenvector
    # condition number ~1e9, where a formula in the eigenbasis of A loses everything.
    d = 12
    upper = jnp.triu(jax.random.normal(jax.random.PRNGKey(0), (d, d)), 1)
    return jnp.diag(jnp.linspace(1.0, 3.0, d)) + 6.0 * upper


@pytest.mark.parametrize("t", [0.01, 0.3])
def test_covariance_accurate_for_ill_conditioned_eigenvectors(t):
    A = ill_conditioned_eigenvectors_case()
    D = jnp.eye(A.shape[0])
    assert jnp.linalg.cond(jnp.linalg.eig(A)[1]) > 1e8
    cov = thermox.conditional.covariance(t, A, D)
    assert relerr(cov, van_loan_covariance(A, D, t)) < 1e-12


def test_covariance_is_zero_at_t_zero():
    A = ill_conditioned_eigenvectors_case()
    cov = thermox.conditional.covariance(0.0, A, jnp.eye(A.shape[0]))
    assert jnp.all(cov == 0.0)


def linalg_grid(burnin, num_samples=100, dt=0.1):
    # The grid thermox.linalg builds: x0 at time 0, one gap of burnin * dt, then dt.
    ts = jnp.arange(burnin, burnin + num_samples + 1) * dt
    return jnp.concatenate([jnp.array([0]), ts])


@pytest.mark.parametrize(
    "ts",
    [
        pytest.param(jnp.arange(0, 1, 0.01), id="readme-arange"),
        pytest.param(linalg_grid(0), id="linalg-burnin-0"),
        pytest.param(linalg_grid(1), id="linalg-burnin-1"),
        pytest.param(linalg_grid(5), id="linalg-burnin-5"),
        pytest.param((jnp.arange(0, 10001) * 0.1).astype(jnp.float32), id="float32"),
        pytest.param(jnp.linspace(0, 100, 300), id="linspace"),
    ],
)
def test_uniform_dt_accepts_grids_uniform_up_to_rounding(ts):
    is_uniform, dt = uniform_dt(ts)
    assert bool(is_uniform)
    assert jnp.isclose(dt, ts[2] - ts[1], rtol=1e-6)


def test_uniform_dt_rejects_jittered_grid():
    ts = jnp.arange(0, 100, 0.1)
    ts = jnp.sort(ts + jax.random.uniform(jax.random.PRNGKey(0), ts.shape) * 0.1)
    assert not bool(uniform_dt(ts)[0])


def contracting_matrix(key, d=4):
    return jax.scipy.linalg.expm(
        -(jax.random.normal(key, (d, d)) / d**0.5 + 3 * jnp.eye(d))
    )


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 1000, 1001])
def test_scan_linear_recurrence_matches_sequential_scan(n):
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(n), 3)
    E = contracting_matrix(k1)
    y0 = jax.random.normal(k2, (4,))
    u = jax.random.normal(k3, (n, 4))
    _, ys = jax.lax.scan(lambda y, u_k: (E @ y + u_k,) * 2, y0, u)
    assert jnp.allclose(_scan_linear_recurrence(E, y0, u), ys, rtol=1e-12, atol=1e-12)


def test_uniform_grid_engines_agree_for_ill_conditioned_eigenvectors():
    # On a uniform grid both engines apply one exp(-A dt); propagated through the
    # eigenbasis of A (per-step path) they disagree at ~1e-10 for this family.
    A = ill_conditioned_eigenvectors_case()
    d = A.shape[0]
    ts = jnp.arange(0.0, 1.0, 0.05)
    key = jax.random.PRNGKey(0)
    x0, b, D = jnp.ones(d), jnp.zeros(d), jnp.eye(d)
    xa = thermox.sample(key, ts, x0, A, b, D, associative_scan=True)
    xs = thermox.sample(key, ts, x0, A, b, D, associative_scan=False)
    assert relerr(xa, xs) < 1e-12


def test_log_prob_on_uniform_grid_exact_for_ill_conditioned_eigenvectors():
    A = ill_conditioned_eigenvectors_case()
    d = A.shape[0]
    ts = jnp.arange(0.0, 1.0, 0.05)
    x0, b, D = jnp.ones(d), jnp.zeros(d), jnp.eye(d)
    xs = thermox.sample(jax.random.PRNGKey(0), ts, x0, A, b, D)
    lp = thermox.log_prob(ts, xs, A, b, D)
    ref = reference_log_prob(ts, xs, A, b, D)
    assert jnp.abs(lp - ref) / jnp.abs(ref) < 1e-11


def test_linalg_expm_of_nonsymmetric_matrix():
    # expnegm's whitened drift is non-normal for a non-symmetric input: on
    # upstream main this estimate is off by 0.6; Monte Carlo noise is ~0.01.
    M = jnp.array([[-1.0, 3.0], [0.0, -2.0]])
    est = thermox.linalg.expm(M, num_samples=100000, dt=0.1, burnin=0, alpha=1.0)
    assert jnp.allclose(est, jax.scipy.linalg.expm(M), atol=1e-1)


# --- dyadic ladder: non-normal A on a non-uniform grid ---------------------------


def jittered_grid(n, dt=0.1, seed=1):
    ts = jnp.arange(0.0, n * dt, dt)
    return jnp.sort(
        ts + 0.5 * dt * jax.random.uniform(jax.random.PRNGKey(seed), ts.shape)
    )


def random_stable_drift(d, seed=2):
    # i.i.d. Gaussian / sqrt(d) plus 1.1 I: non-normal, all eigenvalues in the right half-plane
    return jax.random.normal(jax.random.PRNGKey(seed), (d, d)) / d**0.5 + 1.1 * jnp.eye(
        d
    )


@pytest.mark.parametrize("A,D", NONNORMAL_CASES)
def test_ladder_covariance_matches_reference(A, D):
    # The fold run on covariance matrices instead of draws reproduces Sigma(dt_k)
    # of the grid as given, for every step, from the ladder's own levels.
    from thermox.sampler import _ladder_lattice, transition_expm_and_cov

    ts = jittered_grid(301)
    dts = jnp.diff(ts)
    A_y, PD = thermox.preprocess(A, D)
    delta, n, M = _ladder_lattice(ts)
    assert jnp.max(jnp.abs(n * delta - dts)) <= 1e-12 * jnp.max(ts)
    d = A.shape[0]
    C = jnp.zeros((len(dts), d, d))
    for j in range(M + 1):
        E, cov = transition_expm_and_cov(A_y.val, delta * 2.0**j)
        bit = ((n >> j) & 1).astype(bool)
        C = jnp.where(
            bit[:, None, None], jnp.einsum("ij,tjk,lk->til", E, C, E) + cov, C
        )
    C = jnp.einsum("ij,tjk,lk->til", PD.sqrt, C, PD.sqrt)
    for k in range(len(dts)):
        assert relerr(C[k], van_loan_covariance(A, D, dts[k])) < 1e-10


@pytest.mark.parametrize("path", ["ladder", "per-step"])
def test_ladder_whitened_draws_are_standard_normal(path):
    # 20 000 irregular steps whitened with the reference factor of each step's
    # covariance: covariance I and mean 0 to Monte Carlo accuracy, on both paths.
    from thermox.sampler import (
        _sample_identity_diffusion_ladder,
        _sample_identity_diffusion_scan,
        expm_vp,
    )

    d, T = 8, 20_000
    A, D = random_stable_drift(d), jnp.eye(d)
    ts = jnp.sort(jax.random.uniform(jax.random.PRNGKey(3), (T + 1,))) * T * 0.5
    A_y, _ = thermox.preprocess(A, D)
    b, x0, key = jnp.zeros(d), jnp.zeros(d), jax.random.PRNGKey(0)
    if path == "ladder":
        ys = _sample_identity_diffusion_ladder(key, ts, x0, A_y, b, True)
    else:
        ys = _sample_identity_diffusion_scan(key, ts, x0, A_y, b)
    dts = jnp.diff(ts)
    means = jax.vmap(lambda y, dt: expm_vp(A_y, y, dt))(ys[:-1], dts)
    covs = jax.vmap(lambda dt: van_loan_covariance(A, D, dt))(dts)
    Ls = jnp.linalg.cholesky(covs)
    z = jax.vmap(lambda L, r: jax.scipy.linalg.solve_triangular(L, r, lower=True))(
        Ls, ys[1:] - means
    )
    assert jnp.max(jnp.abs(z.T @ z / T - jnp.eye(d))) < 4 * (2 / T) ** 0.5
    assert jnp.max(jnp.abs(jnp.mean(z, axis=0))) < 4 * (1 / T) ** 0.5


@pytest.mark.parametrize("dtype,tol", [(jnp.float64, 1e-12), (jnp.float32, 1e-5)])
def test_ladder_engines_agree(dtype, tol):
    from thermox.sampler import _sample_identity_diffusion_ladder

    A, D = A_TRI.astype(dtype), jnp.eye(3, dtype=dtype)
    ts = jittered_grid(200).astype(dtype)
    A_y, _ = thermox.preprocess(A, D)
    b, x0, key = jnp.ones(3, dtype), jnp.zeros(3, dtype), jax.random.PRNGKey(0)
    xa = _sample_identity_diffusion_ladder(key, ts, x0, A_y, b, True)
    xs = _sample_identity_diffusion_ladder(key, ts, x0, A_y, b, False)
    # Draws take the default float dtype, as in the other engines: float64 here
    # because the test module enables x64, float32 in thermox's default setting.
    assert relerr(xa, xs) < tol


def test_ladder_vmap_over_keys_matches_single_draws():
    A, D = A_TRI, jnp.eye(3)
    ts = jittered_grid(50)
    b, x0 = jnp.ones(3), jnp.zeros(3)
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    batched = jax.vmap(lambda k: thermox.sample(k, ts, x0, A, b, D))(keys)
    single = jnp.stack([thermox.sample(k, ts, x0, A, b, D) for k in keys])
    assert jnp.array_equal(batched, single)


def test_ladder_zero_gap_repeats_the_state():
    from thermox.sampler import _sample_identity_diffusion_ladder

    ts = jittered_grid(50)
    ts = jnp.concatenate([ts[:20], ts[19:20], ts[20:]])  # repeated time inside the grid
    A_y, _ = thermox.preprocess(A_TRI, jnp.eye(3))
    xs = _sample_identity_diffusion_ladder(
        jax.random.PRNGKey(0), ts, jnp.zeros(3), A_y, jnp.ones(3), True
    )
    assert jnp.all(jnp.isfinite(xs))
    assert jnp.allclose(
        xs[20], xs[19], rtol=1e-14, atol=1e-14
    )  # exp(0) x = x to rounding


@pytest.mark.parametrize("dtype,levels", [(jnp.float64, 53), (jnp.float32, 24)])
def test_ladder_operator_count_is_independent_of_T(dtype, levels):
    # One transition operator per mantissa bit of ts, built inside one scan; no
    # factorization per step (the structure does not change with T).
    from thermox.sampler import _ladder_noise

    A_y, _ = thermox.preprocess(A_TRI.astype(dtype), jnp.eye(3, dtype=dtype))
    texts = []
    for n in (100, 200):
        ts = jittered_grid(n).astype(dtype)
        texts.append(
            str(
                jax.make_jaxpr(lambda k: _ladder_noise(A_y, ts, k))(
                    jax.random.PRNGKey(0)
                )
            )
        )
    import re

    for text in texts:
        assert "eigh" not in text
        assert text.count(f"length={levels}") == 1  # the one scan over the levels
    # The loop structure (the level scan and expm's own squaring loop) is the same for both T.
    assert re.findall(r"length=\d+", texts[0]) == re.findall(r"length=\d+", texts[1])


def test_sample_dispatches_to_ladder_on_nonuniform_grid():
    # thermox.sample takes the ladder for non-normal A on a non-uniform grid.
    from thermox.sampler import _sample_identity_diffusion_ladder

    A, D = A_SYM, D_DIAG
    ts = jittered_grid(50)
    b, x0, key = jnp.ones(3), jnp.zeros(3), jax.random.PRNGKey(0)
    A_y, PD = thermox.preprocess(A, D)
    ys = _sample_identity_diffusion_ladder(
        key, ts, PD.sqrt_inv @ x0, A_y, PD.sqrt_inv @ b, True
    )
    direct = jax.vmap(jnp.matmul, in_axes=(None, 0))(PD.sqrt, ys)
    assert jnp.array_equal(thermox.sample(key, ts, x0, A, b, D), direct)


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_ladder_lattice_is_exact(dtype):
    # The unit is exactly a power of two; dyadic gaps have exactly their binary digits.
    from thermox.sampler import _ladder_lattice

    ts = jnp.array([0.0, 0.5, 0.75, 1.5], dtype)
    delta, n, M = _ladder_lattice(ts)
    assert delta.dtype == dtype and M == (52 if dtype == jnp.float64 else 23)
    assert float(delta) == 2.0 ** (0 - M)  # largest gap 0.75 = 0.75 * 2^0
    assert [int(v) for v in n] == [
        2 ** (M - 1),
        2 ** (M - 2),
        2 ** (M - 1) + 2 ** (M - 2),
    ]


# --- Chebyshev panels: log det Sigma(dt) on a non-uniform grid --------------------


def random_gaps_grid(n, decades=3.0, seed=4):
    # Gaps log-uniform over the given number of decades; every gap is different.
    gaps = 10 ** jax.random.uniform(
        jax.random.PRNGKey(seed), (n,), minval=-decades / 2, maxval=decades / 2
    )
    return jnp.concatenate([jnp.zeros(1), jnp.cumsum(gaps)]) * 0.1


def oscillatory_drift():
    # Eigenvalues 1 +- 10i and 2: |Im lambda| / Re lambda = 10.
    return jnp.array([[1.0, 10.0, 0.5], [-10.0, 1.0, 0.0], [0.0, 0.0, 2.0]])


def test_chebyshev_matrix_is_exact_on_chebyshev_polynomials():
    from thermox.prob import _chebyshev_matrix

    N = 8
    x_nodes = jnp.cos(jnp.pi * jnp.arange(N + 1) / N)
    for k in range(N + 1):
        # The transform of T_k sampled at the nodes is the unit vector e_k.
        coeffs = _chebyshev_matrix(N) @ jnp.cos(k * jnp.arccos(x_nodes))
        assert jnp.allclose(coeffs, jnp.eye(N + 1)[k], atol=1e-12)


def reference_terms(A_y, ys, b, dts):
    """-2 log p of every step from reference_covariance: r^T Sigma^-1 r + log det Sigma
    + d log(2 pi).
    """
    from thermox.sampler import expm_vp

    d = ys.shape[1]
    r = ys[1:] - b - jax.vmap(lambda y, dt: expm_vp(A_y, y - b, dt))(ys[:-1], dts)

    def term(rk, t):
        S = reference_covariance(A_y.val, jnp.eye(d), t)
        return rk @ jnp.linalg.solve(S, rk) + jnp.linalg.slogdet(S)[1]

    return jax.vmap(term)(r, dts) + d * jnp.log(2 * jnp.pi)


@pytest.mark.parametrize("A,D", NONNORMAL_CASES)
def test_panels_match_reference(A, D):
    # The interpolated value against the per-step reference, relative to the total
    # magnitude of the terms, to the accuracy of the covariances themselves.
    from thermox.prob import _log_prob_panels

    ts = random_gaps_grid(200)
    d = A.shape[0]
    b = jnp.arange(1.0, d + 1.0)
    xs = jax.random.normal(jax.random.PRNGKey(3), (len(ts), d))
    A_y, PD = thermox.preprocess(A, D)
    ys = jax.vmap(jnp.matmul, in_axes=(None, 0))(PD.sqrt_inv, xs)
    value, ok = _log_prob_panels(ts, ys, A_y, PD.sqrt_inv @ b)
    assert bool(ok)
    terms = reference_terms(A_y, ys, PD.sqrt_inv @ b, jnp.diff(ts))
    assert jnp.abs(value + 0.5 * jnp.sum(terms)) < 1e-10 * 0.5 * jnp.sum(jnp.abs(terms))


def test_panels_match_reference_in_float32():
    from thermox.prob import _log_prob_panels

    A, D = A_TRI.astype(jnp.float32), jnp.eye(3, dtype=jnp.float32)
    ts = random_gaps_grid(200).astype(jnp.float32)
    A_y, _ = thermox.preprocess(A, D)
    ys = jax.random.normal(jax.random.PRNGKey(3), (len(ts), 3), jnp.float32)
    value, ok = _log_prob_panels(ts, ys, A_y, jnp.ones(3, jnp.float32))
    assert bool(ok)
    terms = reference_terms(
        preprocess_drift_matrix(A_TRI), ys, jnp.ones(3), jnp.diff(ts)
    )
    # float32 accuracy is that of transition_expm_and_cov in float32 (twelve
    # doublings, about 1e-4).
    assert jnp.abs(value + 0.5 * jnp.sum(terms)) < 3e-4 * 0.5 * jnp.sum(jnp.abs(terms))


def test_panels_grad_wrt_drift_matches_reference():
    from thermox.prob import _log_prob_panels

    A, D = A_TRI, jnp.eye(3)
    ts = random_gaps_grid(60)
    ys = jax.random.normal(jax.random.PRNGKey(3), (len(ts), 3))
    b = jnp.ones(3)

    def value(A):
        return _log_prob_panels(ts, ys, preprocess_drift_matrix(A), b)[0]

    g = jax.grad(value)(A)
    g_ref = jax.grad(lambda A: reference_log_prob(ts, ys, A, b, D))(A)
    assert relerr(g, g_ref) < 1e-8


def test_panels_grad_wrt_ts_matches_reference():
    # The interpolant is a polynomial in the gap, so the gradient with respect to
    # the time stamps is finite and equal to the per-step path's.
    from thermox.prob import _log_prob_identity_diffusion_stepwise, _log_prob_panels

    ts = random_gaps_grid(60)
    A_y = preprocess_drift_matrix(A_TRI)
    ys = jax.random.normal(jax.random.PRNGKey(3), (len(ts), 3))
    g = jax.grad(lambda t: _log_prob_panels(t, ys, A_y, jnp.ones(3))[0])(ts)
    g_ref = jax.grad(
        lambda t: _log_prob_identity_diffusion_stepwise(t, ys, A_y, jnp.ones(3))
    )(ts)
    assert jnp.all(jnp.isfinite(g))
    assert relerr(g, g_ref) < 1e-8


def test_panels_decline_oscillatory_drift():
    from thermox.prob import _log_prob_panels

    A_y = preprocess_drift_matrix(oscillatory_drift())
    ts = random_gaps_grid(200)
    ys = jax.random.normal(jax.random.PRNGKey(3), (len(ts), 3))
    assert not bool(_log_prob_panels(ts, ys, A_y, jnp.zeros(3))[1])


def test_panels_decline_zero_gap():
    from thermox.prob import _log_prob_panels

    ts = random_gaps_grid(50)
    ts = jnp.concatenate([ts[:20], ts[19:20], ts[20:]])  # repeated time inside the grid
    A_y = preprocess_drift_matrix(A_TRI)
    ys = jax.random.normal(jax.random.PRNGKey(3), (len(ts), 3))
    assert not bool(_log_prob_panels(ts, ys, A_y, jnp.zeros(3))[1])


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_panels_operator_count_is_independent_of_T(dtype):
    # 17 eigh per used panel, inside one scan over the panels, for T = 100 and
    # T = 200 alike.
    from thermox.prob import _log_prob_panels

    import re

    A_y, _ = thermox.preprocess(A_TRI.astype(dtype), jnp.eye(3, dtype=dtype))
    texts = []
    for n in (100, 200):
        ts = random_gaps_grid(n).astype(dtype)
        ys = jax.random.normal(jax.random.PRNGKey(3), (len(ts), 3), dtype)
        texts.append(
            str(
                jax.make_jaxpr(
                    lambda ts, ys: _log_prob_panels(ts, ys, A_y, jnp.zeros(3, dtype))
                )(ts, ys)
            )
        )
    for text in texts:
        assert (
            text.count("= eigh[") == 1
        )  # one call site, batched over the nodes, inside the panel scan
        assert "[17,3,3]" in text  # the batch of N + 1 node covariances
    assert re.findall(r"length=\d+", texts[0]) == re.findall(r"length=\d+", texts[1])


def test_log_prob_dispatches_to_panels_on_nonuniform_grid():
    # thermox.log_prob takes the panels on a non-uniform grid with non-normal A, and
    # their run-time check passes there.
    from thermox.prob import _log_prob_panels

    A, D = A_SYM, D_DIAG
    ts = jnp.array([0.0, 0.1, 0.5, 0.6, 1.4])
    b = jnp.arange(1.0, 4.0)
    xs = jax.random.normal(jax.random.PRNGKey(3), (len(ts), 3))
    A_y, PD = thermox.preprocess(A, D)
    ys = jax.vmap(jnp.matmul, in_axes=(None, 0))(PD.sqrt_inv, xs)
    value, ok = _log_prob_panels(ts, ys, A_y, PD.sqrt_inv @ b)
    assert bool(ok)
    expected = value + jnp.log(jnp.linalg.det(PD.sqrt_inv)) * (len(ts) - 1)
    assert jnp.array_equal(thermox.log_prob(ts, xs, A, b, D), expected)


def test_log_prob_falls_back_to_per_step_path_when_check_fails():
    # When the check fails, log_prob returns the per-step path's value, bitwise.
    from thermox.prob import _log_prob_identity_diffusion_stepwise, _log_prob_panels

    A = oscillatory_drift()
    ts = random_gaps_grid(60)
    xs = jax.random.normal(jax.random.PRNGKey(3), (len(ts), 3))
    A_y = preprocess_drift_matrix(A)
    assert not bool(_log_prob_panels(ts, xs, A_y, jnp.zeros(3))[1])
    lp = thermox.log_prob(ts, xs, A, jnp.zeros(3), jnp.eye(3))
    assert jnp.array_equal(
        lp, _log_prob_identity_diffusion_stepwise(ts, xs, A_y, jnp.zeros(3))
    )


def test_log_prob_operator_count_is_independent_of_T():
    # log_prob's program on a non-uniform grid carries the batch of 17 node
    # covariances, and its eigh call sites do not multiply with T (the per-step
    # fallback branch keeps its own T-length loop, as expected).
    A, D = A_TRI, jnp.eye(3)
    texts = []
    for n in (100, 200):
        ts = random_gaps_grid(n)
        xs = jax.random.normal(jax.random.PRNGKey(3), (len(ts), 3))
        texts.append(
            str(
                jax.make_jaxpr(
                    lambda ts, xs: thermox.log_prob(ts, xs, A, jnp.zeros(3), D)
                )(ts, xs)
            )
        )
    assert "[17,3,3]" in texts[0]
    assert texts[0].count("= eigh[") == texts[1].count("= eigh[")
