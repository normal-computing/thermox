import jax
from jax import numpy as jnp
import optax

import thermox


def test_log_prob_numeric_identity_diffusion():
    A_true = jnp.array([[3, 2, 1], [2, 4.0, 2], [1, 2, 5.0]])
    b_true = jnp.array([1, 2.0, 3])
    D_true = jnp.eye(3)

    nts = 1000
    ts = jnp.sort(jax.random.uniform(jax.random.PRNGKey(0), (nts,)) * 1000.0)
    x0 = jax.random.normal(jax.random.PRNGKey(0), b_true.shape)

    rk = jax.random.PRNGKey(0)
    samps = thermox.sample(rk, ts, x0, A_true, b_true, D_true)

    def transition_mean(x0, dt):
        return b_true + jax.scipy.linalg.expm(-A_true * dt) @ (x0 - b_true)

    @jax.jit
    def transition_cov_integrand(s, t):
        exp_A_st = jax.scipy.linalg.expm(A_true * (s - t))
        exp_A_T_st = jax.scipy.linalg.expm(A_true.T * (s - t))
        return exp_A_st @ D_true @ exp_A_T_st

    def transition_cov_numeric(dt):
        s_linsp = jnp.linspace(0, dt, 1000)
        evals = jax.vmap(lambda s: transition_cov_integrand(s, dt))(s_linsp)
        return jax.scipy.integrate.trapezoid(evals, s_linsp, axis=0)

    def transition_log_prob_numeric(xt, x0, t):
        mean = transition_mean(x0, t)
        cov = transition_cov_numeric(t)
        return jax.scipy.stats.multivariate_normal.logpdf(xt, mean, cov)

    log_probs_numeric = jax.vmap(
        lambda xtp1, xt, dt: transition_log_prob_numeric(xtp1, xt, dt)
    )(samps[1:], samps[:-1], ts[1:] - ts[:-1])

    log_prob = thermox.log_prob(ts, samps, A_true, b_true, D_true)

    assert jnp.isclose(log_prob, log_probs_numeric.sum(), rtol=1e-2)


def test_log_prob_numeric():
    A_true = jnp.array([[3, 2, 1], [2, 4.0, 2], [1, 2, 5.0]])
    b_true = jnp.array([1, 2.0, 3])
    D_true = jnp.array([[1, 0.3, -0.1], [0.3, 1, 0.2], [-0.1, 0.2, 1.0]])

    nts = 1000
    ts = jnp.sort(jax.random.uniform(jax.random.PRNGKey(0), (nts,)) * 1000.0)
    x0 = jax.random.normal(jax.random.PRNGKey(0), b_true.shape)

    rk = jax.random.PRNGKey(0)
    samps = thermox.sample(rk, ts, x0, A_true, b_true, D_true)

    def transition_mean(x0, dt):
        return b_true + jax.scipy.linalg.expm(-A_true * dt) @ (x0 - b_true)

    @jax.jit
    def transition_cov_integrand(s, t):
        exp_A_st = jax.scipy.linalg.expm(A_true * (s - t))
        exp_A_T_st = jax.scipy.linalg.expm(A_true.T * (s - t))
        return exp_A_st @ D_true @ exp_A_T_st

    def transition_cov_numeric(dt):
        s_linsp = jnp.linspace(0, dt, 1000)
        evals = jax.vmap(lambda s: transition_cov_integrand(s, dt))(s_linsp)
        return jax.scipy.integrate.trapezoid(evals, s_linsp, axis=0)

    def transition_log_prob_numeric(xt, x0, t):
        mean = transition_mean(x0, t)
        cov = transition_cov_numeric(t)
        return jax.scipy.stats.multivariate_normal.logpdf(xt, mean, cov)

    log_probs_numeric = jax.vmap(
        lambda xtp1, xt, dt: transition_log_prob_numeric(xtp1, xt, dt)
    )(samps[1:], samps[:-1], ts[1:] - ts[:-1])

    log_prob = thermox.log_prob(ts, samps, A_true, b_true, D_true)

    assert jnp.isclose(log_prob, log_probs_numeric.sum(), rtol=1e-2)


def test_MLE():
    A_true = jnp.array([[3, 2, 1], [2, 4.0, 2], [1, 2, 5.0]])
    b_true = jnp.array([1, 2.0, 3])
    D_true = jnp.array([[1, 0.3, -0.1], [0.3, 1, 0.2], [-0.1, 0.2, 1.0]])

    nts = 300
    ts = jnp.linspace(0, 100, nts)
    x0 = jnp.zeros_like(b_true)

    n_trajecs = 5
    rks = jax.random.split(jax.random.PRNGKey(0), n_trajecs)

    samps = jax.vmap(lambda key: thermox.sample(key, ts, x0, A_true, b_true, D_true))(
        rks
    )

    A_init = jnp.eye(3) + jax.random.normal(rks[0], (3, 3)) * 1e-1
    b_init = jnp.zeros(3)
    D_sqrt_init = jnp.eye(3)

    log_prob_true = thermox.log_prob(ts, samps[0], A_true, b_true, D_true)
    log_prob_init = thermox.log_prob(
        ts, samps[0], A_init, b_init, D_sqrt_init @ D_sqrt_init.T
    )

    assert log_prob_true > log_prob_init

    # Gradient descent
    def loss(params):
        A, b, D_sqrt = params
        D_sqrt = jnp.tril(D_sqrt)
        D = D_sqrt @ D_sqrt.T
        return -jax.vmap(lambda s: thermox.log_prob(ts, s, A, b, D))(
            samps
        ).mean() / len(ts)

    val_and_g = jax.jit(jax.value_and_grad(loss))

    ps = (A_init, b_init, D_sqrt_init)
    ps_true = (A_true, b_true, jnp.linalg.cholesky(D_true))

    v, g = val_and_g(ps)
    v_true, g_true = val_and_g(ps_true)

    assert v_true < v
    for i in range(len(ps)):
        assert jnp.all(jnp.abs(g_true[i]) <= jnp.abs(g[i]) * 1.5)

    n_steps = 20000
    neg_log_probs = jnp.zeros(n_steps)

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(ps)

    for i in range(n_steps):
        neg_log_prob, grad = val_and_g(ps)
        if jnp.isnan(neg_log_prob) or any([jnp.isnan(g).any() for g in grad]):
            break
        updates, opt_state = optimizer.update(grad, opt_state)
        ps = optax.apply_updates(ps, updates)
        neg_log_probs = neg_log_probs.at[i].set(neg_log_prob)

    A_recover = ps[0]
    b_recover = ps[1]
    D_recover = ps[2] @ ps[2].T

    assert jnp.allclose(A_recover, A_true, atol=1e0)
    assert jnp.allclose(b_recover, b_true, atol=1e0)
    assert jnp.allclose(D_recover, D_true, atol=1e0)
