import jax
from jax import numpy as jnp

import thermox


def test_log_prob_numeric_identity_diffusion():
    A_true = jnp.array([[3, 2, 1], [2, 4.0, 2], [1, 2, 5.0]])
    b_true = jnp.array([1, 2.0, 3])
    D_true = jnp.eye(3)

    nts = 1000
    ts = jnp.sort(jax.random.uniform(jax.random.PRNGKey(0), (nts,)) * 1000.0)
    x0 = jax.random.normal(jax.random.PRNGKey(0), b_true.shape)

    rk = jax.random.PRNGKey(0)
    samps = thermox.collect_samples(rk, ts, x0, A_true, b_true, D_true)

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
    samps = thermox.collect_samples(rk, ts, x0, A_true, b_true, D_true)

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
    ts = jnp.sort(jax.random.uniform(jax.random.PRNGKey(0), (nts,)) * 10.0)
    x0 = jnp.zeros_like(b_true)

    n_trajecs = 3
    rk = jax.random.PRNGKey(0)
    rks = jax.random.split(rk, n_trajecs)

    samps = jax.vmap(
        lambda key: thermox.collect_samples(key, ts, x0, A_true, b_true, D_true)
    )(rks)

    rk_inits = jax.random.split(rk, 3)
    A_sqrt_init = jax.random.normal(rk_inits[0], (3, 3))
    A_sqrt_init = jnp.tril(A_sqrt_init)
    b_init = jax.random.normal(rk_inits[1], (3,))
    D_sqrt_init = jax.random.normal(rk_inits[2], (3, 3))
    D_sqrt_init = jnp.tril(D_sqrt_init)

    log_prob_true = thermox.log_prob(ts, samps[0], A_true, b_true, D_true)
    log_prob_init = thermox.log_prob(
        ts, samps[0], A_sqrt_init @ A_sqrt_init.T, b_init, D_sqrt_init @ D_sqrt_init.T
    )

    assert log_prob_true > log_prob_init

    # JAX weird nans when using jax.linalg.eigh on identity
    # def f(m):
    #     return jnp.linalg.eigh(m)[1].sum()

    # id_mat = jnp.eye(3)
    # print(jax.value_and_grad(f)(id_mat))
    # print(jax.value_and_grad(f)(id_mat + 1e-7))

    # Gradient descent
    def loss(params):
        A_sqrt, b, D_sqrt = params
        A_sqrt = jnp.tril(A_sqrt)
        D_sqrt = jnp.tril(D_sqrt)
        A = A_sqrt @ A_sqrt.T
        D = D_sqrt @ D_sqrt.T
        return -jax.vmap(lambda s: thermox.log_prob(ts, s, A, b, D, A_spd=True))(
            samps
        ).mean() / len(ts)

    val_and_g = jax.jit(jax.value_and_grad(loss))

    ps = (A_sqrt_init, b_init, D_sqrt_init)
    ps_true = (jnp.linalg.cholesky(A_true), b_true, jnp.linalg.cholesky(D_true))

    v, g = val_and_g(ps)
    v_true, g_true = val_and_g(ps_true)

    assert v_true < v
    for i in range(len(ps)):
        assert jnp.all(jnp.abs(g_true[i]) <= jnp.abs(g[i]))

    n_steps = 5000
    neg_log_probs = jnp.zeros(n_steps)

    # lr = 1e-4

    # for i in range(n_steps):
    #     neg_log_prob, grad = val_and_g(ps)
    #     ps = [p - lr * g for p, g in zip(ps, grad)]
    #     neg_log_probs = neg_log_probs.at[i].set(neg_log_prob)

    import optax

    optimizer = optax.adam(1e-1)
    opt_state = optimizer.init(ps)

    for i in range(n_steps):
        neg_log_prob, grad = val_and_g(ps)
        updates, opt_state = optimizer.update(grad, opt_state)
        ps = optax.apply_updates(ps, updates)
        neg_log_probs = neg_log_probs.at[i].set(neg_log_prob)

    print(neg_log_probs[-1])
    print(v_true)

    import matplotlib.pyplot as plt

    plt.plot(neg_log_probs)
    plt.show()

    A_recover = ps[0] @ ps[0].T
    b_recover = ps[1]
    D_recover = ps[2] @ ps[2].T

    print(b_recover)

    samps2 = thermox.collect_samples(rk, ts, x0, A_true, b_recover, D_true)

    plt.figure()
    plt.plot(samps[0])
    plt.figure()
    plt.plot(samps2)
    plt.show()

    assert jnp.allclose(A_recover, A_true, rtol=1e-2)
    assert jnp.allclose(b_recover, b_true, rtol=1e-2)
    assert jnp.allclose(D_recover, D_true, rtol=1e-2)
