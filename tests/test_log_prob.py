import jax
from jax import numpy as jnp

import thermox


def test_log_prob_numeric():
    A = jnp.array([[3, 2, 1], [2, 4.0, 2], [1, 2, 5.0]])
    b = jnp.array([1, 2.0, 3])
    D = jnp.array([[1, 0.3, -0.1], [0.3, 1, 0.2], [-0.1, 0.2, 1.0]])

    # ts = jnp.arange(0, 100, 1.0)
    ts = jnp.array([0, 1.0])
    x0 = jnp.zeros_like(b)

    rk = jax.random.PRNGKey(0)
    samps = thermox.collect_samples(rk, ts, x0, A, b, D)

    def transition_mean(x0, dt):
        return b + jax.scipy.linalg.expm(-A * dt) @ (x0 - b)

    @jax.jit
    def transition_cov_integrand(s, t):
        exp_A_st = jax.scipy.linalg.expm(A * (s - t))
        exp_A_T_st = jax.scipy.linalg.expm(A.T * (s - t))
        return jnp.dot(exp_A_st, jnp.dot(D, exp_A_T_st))

    def transition_cov_numeric(dt):
        s_linsp = jnp.linspace(0, dt, 10000)
        evals = jax.vmap(lambda s: transition_cov_integrand(s, dt))(s_linsp)
        return jax.scipy.integrate.trapezoid(evals, s_linsp, axis=0)

    def transition_log_prob_numeric(xt, x0, t):
        mean = transition_mean(x0, t)
        cov = transition_cov_numeric(t)
        return jax.scipy.stats.multivariate_normal.logpdf(xt, mean, cov)

    log_probs_numeric = jax.vmap(
        lambda xtm1, xt, dt: transition_log_prob_numeric(xtm1, xt, dt)
    )(samps[:-1], samps[1:], ts[1:] - ts[:-1])

    log_prob = thermox.log_prob(ts, samps, A, b, D, A_spd=True)

    assert jnp.isclose(log_prob, log_probs_numeric.sum(), rtol=1e-2)


def test_MLE():
    A_true = jnp.array([[3, 2, 1], [2, 4.0, 2], [1, 2, 5.0]])
    b_true = jnp.array([1, 2.0, 3])
    D_true = jnp.array([[1, 0.3, -0.1], [0.3, 1, 0.2], [-0.1, 0.2, 1.0]])

    ts = jnp.arange(0, 1000, 1.0)
    x0 = jnp.zeros_like(b_true)

    rk = jax.random.PRNGKey(0)
    samps = thermox.collect_samples(rk, ts, x0, A_true, b_true, D_true)

    A_sqrt_init = (
        jnp.eye(3) + 1e-4
    )  # jax throws weird nans if this is exactly the identity
    b_init = jnp.zeros(3)
    D_sqrt_init = jnp.eye(3)

    log_prob_true = thermox.log_prob(ts, samps, A_true, b_true, D_true)
    log_prob_init = thermox.log_prob(
        ts, samps, A_sqrt_init @ A_sqrt_init.T, b_init, D_sqrt_init @ D_sqrt_init.T
    )

    assert log_prob_true > log_prob_init

    # def f(m):
    #     return jnp.linalg.eigh(m)[1].sum()

    # id_mat = jnp.eye(3)
    # print(jax.value_and_grad(f)(id_mat))
    # print(jax.value_and_grad(f)(id_mat + 1e-7))

    # Gradient descent
    def loss(params):
        A_sqrt, b, D_sqrt = params
        A = A_sqrt @ A_sqrt.T
        D = D_sqrt @ D_sqrt.T
        return -thermox.log_prob(ts, samps, A, b, D, A_spd=True)

    ps = (A_sqrt_init, b_init, D_sqrt_init)
    n_steps = 50
    lr = 1e-5
    neg_log_probs = jnp.zeros(n_steps)

    val_and_g = jax.jit(jax.value_and_grad(loss))

    for i in range(n_steps):
        neg_log_prob, grad = val_and_g(ps)
        ps = [p - lr * g for p, g in zip(ps, grad)]
        neg_log_probs = neg_log_probs.at[i].set(neg_log_prob)

    # import matplotlib.pyplot as plt

    # plt.plot(neg_log_probs[:80])
    # plt.show()
