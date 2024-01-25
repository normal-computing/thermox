import jax
import jax.numpy as jnp
from jax.lax import fori_loop, scan
from jax import Array


def collect_samples_identity_diffusion(
    key: Array,
    ts: Array,
    x0: Array,
    A: Array,
    b: Array,
) -> Array:
    """
    Collects samples from the Ornstein-Uhlenbeck process, defined as:

    dx = - A * (x - b) dt + dW

    by using exact diagonalization.

    Args:
        - key: jax PRNGKey.
        - ts: array-like, times at which samples are collected. Includes time for x0.
        - x0: initial state of the process.
        - A: drift matrix.
        - b: drift displacement vector.

    Returns:
        - samples: array-like, desired samples.
            shape: (len(ts), ) + x0.shape
    """

    eigvals, eigvecs = jnp.linalg.eig(A)
    eigvecs_inv = jnp.linalg.inv(eigvecs)

    def expm_vp(v, dt):
        out = eigvecs_inv @ v
        out = jnp.exp(-eigvals * dt) * out
        out = eigvecs @ out
        return out.real

    def transition_mean(x, dt):
        return b + expm_vp(x - b, dt)

    symA = 0.5 * (A + A.T)
    symA_eigvals, symA_eigvecs = jnp.linalg.eig(symA)

    def transition_cov_sqrt_vp(v, dt):
        diag = ((1 - jnp.exp(-2 * symA_eigvals * dt)) / (2 * symA_eigvals)) ** 0.5
        out = diag * v
        out = symA_eigvecs @ out
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


def collect_samples(
    key: Array, ts: Array, x0: Array, A: Array, b: Array, D: Array
) -> Array:
    """
    Collects samples from the Ornstein-Uhlenbeck process, defined as:

    dx = - A * (x - b) dt + sqrt(D) dW

    by using exact diagonalization.

    Args:
        - key: jax PRNGKey.
        - ts: array-like, times at which samples are collected. Includes time for x0.
        - x0: initial state of the process.
        - A: drift matrix.
        - b: drift displacement vector.
        - D: diffusion matrix.

    Returns:
        - samples: array-like, desired samples.
            shape: (len(ts), ) + x0.shape
    """
    D_sqrt = jnp.linalg.cholesky(D)
    D_sqrt_inv = jnp.linalg.inv(D_sqrt)
    y0 = D_sqrt_inv @ x0
    A_y = D_sqrt_inv @ A @ D_sqrt
    b_y = D_sqrt_inv @ b
    ys = collect_samples_identity_diffusion(key, ts, y0, A_y, b_y)
    return jax.vmap(jnp.matmul, in_axes=(None, 0))(D_sqrt, ys)


# def collect_samples_full_diffusion_matrix(
#     key,
#     A,
#     b,
#     D,
#     num_samples,
#     dt,
#     x0: jax.Array = None,
#     burnin=0,
#     solver_steps_per_dt=10,
# ):
#     """
#     Collects samples from an overdamped Langevin (ODL)
#     process via an Euler discretisation. The times are
#     in units of the time constant RC.
#     dx = - A(x - b) dt + sqrt(D) dW
#     Args:
#         - key: jax PRNGKey.
#         - x0: vector, initial state of the Langevin process.
#         - A: array-like, matrix to be inverted or on the right hand side
#         of the linear system to be solved.
#         - b: vector, on the left hand side of the linear system,
#         corresponding to a DC voltage bias.
#         - D: diffusion matrix
#         - num_samples: float, number of samples to be collected.
#         - dt: float, time step.
#         - burnin: int, time before which samples are not collected.
#     Returns:
#         - samples: array-like, desired samples.
#     """
#     if x0 is None:
#         x0 = jnp.zeros_like(b)

#     D_sqrt = jnp.linalg.cholesky(D)

#     def next_x(x_in, t_diff, tkey, steps_per_dt):
#         ts = jnp.linspace(0, t_diff, steps_per_dt + 1)
#         keys = jax.random.split(tkey, steps_per_dt)

#         def body_fun(i, x):
#             stepsize = ts[i + 1] - ts[i]
#             return (
#                 x
#                 - stepsize * (A @ (x - b))
#                 + jnp.sqrt(stepsize)
#                 * D_sqrt
#                 @ jax.random.normal(keys[i], shape=x.shape)
#             )

#         return fori_loop(0, steps_per_dt, body_fun, x_in)

#     def scan_body(x_and_key, i):
#         x, rk = x_and_key
#         rk, rk_use = jax.random.split(rk)
#         x = next_x(x, dt, rk_use, solver_steps_per_dt)
#         return (x, rk), x

#     xs = scan(scan_body, (x0, key), jnp.arange(num_samples + burnin))[1]

#     return xs[burnin:]


# # Test
# key = jax.random.PRNGKey(0)
# dt = 0.01
# ts = jnp.arange(0, 1000, dt)
# A = jnp.array([[4, 1], [1, 2]], dtype=jnp.float64)
# # b = jnp.array([1, 2], dtype=jnp.float64)
# b = jnp.zeros(2, dtype=jnp.float64)
# # D = jnp.array([[1, 0], [0, 2]], dtype=jnp.float64)
# D = jnp.eye(2, dtype=jnp.float64)

# # x0 = jnp.array([1, 3], dtype=jnp.float64)
# x0 = jnp.zeros(2, dtype=jnp.float64)

# samps1 = collect_samples(key, ts, x0, A, b, D)
# samps2 = collect_samples_full_diffusion_matrix(
#     key, A, b, D, len(ts), dt, x0=x0, burnin=0, solver_steps_per_dt=1000
# )

# import matplotlib.pyplot as plt

# plt.ion()

# plt.figure()
# plt.plot(ts, samps1[:, 0], label="samps1")
# plt.plot(ts, samps2[:, 0], label="samps2")
# plt.legend()

# plt.figure()
# plt.plot(ts, samps1[:, 1], label="samps1")
# plt.plot(ts, samps2[:, 1], label="samps2")
# plt.legend()


# cov1 = jnp.cov(samps1.T)
# cov2 = jnp.cov(samps2.T)


# expected_cov = jnp.linalg.inv(A) @ (D / 2)


# mean1 = jnp.mean(samps1, axis=0)
# mean2 = jnp.mean(samps2, axis=0)

# expected_mean = b
