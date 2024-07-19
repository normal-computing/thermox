from functools import partial
from jax import random, jit, config, numpy as jnp
from jax.scipy.linalg import expm
from jax.lax import scan
import numpy as np
import argparse
from tqdm import tqdm
import thermox
import pickle

from examples.matrix_exponentials import matrix_generation

# Set the precision of the computation
config.update("jax_enable_x64", True)

np.random.seed(42)


# Load n_repeats, matrix_type and alpha from the command line
parser = argparse.ArgumentParser()
parser.add_argument("--n_repeats", type=int, default=1)
parser.add_argument("--matrix_type", type=str, default="wishart")
parser.add_argument("--alpha", type=float, default=0.0)
args = parser.parse_args()


sample = jit(thermox.sample)
get_matrix = getattr(matrix_generation, args.matrix_type)


NT = 10000
dt = 12
ts = jnp.arange(NT) * dt
N_burn = 0
keys = random.split(random.PRNGKey(42), args.n_repeats)


gamma = 1
beta = 1
alpha = args.alpha


@jit
def samps_to_autocovs_errs(samps, true_exp):
    def body_func(prev_mat, n):
        new_mat = prev_mat * n / (n + 1) + jnp.outer(samps[n], samps[n - 1]) / (n + 1)
        err = jnp.linalg.norm(new_mat * jnp.exp(alpha) - true_exp)
        return new_mat, err

    return scan(
        body_func,
        jnp.zeros((samps.shape[1], samps.shape[1])),
        jnp.arange(1, samps.shape[0]),
    )[1]


D = [64, 128, 256, 512]
ERR_abs = np.zeros((args.n_repeats, len(D), NT))
ERR_rel = np.zeros_like(ERR_abs)

for repeat in tqdm(range(args.n_repeats)):
    key = keys[repeat]
    for i in range(len(D)):
        d = D[i]
        print(f"Repeat {repeat}/{args.n_repeats}, \t D = {d}")

        A = get_matrix(d, key)

        exact_exp_min_A = expm(-A)

        A_shifted = (A + alpha * jnp.eye(A.shape[0])) / dt
        B = A_shifted + A_shifted.T

        A_shifted_lambda_min = jnp.min(jnp.linalg.eig(A_shifted / gamma)[0].real)
        print("A Eig min: ", A_shifted_lambda_min)

        D_lambda_min = jnp.min(jnp.linalg.eig(B / (gamma * beta))[0].real)
        print("D Eig min: ", D_lambda_min)

        x0 = np.zeros(d)
        X = sample(
            key,
            ts,
            x0,
            A_shifted / gamma,
            np.zeros(d),
            B / (gamma * beta),
        )

        err_abs = samps_to_autocovs_errs(X, exact_exp_min_A)

        ERR_abs[repeat, i, 1:] = err_abs
        ERR_rel[repeat, i, 1:] = err_abs / jnp.linalg.norm(exact_exp_min_A)

    with open(
        f"examples/matrix_exponentials/results_{args.matrix_type}.pkl", "wb"
    ) as f:
        pickle.dump(
            {
                "D": D,
                "dt": dt,
                "alpha": alpha,
                "ERR_abs": ERR_abs,
                "ERR_rel": ERR_rel,
            },
            f,
        )
