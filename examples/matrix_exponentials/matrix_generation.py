from jax import random, numpy as jnp
from scipy.stats import ortho_group


def wishart(d: int, key: random.PRNGKey) -> jnp.ndarray:
    n = 2 * d  # degrees of freedom
    G = random.normal(key, shape=(d, n))
    A_wishart = (G @ G.T) / n
    return A_wishart


def orthogonal(d: int, _) -> jnp.ndarray:
    return ortho_group.rvs(d)
