from thermox.sampler import collect_samples, collect_samples_diffusion_matrix
import jax
import jax.numpy as jnp
import timeit

key = jax.random.PRNGKey(0)
dimension = 1000
x0 = jnp.zeros(dimension)
A = jax.random.normal(key, (dimension, dimension))
A = A @ A.T
b = jax.random.normal(key, (dimension,))
num_samples = 10000

timeit.timeit('collect_samples(key, x0, A, b, num_samples)')

timeit.timeit('collect_samples_diffusion_matrix(key, x0, A, b, num_samples)')