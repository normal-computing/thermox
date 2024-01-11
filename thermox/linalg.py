from sampler import collect_samples

def solve(
    A,
    b,
    num_samples: int=10000,
    dt: float=0.1, 
    gamma: float=1., 
    k0: float=1., 
    burnin: int=0,
    seed: int=0,
) -> jax.Array:
    """
    Obtain the solution of the linear system

    Ax = b

    by collecting samples from an Ornstein-Uhlenbeck
    process and calculating the mean over the samples.

    Args:
        - A: drift matrix.
        - b: mean displacement vector.
        - num_samples: float, number of samples to be collected.
        - dt: float, time step.
        - gamma: float, damping constant.
        - k0: float, noise variance.
        - burnin: burn-in, time before which samples are not collected.
        - seed: random seed
    
    Returns:
        - x: approximate solution of the linear system.
    """
    key = jax.random.PRNGKey(0)
    samples = collect_samples()

