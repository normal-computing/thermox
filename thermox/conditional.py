from jax import numpy as jnp
from jax import Array

from thermox.utils import (
    ProcessedDriftMatrix,
    ProcessedDiffusionMatrix,
    handle_matrix_inputs,
)
from thermox.sampler import expm_vp


def mean(
    t: float,
    x0: Array,
    A: Array | ProcessedDriftMatrix,
    b: Array,
    D: Array | ProcessedDiffusionMatrix,
) -> Array:
    """Computes the mean of p(x_t | x_0)

    For x_t evolving according to the SDE:

    dx = - A * (x - b) dt + sqrt(D) dW

    Args:
        ts: Times at which samples are collected. Includes time for x0.
        x0: Initial state of the process.
        A: Drift matrix (Array or thermox.ProcessedDriftMatrix).
            Note: If a thermox.ProcessedDriftMatrix instance is used as input,
            must be the transformed drift matrix, A_y, given by thermox.preprocess,
            not thermox.utils.preprocess_drift_matrix.
        b: Drift displacement vector.
        D: Diffusion matrix (Array or thermox.ProcessedDiffusionMatrix).

    """
    A_y, D = handle_matrix_inputs(A, D)

    y0 = D.sqrt_inv @ (x0 - b)
    return b + D.sqrt @ expm_vp(A_y, y0, t)


def covariance(
    t: float,
    A: Array | ProcessedDriftMatrix,
    D: Array | ProcessedDiffusionMatrix,
) -> Array:
    """Computes the covariance of p(x_t | x_0)

    For x evolving according to the SDE:

    dx = - A * (x - b) dt + sqrt(D) dW

    Args:
        ts: Times at which samples are collected. Includes time for x0.
        A: Drift matrix (Array or thermox.ProcessedDriftMatrix).
            Note: If a thermox.ProcessedDriftMatrix instance is used as input,
            must be the transformed drift matrix, A_y, given by thermox.preprocess,
            not thermox.utils.preprocess_drift_matrix.
        D: Diffusion matrix (Array or thermox.ProcessedDiffusionMatrix).
    """
    A_y, D = handle_matrix_inputs(A, D)

    identity_diffusion_cov = (
        A_y.sym_eigvecs
        @ jnp.diag((1 - jnp.exp(-2 * A_y.sym_eigvals * t)) / (2 * A_y.sym_eigvals))
        @ A_y.sym_eigvecs.T
    )
    return D.sqrt @ identity_diffusion_cov @ D.sqrt.T


def mean_and_covariance(
    t: float,
    x0: Array,
    A: Array | ProcessedDriftMatrix,
    b: Array,
    D: Array | ProcessedDiffusionMatrix,
) -> tuple[Array, Array]:
    """Computes the mean and covariance of p(x_t | x_0)

    For x evolving according to the SDE:

    dx = - A * (x - b) dt + sqrt(D) dW

    Args:
        ts: Times at which samples are collected. Includes time for x0.
        x0: Initial state of the process.
        A: Drift matrix (Array or thermox.ProcessedDriftMatrix).
            Note: If a thermox.ProcessedDriftMatrix instance is used as input,
            must be the transformed drift matrix, A_y, given by thermox.preprocess,
            not thermox.utils.preprocess_drift_matrix.
        b: Drift displacement vector.
        D: Diffusion matrix (Array or thermox.ProcessedDiffusionMatrix).

    """
    A, D = handle_matrix_inputs(A, D)
    mean_val = mean(t, x0, A, b, D)
    covariance_val = covariance(t, A, D)
    return mean_val, covariance_val
