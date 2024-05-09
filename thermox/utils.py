from typing import NamedTuple, Tuple
from jax import numpy as jnp
from jax import Array
from fmmax.utils import (
    eig,
)  # differentiable and jit-able eigendecomposition, not yet available in jax, see https://github.com/google/jax/issues/2748


class ProcessedDriftMatrix(NamedTuple):
    """Stores eigendecompositions of A, (A+A^T)/2"""

    val: Array
    eigvals: Array
    eigvecs: Array
    eigvecs_inv: Array
    sym_eigvals: Array
    sym_eigvecs: Array


def preprocess_drift_matrix(A: Array) -> ProcessedDriftMatrix:
    """Preprocesses matrix A (calculates eigendecompositions of A and (A+A^T)/2)

    Args:
        A: drift matrix.

    Returns:
        ProcessedDriftMatrix containing eigendeomcomposition of A and (A+A^T)/2.
    """

    A_eigvals, A_eigvecs = eig(A + 0.0j)

    A_eigvals = A_eigvals.real
    A_eigvecs = A_eigvecs.real

    A_eigvecs_inv = jnp.linalg.inv(A_eigvecs)

    symA = 0.5 * (A + A.T)
    symA_eigvals, symA_eigvecs = jnp.linalg.eigh(symA)

    return ProcessedDriftMatrix(
        A,
        A_eigvals.real,
        A_eigvecs,
        A_eigvecs_inv,
        symA_eigvals,
        symA_eigvecs,
    )


class ProcessedDiffusionMatrix(NamedTuple):
    """Stores preprocessed diffusion matrix D^0.5 and D^-0.5 via Cholesky"""

    val: Array
    sqrt: Array
    sqrt_inv: Array


def preprocess_diffusion_matrix(D: Array) -> ProcessedDiffusionMatrix:
    """Preprocesses diffusion matrix D (calculates D^0.5 and D^-0.5 via Cholesky)

    Args:
        D: diffusion matrix.

    Returns:
        ProcessedDiffusionMatrix containing D^0.5 and D^-0.5.
    """
    D_sqrt = jnp.linalg.cholesky(D)
    D_sqrt_inv = jnp.linalg.inv(D_sqrt)
    return ProcessedDiffusionMatrix(D, D_sqrt, D_sqrt_inv)


def preprocess(
    A: Array, D: Array
) -> Tuple[ProcessedDriftMatrix, ProcessedDiffusionMatrix]:
    """Transforms the drift matrix A to A_y = D^-0.5 @ A @ D^0.5 for diffusion matrix D
    and preprocesses (calculates eigendecompositions (A_y+A_y^T)/2 as well as
    D^0.5 and D^-0.5)

    Args:
        A: drift matrix.
        D: diffusion matrix.

    Returns:
        ProcessedDriftMatrix containing eigendecomposition of A_y and (A_y+A_y^T)/2.
            where A_y = D^-0.5 @ A @ D^0.5
        ProcessedDiffusionMatrix containing D^0.5 and D^-0.5.
    """
    PD = preprocess_diffusion_matrix(D)
    A_y = PD.sqrt_inv @ A @ PD.sqrt
    PA_y = preprocess_drift_matrix(A_y)
    return PA_y, PD
