from typing import NamedTuple, Tuple
from jax import numpy as jnp
from jax import Array
from fmmax.eig import (
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
        A: Drift matrix.

    Returns:
        ProcessedDriftMatrix containing eigendeomcomposition of A and (A+A^T)/2.
    """

    A_eigvals, A_eigvecs = eig(A + 0.0j)
    A_eigvecs_inv = jnp.linalg.inv(A_eigvecs)

    symA = 0.5 * (A + A.T)
    symA_eigvals, symA_eigvecs = jnp.linalg.eigh(symA)

    return ProcessedDriftMatrix(
        A,
        A_eigvals,
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
        D: Diffusion matrix.

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
        A: Drift matrix.
        D: Diffusion matrix.

    Returns:
        ProcessedDriftMatrix containing eigendecomposition of A_y and (A_y+A_y^T)/2.
            where A_y = D^-0.5 @ A @ D^0.5
        ProcessedDiffusionMatrix containing D^0.5 and D^-0.5.
    """
    PD = preprocess_diffusion_matrix(D)
    A_y = PD.sqrt_inv @ A @ PD.sqrt
    PA_y = preprocess_drift_matrix(A_y)
    return PA_y, PD


def handle_matrix_inputs(
    A: Array | ProcessedDriftMatrix, D: Array | ProcessedDiffusionMatrix
) -> Tuple[ProcessedDriftMatrix, ProcessedDiffusionMatrix]:
    """Checks the type of the input drift matrix, A, and diffusion matrix, D,
    and ensures that they are processed in the correct way.
    Helper function for sample and log_prob functions.

    Args:
        A: Drift matrix (Array or thermox.ProcessedDriftMatrix).
        D: Diffusion matrix (Array or thermox.ProcessedDiffusionMatrix).

    Returns:
        ProcessedDriftMatrix containing eigendecomposition of A_y and (A_y+A_y^T)/2.
            where A_y = D^-0.5 @ A @ D^0.5
        ProcessedDiffusionMatrix containing D^0.5 and D^-0.5.
    """
    if isinstance(A, Array) or isinstance(D, Array):
        if isinstance(A, ProcessedDriftMatrix):
            A = A.val
        if isinstance(D, ProcessedDiffusionMatrix):
            D = D.val
        A, D = preprocess(A, D)
    return A, D
