from typing import Tuple
from jax import numpy as jnp
from jax import Array
from dataclasses import dataclass


@dataclass
class ProcessedDriftMatrix:
    """Stores eigendecompositions of A, (A+A^T)/2"""

    val: Array
    eigvals: Array
    eigvecs: Array
    eigvecs_inv: Array
    sym_eigvals: Array
    sym_eigvecs: Array


def preprocess_drift_matrix(A: Array, A_spd: bool = False) -> ProcessedDriftMatrix:
    """Preprocesses matrix A (calculates eigendecompositions of A and (A+A^T)/2)

    Args:
        - A: drift matrix.
        - A_spd: if true uses jax.linalg.eigh to calculate eigendecomposition of A.
            If false uses jax.scipy.linalg.eig.
            jax.linalg.eigh supports gradients but assumes A is Hermitian
            (i.e. real symmetric).
            See https://github.com/google/jax/issues/2748

    Returns:
        ProcessedDriftMatrix containing eigendeomcomposition of A and (A+A^T)/2.
    """
    if A_spd:
        A_eigvals, A_eigvecs = jnp.linalg.eigh(A)
    else:
        A_eigvals, A_eigvecs = jnp.linalg.eig(A)

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


@dataclass
class ProcessedDiffusionMatrix:
    """Stores preprocessed diffusion matrix D^0.5 and D^-0.5 via Cholesky"""

    val: Array
    sqrt: Array
    sqrt_inv: Array


def preprocess_diffusion_matrix(D: Array) -> ProcessedDiffusionMatrix:
    """Preprocesses diffusion matrix D (calculates D^0.5 and D^-0.5 via Cholesky)

    Args:
        - D: diffusion matrix.

    Returns:
        ProcessedDiffusionMatrix containing D^0.5 and D^-0.5.
    """
    D_sqrt = jnp.linalg.cholesky(D)
    D_sqrt_inv = jnp.linalg.inv(D_sqrt)
    return ProcessedDiffusionMatrix(D, D_sqrt, D_sqrt_inv)


def preprocess(
    A: Array, D: Array, A_spd: bool = False
) -> Tuple[ProcessedDriftMatrix, ProcessedDiffusionMatrix]:
    """Transforms the drift matrix A to A_y = D^0.5 @ A @ D^-0.5 for diffusion matrix D
    and preprocesses (calculates eigendecompositions (A_y+A_y^T)/2 as well as
    D^0.5 and D^-0.5)

    Args:
        - A: drift matrix.
        - D: diffusion matrix.
        - A_spd: if true uses jax.linalg.eigh to calculate eigendecomposition of A.
            If false uses jax.scipy.linalg.eig.
            jax.linalg.eigh supports gradients but assumes A is Hermitian
            (i.e. real symmetric).
            See https://github.com/google/jax/issues/2748

    Returns:
        ProcessedDriftMatrix containing eigendecomposition of A_y and (A_y+A_y^T)/2.
            where A_y = D^0.5 @ A @ D^-0.5
        ProcessedDiffusionMatrix containing D^0.5 and D^-0.5.
    """
    PD = preprocess_diffusion_matrix(D)
    A_y = PD.sqrt_inv @ A @ PD.sqrt
    PA_y = preprocess_drift_matrix(A_y, A_spd)
    return PA_y, PD
