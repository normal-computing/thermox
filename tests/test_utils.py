from jax import numpy as jnp

from thermox.utils import (
    handle_matrix_inputs,
    ProcessedDriftMatrix,
    ProcessedDiffusionMatrix,
    preprocess,
)


def test_handle_matrix_inputs_arrays():
    A = jnp.array([[1, 3], [1, 4]])
    D = jnp.array([[9, 4], [4, 20]])

    a, d = preprocess(A, D)

    A_star, D_star = preprocess(A, D)

    assert isinstance(A_star, ProcessedDriftMatrix)
    assert isinstance(D_star, ProcessedDiffusionMatrix)
    assert jnp.all(a.val == A_star.val)


def test_handle_matrix_inputs_processed():
    A = jnp.array([[1, 3], [1, 4]])
    D = jnp.array([[9, 4], [4, 20]])

    a, d = preprocess(A, D)

    A_star, D_star = handle_matrix_inputs(a, d)

    assert isinstance(A_star, ProcessedDriftMatrix)
    assert isinstance(D_star, ProcessedDiffusionMatrix)
    assert jnp.all(a.val == A_star.val)


def test_handle_matrix_inputs_array_drift_processed_diffusion():
    A = jnp.array([[1, 3], [1, 4]])
    D = jnp.array([[9, 4], [4, 20]])

    a, d = preprocess(A, D)

    A_star, D_star = handle_matrix_inputs(A, d)

    assert isinstance(A_star, ProcessedDriftMatrix)
    assert isinstance(D_star, ProcessedDiffusionMatrix)
    assert jnp.all(a.val == A_star.val)


def test_handle_matrix_inputs_array_diffusion_processed_drift():
    A = jnp.array([[1, 3], [1, 4]])
    D = jnp.array([[9, 4], [4, 20]])

    a, d = preprocess(A, D)

    A_star, D_star = handle_matrix_inputs(a, D)

    assert isinstance(A_star, ProcessedDriftMatrix)
    assert isinstance(D_star, ProcessedDiffusionMatrix)
    assert not jnp.all(a.val == A_star.val)
