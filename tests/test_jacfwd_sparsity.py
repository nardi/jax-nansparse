import jax
import jax.experimental.sparse as jspr
import jax.numpy as jnp
import numpy as np
import pytest

import jax_nansparse as jns

N = 3
"""Used as size for dummy input arrays."""


def test_identity_function():
    """For the identity function we expect an identity matrix as sparsity pattern."""

    def f(x):
        return x

    x = np.arange(N, dtype=float)

    f_jac_sparsity = jns.jacfwd_sparsity(f)(x)

    assert np.allclose(f_jac_sparsity, np.eye(N, dtype=bool))


def test_identity_function_jit():
    """Same as above, but test that the sparsity computation can be jitted."""

    def f(x):
        return x

    x = np.arange(N, dtype=float)

    f_jac_sparsity = jax.jit(jns.jacfwd_sparsity(f))(x)

    assert np.allclose(f_jac_sparsity, np.eye(N, dtype=bool))


def test_zero_preserving_elementwise_function():
    """This function outputs 0 if x is 0 and a nonzero value for any other x, so we again expect an
    identity matrix as sparsity pattern."""

    def f(x):
        return jnp.sin(x**2) / jnp.exp(x)

    x = np.arange(N, dtype=float)

    f_jac_sparsity = jns.jacfwd_sparsity(f)(x)

    assert np.allclose(f_jac_sparsity, np.eye(N, dtype=bool))


def test_densifying_function():
    """This function mixes all inputs together, so we expect a fully dense matrix as sparsity
    pattern."""

    def f(x):
        return x / jnp.sum(x)

    x = np.arange(N, dtype=float)

    f_jac_sparsity = jns.jacfwd_sparsity(f)(x)

    assert np.allclose(f_jac_sparsity, np.full((N, N), True))


@pytest.mark.xfail
def test_conditional_function():
    """This function filters out small x values, so we (incorrectly) deduce the sparsity pattern to
    be all zeroes. In actuality it should be an identity matrix (since we shouldn't depend on the
    value of x when deriving the pattern)."""

    def f(x):
        return jnp.where(x > 5, x, 0.0)

    x = np.arange(N, dtype=float)

    f_jac_sparsity = jns.jacfwd_sparsity(f)(x)

    assert np.allclose(f_jac_sparsity, np.eye(N, dtype=bool))


def test_dense_matrix_multiplication():
    """This function multiplies x by an identity matrix, so we would expect an identity matrix as
    sparsity pattern. However because NaNs are dominant and each row-column dot product contains at
    least one NaN, all values in the result will be NaN."""

    Id = np.eye(N, dtype=float)

    def f(x):
        return Id @ x

    x = np.arange(N, dtype=float)

    f_jac_sparsity = jns.jacfwd_sparsity(f)(x)

    assert np.allclose(f_jac_sparsity, np.full((N, N), True))


def test_sparse_matrix_multiplication():
    """This function multiplies x by a sparse identity matrix, in which case zero values never enter
    the computation. As such, the sparsity pattern is an identity matrix, as expected."""

    Id = jspr.BCOO.fromdense(np.eye(N, dtype=float))

    def f(x):
        return Id @ x

    x = np.arange(N, dtype=float)

    f_jac_sparsity = jns.jacfwd_sparsity(f)(x)

    assert np.allclose(f_jac_sparsity, np.eye(N, dtype=bool))
