"""Functions for calculating sparsity patterns of Jacobians."""

import functools
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .exceptions import UnsupportedArgumentException

_batched_jvp_columnwise = jax.vmap(jax.jvp, in_axes=(None, None, -1))

_NO_VALUE = object()


def jacfwd_sparsity(
    fun: Callable,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = _NO_VALUE,  # type: ignore
):
    """Produces a function that calculates the sparsity pattern of the outputs of `jax.jacfwd(fun)`.

    Currently, only differentiating with regard to the first argument (`argnums=0`) is supported,
    and this argument must be a one-dimensional array. Any other arguments will be closed over.
    """

    exceptions = []
    if has_aux:
        exceptions.append(UnsupportedArgumentException("has_aux", has_aux))
    if holomorphic is not _NO_VALUE:
        exceptions.append(UnsupportedArgumentException("holomorphic", holomorphic))
    if argnums != 0:
        exceptions.append(UnsupportedArgumentException("argnums", argnums))
    if exceptions:
        raise ExceptionGroup("Invalid arguments to jacfwd_sparsity", exceptions)

    @functools.wraps(fun)
    def fun_jac(diff_arg, *extra_args, **extra_kwargs):
        if not hasattr(diff_arg, "dtype") or diff_arg.dtype not in (np.float32, np.float64):
            raise Exception(
                "jacfwd_sparsity only supports differentiating "
                "with respect to arrays with float dtype."
            )

        if not hasattr(diff_arg, "shape") or len(diff_arg.shape) != 1:
            raise Exception(
                "jacfwd_sparsity currently only supports differentiating "
                "with respect to a 1D array argument."
            )

        (in_size,) = diff_arg.shape

        def fun_partial(diff_arg):
            return fun(diff_arg, *extra_args, **extra_kwargs)

        jac_basis = np.diag(np.full(in_size, np.nan))

        _, jac = _batched_jvp_columnwise(fun_partial, (diff_arg,), (jac_basis,))

        jac_sparsity = jnp.isnan(jac)

        return jac_sparsity

    return fun_jac
