# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains the JAX backend."""

from __future__ import annotations

from functools import partial
from warnings import warn

import numpy as np

from mrmustard.parameters import Variable

from .backend_base import BackendBase

try:
    import jax
except ImportError:
    raise ImportError(
        "The JAX backend requires the `jax_backend` group. Please install it using `uv pip install -g jax_backend`."
    ) from None
else:
    import equinox as eqx
    import jax.numpy as jnp
    import jax.scipy as jsp

    from mrmustard.mathlib.jax_vjps import (
        beamsplitter_jax,
        complex_gaussian_integral_1_jax,
        complex_gaussian_integral_2_jax,
        displacement_jax,
        hermite_renormalized_1leftoverMode_jax,
        hermite_renormalized_batched_jax,
        hermite_renormalized_binomial_jax,
        hermite_renormalized_diagonal_jax,
        hermite_renormalized_jax,
        homodyne_projector_jax,
        squeezed_jax,
        squeezer_jax,
    )

# ~~~~~~~
# Helpers
# ~~~~~~~


def get_all_subclasses(cls):
    r"""Returns all subclasses of a given class."""
    all_subclasses = []
    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))
    return all_subclasses


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# JAX Gaussian Integral Implementations
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class BackendJax(BackendBase):
    r"""A JAX backend implementation."""

    int32 = jnp.int32
    int64 = jnp.int64
    float32 = jnp.float32
    float64 = jnp.float64
    complex64 = jnp.complex64
    complex128 = jnp.complex128

    def __init__(self):
        super().__init__(name="jax")

    def __repr__(self):
        return "BackendJax()"

    def _tree_flatten(self):
        return (), ()

    @classmethod
    def _tree_unflatten(cls, aux, children):
        return cls(*children, *aux)

    @jax.jit
    def abs(self, array):
        return jnp.abs(array)

    def all(self, array):
        return jnp.all(array)

    @jax.jit
    def angle(self, array):
        return jnp.angle(array)

    @jax.jit
    def any(self, array):
        return jnp.any(array)

    def arange(self, start, limit=None, delta=1, dtype=None):
        dtype = dtype or self.float64
        return jnp.arange(start, limit, delta, dtype=dtype)

    @partial(jax.jit, static_argnames=["axis"])
    def argmax(self, array, axis=None):
        return jnp.argmax(array, axis=axis)

    @partial(jax.jit, static_argnames=["axis"])
    def argmin(self, array, axis=None):
        return jnp.argmin(array, axis=axis)

    @partial(jax.jit, static_argnames=["axis"])
    def argsort(self, array, axis=None):
        return jnp.argsort(array, axis=axis)

    def complex_gaussian_integral_1_single(self, A, b, idx12, A_out, b_out, log_c_out):
        """Single (non-batched) complex Gaussian integral for one Abc using JAX.

        Args:
            A: Complex matrix
            b: Complex vector
            idx12: Indices to integrate over
            A_out: Ignored (for API compatibility)
            b_out: Ignored (for API compatibility)
            log_c_out: Ignored (for API compatibility)

        Returns:
            Tuple of (A_out, b_out, log_c_out)
        """
        return complex_gaussian_integral_1_jax(A, b, idx12)

    def complex_gaussian_integral_1_batched(self, A, b, idx12, A_out, b_out, log_c_out):
        """Batched complex Gaussian integral for one Abc using JAX vmap.

        Args:
            A: Batched complex matrices
            b: Batched complex vectors
            idx12: Indices to integrate over (same for all batch elements)
            A_out: Ignored (for API compatibility)
            b_out: Ignored (for API compatibility)
            log_c_out: Ignored (for API compatibility)

        Returns:
            Tuple of (A_out, b_out, log_c_out) with batch dimensions
        """
        # Get batch shapes
        batch_shape_A = A.shape[:-2]
        batch_shape_b = b.shape[:-1]

        # Broadcast to common batch shape
        target_batch_shape = jnp.broadcast_shapes(batch_shape_A, batch_shape_b)

        # Broadcast A and b to target shape
        A_broadcast = jnp.broadcast_to(A, target_batch_shape + A.shape[-2:])
        b_broadcast = jnp.broadcast_to(b, target_batch_shape + b.shape[-1:])

        # Apply vmap for each batch dimension
        vmapped_fn = complex_gaussian_integral_1_jax
        for _ in range(len(target_batch_shape)):
            vmapped_fn = jax.vmap(vmapped_fn, in_axes=(0, 0, None))

        return vmapped_fn(A_broadcast, b_broadcast, idx12)

    def complex_gaussian_integral_2_single(
        self, A1, b1, A2, b2, idx1, idx2, A_out, b_out, log_c_out
    ):
        """Single (non-batched) complex Gaussian integral for two Abc using JAX.

        Args:
            A1: First A matrix
            b1: First b vector
            A2: Second A matrix
            b2: Second b vector
            idx1: Integer array of indices for first Abc
            idx2: Integer array of indices for second Abc
            A_out: Ignored (for API compatibility)
            b_out: Ignored (for API compatibility)
            log_c_out: Ignored (for API compatibility)

        Returns:
            Tuple of (A_out, b_out, log_c_out)
        """
        return complex_gaussian_integral_2_jax(A1, b1, A2, b2, idx1, idx2)

    def complex_gaussian_integral_2_batched(
        self, A1, b1, A2, b2, idx1, idx2, A_out, b_out, log_c_out
    ):
        """Batched complex Gaussian integral for two Abc using JAX vmap.

        Args:
            A1: First batched A matrix
            b1: First batched b vector
            A2: Second batched A matrix
            b2: Second batched b vector
            idx1: Integer array of indices for first Abc
            idx2: Integer array of indices for second Abc
            A_out: Ignored (for API compatibility)
            b_out: Ignored (for API compatibility)
            log_c_out: Ignored (for API compatibility)

        Returns:
            Tuple of (A_out, b_out, log_c_out) with batch dimensions
        """
        # Get batch shapes
        batch_shape_A1 = A1.shape[:-2]
        batch_shape_b1 = b1.shape[:-1]
        batch_shape_A2 = A2.shape[:-2]
        batch_shape_b2 = b2.shape[:-1]

        # Broadcast to common batch shape
        target_batch_shape = jnp.broadcast_shapes(
            batch_shape_A1, batch_shape_b1, batch_shape_A2, batch_shape_b2
        )

        # Broadcast all inputs to target shape
        A1_broadcast = jnp.broadcast_to(A1, target_batch_shape + A1.shape[-2:])
        b1_broadcast = jnp.broadcast_to(b1, target_batch_shape + b1.shape[-1:])
        A2_broadcast = jnp.broadcast_to(A2, target_batch_shape + A2.shape[-2:])
        b2_broadcast = jnp.broadcast_to(b2, target_batch_shape + b2.shape[-1:])

        # Apply vmap for each batch dimension
        vmapped_fn = complex_gaussian_integral_2_jax
        for _ in range(len(target_batch_shape)):
            vmapped_fn = jax.vmap(vmapped_fn, in_axes=(0, 0, 0, 0, None, None))

        return vmapped_fn(A1_broadcast, b1_broadcast, A2_broadcast, b2_broadcast, idx1, idx2)

    def asnumpy(self, tensor):
        return np.array(tensor)

    def BackendError(self):
        return jax.errors.TracerArrayConversionError

    @partial(jax.jit, static_argnames=["shape"])
    def broadcast_to(self, array, shape):
        return jnp.broadcast_to(array, shape)

    def broadcast_arrays(self, *arrays):
        return jnp.broadcast_arrays(*arrays)

    def prod(self, x, axis):
        return jnp.prod(jnp.asarray(x), axis=axis)

    def astensor(self, array, dtype=None):
        return jnp.asarray(array, dtype=dtype)

    @jax.jit
    def log(self, array):
        return jnp.log(array)

    def atleast_nd(self, array, n, dtype=None):
        return jnp.array(array, ndmin=n, dtype=dtype)

    @partial(jax.jit, static_argnames=["dtype"])
    def cast(self, array, dtype=None):
        if dtype is None:
            return array
        return jnp.asarray(array, dtype=dtype)

    @partial(jax.jit, static_argnames=["axis"])
    def concat(self, values, axis):
        try:
            return jnp.concatenate(values, axis)
        except ValueError:
            return jnp.asarray(values)

    @partial(jax.jit, static_argnames=["axis"])
    def sort(self, array, axis=-1):
        return jnp.sort(array, axis)

    def allclose(self, array1, array2, atol, rtol):
        return jnp.allclose(jnp.asarray(array1), jnp.asarray(array2), atol=atol, rtol=rtol)

    @partial(jax.jit, static_argnames=["a_min", "a_max"])
    def clip(self, array, a_min, a_max):
        return jnp.clip(array, a_min, a_max)

    def conj(self, array):
        return jnp.conj(array)

    def pow(self, x, y):
        return jnp.power(x, y)

    @jax.jit
    def outer(self, array1, array2):
        return self.tensordot(array1, array2, [[], []])

    def tile(self, array, repeats):
        return jnp.tile(array, repeats)

    @jax.jit
    def update_add_tensor(self, tensor, indices, values):
        indices = self.atleast_nd(indices, 2)
        return tensor.at[tuple(indices.T)].add(values)

    @jax.jit
    def matvec(self, a, b):
        return jnp.matmul(a, b[..., None])[..., 0]

    @jax.jit
    def cos(self, array):
        return jnp.cos(array)

    @jax.jit
    def cosh(self, array):
        return jnp.cosh(array)

    @jax.jit
    def det(self, matrix):
        return jnp.linalg.det(matrix)

    def diagonal(self, array, offset, axis1, axis2):
        return jnp.diagonal(array, offset=offset, axis1=axis1, axis2=axis2)

    def diag(self, array, k=0):
        return jnp.diag(array, k=k)

    @jax.jit
    def exp(self, array):
        return jnp.exp(array)

    @partial(jax.jit, static_argnames=["axis"])
    def expand_dims(self, array, axis):
        return jnp.expand_dims(array, axis)

    @partial(jax.jit, static_argnames=["axis"])
    def squeeze(self, array, axis=None):
        return jnp.squeeze(array, axis=axis)

    @jax.jit
    def expm(self, matrix):
        return jsp.linalg.expm(matrix)

    def eye(self, size, dtype=None):
        dtype = dtype or self.float64
        return jnp.eye(size, dtype=dtype)

    @jax.jit
    def eye_like(self, array):
        return jnp.eye(array.shape[-1], dtype=array.dtype)

    @jax.jit
    def equal(self, a, b):
        return jnp.equal(a, b)

    def gather(self, array, indices, axis=0):
        return jnp.take(jnp.asarray(array), jnp.asarray(indices, dtype=jnp.int64), axis=axis)

    @jax.jit
    def imag(self, array):
        return jnp.imag(array)

    @jax.jit
    def inv(self, tensor):
        return jnp.linalg.inv(tensor)

    def iscomplexobj(self, x):
        return jnp.iscomplexobj(x)

    def isnan(self, array):
        return jnp.isnan(array)

    def issubdtype(self, arg1, arg2):
        return jnp.issubdtype(arg1, arg2)

    @jax.jit
    def lgamma(self, array):
        return jax.lax.lgamma(array)

    @jax.jit
    def make_complex(self, real, imag):
        return real + 1j * imag

    @jax.jit
    def matmul(self, *matrices):
        try:
            return jnp.linalg.multi_dot(matrices)
        except ValueError:
            mat = matrices[0]
            for matrix in matrices[1:]:
                mat = jnp.matmul(mat, matrix)
            return mat

    @jax.jit
    def max(self, array):
        return jnp.max(array)

    @jax.jit
    def maximum(self, a, b):
        return jnp.maximum(a, b)

    @jax.jit
    def minimum(self, a, b):
        return jnp.minimum(a, b)

    @jax.jit
    def mod(self, a, b):
        return jnp.mod(a, b)

    @partial(jax.jit, static_argnames=["old", "new"])
    def moveaxis(self, array, old, new):
        return jnp.moveaxis(array, old, new)

    @partial(jax.jit, static_argnames=["axis"])
    def mean(self, array, axis=None):
        return jnp.mean(array, axis=axis)

    def ones(self, shape, dtype=None):
        dtype = dtype or self.float64
        return jnp.ones(shape, dtype=dtype)

    def full(self, shape, fill_value, dtype=None):
        dtype = dtype or jnp.result_type(fill_value)
        return jnp.full(shape, fill_value, dtype=dtype)

    @jax.jit
    def ones_like(self, array):
        return jnp.ones_like(array)

    @jax.jit
    def infinity_like(self, array):
        return jnp.full_like(array, jnp.inf, dtype="complex128")

    def conditional(self, cond, true_fn, false_fn, *args):
        return jax.lax.cond(jnp.all(cond), true_fn, false_fn, *args)

    def error_if(self, array, condition, msg):
        try:
            eqx.error_if(array, condition, msg)
        except eqx.EquinoxRuntimeError as e:
            raise ValueError(msg) from e

    def pad(self, array, paddings, mode="constant", constant_values=0):
        return jnp.pad(array, paddings, mode=mode.lower(), constant_values=constant_values)

    @jax.jit
    def pinv(self, matrix):
        return jnp.linalg.pinv(matrix)

    @jax.jit
    def real(self, array):
        return jnp.real(array)

    def reshape(self, array, shape):
        return jnp.reshape(array, shape)

    def shape(self, array):
        return jnp.shape(array)

    @jax.jit
    def sin(self, array):
        return jnp.sin(array)

    @jax.jit
    def sinh(self, array):
        return jnp.sinh(array)

    @jax.jit
    def solve(self, matrix, rhs):
        if len(rhs.shape) == len(matrix.shape) - 1:
            rhs = jnp.expand_dims(rhs, -1)
            return jnp.linalg.solve(matrix, rhs)[..., 0]
        return jnp.linalg.solve(matrix, rhs)

    @partial(jax.jit, static_argnames=["dtype"])
    def sqrt(self, x, dtype=None):
        return jnp.sqrt(self.cast(jnp.asarray(x), dtype))

    @partial(jax.jit, static_argnames=["axis"])
    def stack(self, arrays, axis=0):
        return jnp.stack(arrays, axis=axis)

    @jax.jit
    def kron(self, tensor1, tensor2):
        return jnp.kron(tensor1, tensor2)

    def sum(self, array, axes=None):
        return jnp.sum(jnp.asarray(array), axis=axes)

    def swapaxes(self, array, axis1, axis2):
        return jnp.swapaxes(array, axis1, axis2)

    @partial(jax.jit, static_argnames=["axis", "keepdims"])
    def norm(self, array, axis=None, keepdims=False):
        return jnp.linalg.norm(array, axis=axis, keepdims=keepdims)

    def map_fn(self, func, elements):
        return jax.vmap(func)(elements)

    def tensordot(self, a, b, axes):
        return jnp.tensordot(a, b, axes)

    @partial(jax.jit, static_argnames=["dtype"])
    def trace(self, array, dtype=None):
        return self.cast(jnp.trace(array, axis1=-1, axis2=-2), dtype)

    def transpose(self, a, perm=None):
        return jnp.transpose(a, perm)

    @jax.jit
    def tan(self, array):
        return jnp.tan(array)

    @jax.jit
    def tanh(self, array):
        return jnp.tanh(array)

    def zeros(self, shape, dtype=None):
        dtype = dtype or self.float64
        return jnp.zeros(shape, dtype=dtype)

    @partial(jax.jit, static_argnames=["dtype"])
    def zeros_like(self, array, dtype="complex128"):
        return jnp.zeros_like(array, dtype=dtype)

    def xlogy(self, x, y):
        return jax.scipy.special.xlogy(x, y)

    @staticmethod
    @jax.jit
    def eigh(tensor):
        return jnp.linalg.eigh(tensor)

    @staticmethod
    @jax.jit
    def eigvals(tensor):
        return jnp.linalg.eigvals(tensor)

    @partial(jax.jit, static_argnames=["dtype", "rtol", "atol"])
    def sqrtm(self, tensor, dtype, rtol=1e-05, atol=1e-08):
        # There is no GPU Schur decomposition for jax, which jax sqrtm relies on.
        # https://github.com/jax-ml/jax/issues/28927
        # Implementing our own eigenvalue decomposition based sqrtm, which is differentiable and
        # works on GPU/TPU.

        def eig_sqrtm_hermitian(tensor):
            eigenvalues, eigenvectors = jnp.linalg.eigh(tensor)
            tol = atol + rtol * jnp.max(jnp.abs(eigenvalues))
            clamped_eigenvalues = jnp.where(jnp.abs(eigenvalues) < tol, 0, eigenvalues)
            sqrt_eigenvalues = jnp.sqrt(clamped_eigenvalues.astype(jnp.complex128))
            return eigenvectors @ jnp.diag(sqrt_eigenvalues) @ jnp.conj(eigenvectors.T)

        def eig_sqrtm_non_hermitian(tensor):
            eigenvalues, eigenvectors = jnp.linalg.eig(tensor)
            sqrt_eigenvalues = jnp.sqrt(eigenvalues)
            scaled_eigenvectors = eigenvectors @ jnp.diag(sqrt_eigenvalues)
            return jnp.linalg.solve(eigenvectors.T, scaled_eigenvectors.T).T

        def eig_sqrtm(tensor):
            hermitian = jnp.allclose(tensor, jnp.conj(jnp.transpose(tensor)), rtol=rtol, atol=atol)
            return jax.lax.cond(
                hermitian,
                lambda: eig_sqrtm_hermitian(tensor),
                lambda: eig_sqrtm_non_hermitian(tensor),
            )

        ret = jax.lax.cond(
            jnp.allclose(tensor, 0, rtol=rtol, atol=atol),
            lambda _: self.zeros_like(tensor, dtype="complex128"),
            lambda _: eig_sqrtm(tensor),
            None,
        )

        if dtype is None:
            return self.cast(ret, self.complex128)
        return self.cast(ret, dtype)

    @jax.jit
    def reorder_AB_bargmann(self, A, B):
        r"""In mrmustard.math.numba.compactFock~ dimensions of the Fock representation are ordered like [mode0,mode0,mode1,mode1,...]
        while in mrmustard.physics.bargmann_utils the ordering is [mode0,mode1,...,mode0,mode1,...]. Here we reorder A and B.
        """
        ordering = jnp.arange(2 * A.shape[0] // 2).reshape(2, -1).T.flatten()
        A = self.gather(A, ordering, axis=1)
        A = self.gather(A, ordering)
        B = self.gather(B, ordering, axis=0)
        return A, B

    # ~~~~~~~~~~~~~~~~~~~~
    # hermite_renormalized
    # ~~~~~~~~~~~~~~~~~~~~

    def hermite_renormalized(self, A, b, c, shape, stable, out=None):
        if out is not None:
            warn(
                "Using the out= keyword argument with the jax backend"
                " results in increased memory usage.",
                stacklevel=2,
            )
            out[...] = hermite_renormalized_jax(A, b, c, shape, stable)
            return out
        return hermite_renormalized_jax(A, b, c, shape, stable)

    def hermite_renormalized_batched(self, A, b, c, shape, stable, out=None):
        if out is not None:
            warn(
                "Using the out= keyword argument with the jax backend"
                " results in increased memory usage.",
                stacklevel=2,
            )
            out[...] = hermite_renormalized_batched_jax(A, b, c, shape, stable)
            return out
        return hermite_renormalized_batched_jax(A, b, c, shape, stable)

    def hermite_renormalized_binomial(self, A, B, C, shape, max_l2, global_cutoff):
        return hermite_renormalized_binomial_jax(A, B, C, shape, max_l2, global_cutoff)

    def hermite_renormalized_diagonal(self, A, B, C, cutoffs, reorderedAB):
        A, B = self.reorder_AB_bargmann(A, B) if reorderedAB else (A, B)
        return hermite_renormalized_diagonal_jax(A, B, C, cutoffs)[0]

    def hermite_renormalized_1leftoverMode(self, A, B, C, output_cutoff, pnr_cutoffs, reorderedAB):
        A, B = self.reorder_AB_bargmann(A, B) if reorderedAB else (A, B)
        return hermite_renormalized_1leftoverMode_jax(A, B, C, output_cutoff, pnr_cutoffs)[0]

    # ~~~~~~~~~~~~~~~~~~~~~~~
    # Fock lattice strategies
    # ~~~~~~~~~~~~~~~~~~~~~~~

    def displacement(self, alpha, shape):
        return displacement_jax(alpha, shape)

    def beamsplitter(self, theta, phi, shape, method):
        return beamsplitter_jax(theta, phi, shape, method)

    def homodyne_projector(self, fock_dim, A, b, c, out):
        if out is not None:
            warn(
                "Using the out= keyword argument with the jax backend"
                " results in increased memory usage.",
                stacklevel=2,
            )
            out[...] = homodyne_projector_jax(fock_dim, A, b, c)
            return out
        return homodyne_projector_jax(fock_dim, A, b, c)

    def squeezed(self, r, phi, shape):
        return squeezed_jax(r, phi, shape)

    def squeezer(self, r, phi, shape):
        return squeezer_jax(r, phi, shape)


# defining custom pytree nodes
jax.tree_util.register_pytree_node(BackendJax, BackendJax._tree_flatten, BackendJax._tree_unflatten)
jax.tree_util.register_pytree_node(Variable, Variable._tree_flatten, Variable._tree_unflatten)
