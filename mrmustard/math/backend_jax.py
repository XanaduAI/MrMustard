# Copyright 2023 Xanadu Quantum Technologies Inc.

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

# pylint: disable = missing-function-docstring, missing-class-docstring, fixme, too-many-positional-arguments

from __future__ import annotations
from typing import Callable, Sequence
from functools import partial

import os
import jax

jax.config.update("jax_default_device", jax.devices("cpu")[0])  # comment this line to run on GPU
jax.config.update("jax_enable_x64", True)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # comment this line to run on GPU
os.environ["JAX_PLATFORM_NAME"] = "cpu"  # comment this line to run on GPU
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from .autocast import Autocast
from .backend_base import BackendBase
from .lattice import strategies

from mrmustard.math.lattice.strategies.compactFock.inputValidation import (
    hermite_multidimensional_1leftoverMode,
    hermite_multidimensional_diagonal,
    hermite_multidimensional_diagonal_batch,
)


def identify_shape_hermite_multidimensional_diagonal(B, cutoffs):
    r"""Identify the shapes of the output of hermite_multidimensional_diagonal.

    Args:
        B: The B vector.
        cutoffs: The cutoffs of the modes.

    Returns:
        The shapes of the output of hermite_multidimensional_diagonal.
        Returns a tuple of size (5,)
    """
    M = len(cutoffs)
    return_shapes = []
    if B.ndim == 1:
        return_shapes.append(cutoffs)
        return_shapes.append((M,))
        if M == 1:
            return_shapes.append((1, 1, 1))
            return_shapes.append((1, 1, 1))
        else:
            return_shapes.append((1, 1, M))
            return_shapes.append((1, 1, M))
        return_shapes.append((2 * M,))

    elif B.ndim == 2:
        batch_length = B.shape[1]
        return_shapes.append(cutoffs + (batch_length,))
        return_shapes.append((M,) + cutoffs + (batch_length,))
        if M == 1:
            return_shapes.append((1, 1, 1) + (batch_length,))
            return_shapes.append((1, 1, 1) + (batch_length,))
        else:
            return_shapes.append((M, M - 1) + cutoffs + (batch_length,))
            return_shapes.append((M, M - 1) + cutoffs + (batch_length,))
        return_shapes.append((2 * M,) + cutoffs + (batch_length,))

    return tuple(return_shapes)


def identify_shape_hermite_multidimensional_1leftoverMode(cutoffs: tuple[int]):
    r"""Identify the shapes of the output of hermite_multidimensional_1leftoverMode.

    Args:
        cutoffs: The cutoffs of the modes.
    """
    r"""Identify the shapes of the output of hermite_multidimensional_1leftoverMode.

    Args:
        cutoffs: The cutoffs of the modes.

    Returns:
        The shapes of the output of hermite_multidimensional_1leftoverMode.
        Returns a tuple of size (5,) containing the shapes of arr0, arr2, arr1010, arr1001, arr1
    """
    M = len(cutoffs)
    cutoff_leftoverMode = cutoffs[0]
    cutoffs_tail = cutoffs[1:]

    return_shapes = []
    return_shapes.append((cutoff_leftoverMode, cutoff_leftoverMode) + cutoffs_tail)
    return_shapes.append((cutoff_leftoverMode, cutoff_leftoverMode) + (M - 1,) + cutoffs_tail)
    if M == 2:
        return_shapes.append((1, 1, 1, 1, 1))
        return_shapes.append((1, 1, 1, 1, 1))
    else:
        shape = (cutoff_leftoverMode, cutoff_leftoverMode) + (M - 1, M - 2) + cutoffs_tail
        return_shapes.append(shape)
        return_shapes.append(shape)
    return_shapes.append((cutoff_leftoverMode, cutoff_leftoverMode) + (2 * (M - 1),) + cutoffs_tail)
    return tuple(return_shapes)


# pylint: disable=too-many-public-methods
class BackendJax(BackendBase):
    """A JAX backend implementation."""

    int32 = jnp.int32
    int64 = jnp.int64
    float32 = jnp.float32
    float64 = jnp.float64
    complex64 = jnp.complex64
    complex128 = jnp.complex128
    JIT_FLAG = False

    def __init__(self):
        super().__init__(name="jax")

    def __repr__(self) -> str:
        return "BackendJax()"

    @partial(jax.jit, static_argnames=["self"])
    def abs(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.abs(array)

    @partial(jax.jit, static_argnames=["self"])
    def any(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.any(array)

    def arange(self, start: int, limit: int = None, delta: int = 1, dtype=None) -> jnp.ndarray:
        dtype = dtype or self.float64
        return jnp.arange(start, limit, delta, dtype=dtype)

    @partial(jax.jit, static_argnames=["self"])
    def argwhere(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.argwhere(array)

    def asnumpy(self, tensor: jnp.ndarray) -> np.ndarray:
        return np.array(tensor)

    # @partial(jax.jit, static_argnames=['self', 'axes'])
    def block(self, blocks: list[list[jnp.ndarray]], axes=(-2, -1)) -> jnp.ndarray:
        rows = [self.concat(row, axis=axes[1]) for row in blocks]
        return self.concat(rows, axis=axes[0])

    @partial(jax.jit, static_argnames=["self", "axis"])
    def prod(self, x: jnp.ndarray, axis: int | None):
        return jnp.prod(x, axis=axis)

    def assign(self, tensor: jnp.ndarray, value: jnp.ndarray) -> jnp.ndarray:
        tensor = value
        return tensor  # JAX arrays are immutable, so we just return the new value

    def astensor(self, array: np.ndarray | jnp.ndarray, dtype=None) -> jnp.ndarray:
        return jnp.asarray(array, dtype=dtype)

    def getnan(self, size):
        return jnp.full(size, jnp.nan)

    @partial(jax.jit, static_argnames=["self"])
    def log(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.log(array)

    @partial(jax.jit, static_argnames=["self", "dtype"])
    def atleast_1d(self, array: jnp.ndarray, dtype=None) -> jnp.ndarray:
        return jnp.atleast_1d(jnp.array(array, dtype=dtype))

    @partial(jax.jit, static_argnames=["self", "dtype"])
    def atleast_2d(self, array: jnp.ndarray, dtype=None) -> jnp.ndarray:
        return jnp.atleast_2d(jnp.array(array, dtype=dtype))

    @partial(jax.jit, static_argnames=["self", "dtype"])
    def atleast_3d(self, array: jnp.ndarray, dtype=None) -> jnp.ndarray:
        if isinstance(array, tuple):
            raise ValueError("Tuple input is not supported for atleast_3d.")
        array = self.atleast_2d(self.atleast_1d(array))
        if len(array.shape) == 2:
            array = array[None, ...]
        return array

    @partial(jax.jit, static_argnames=["self"])
    def block_diag(self, mat1: jnp.ndarray, mat2: jnp.ndarray) -> jnp.ndarray:
        Za = self.zeros((mat1.shape[-2], mat2.shape[-1]), dtype=mat1.dtype)
        Zb = self.zeros((mat2.shape[-2], mat1.shape[-1]), dtype=mat1.dtype)
        return self.concat(
            [self.concat([mat1, Za], axis=-1), self.concat([Zb, mat2], axis=-1)],
            axis=-2,
        )

    def constraint_func(self, bounds: tuple[float | None, float | None]) -> Callable:
        lower = -jnp.inf if bounds[0] is None else bounds[0]
        upper = jnp.inf if bounds[1] is None else bounds[1]

        @jax.jit
        def constraint(x):
            return jnp.clip(x, lower, upper)

        return constraint

    @partial(jax.jit, static_argnames=["self", "dtype"])
    def cast(self, array: jnp.ndarray, dtype=None) -> jnp.ndarray:
        if dtype is None:
            return array
        return jnp.asarray(array, dtype=dtype)

    @partial(jax.jit, static_argnames=["self", "axis"])
    def concat(self, values: Sequence[jnp.ndarray], axis: int) -> jnp.ndarray:
        try:
            return jnp.concatenate(values, axis)
        except ValueError:
            return jnp.array(values)

    @partial(jax.jit, static_argnames=["self", "axis"])
    def sort(self, array: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
        return jnp.sort(array, axis)

    @partial(jax.jit, static_argnames=["self"])
    def allclose(self, array1: jnp.ndarray, array2: jnp.ndarray, atol=1e-9) -> bool:
        return jnp.allclose(array1, array2, atol=atol)

    @partial(jax.jit, static_argnames=["self", "a_min", "a_max"])
    def clip(self, array: jnp.ndarray, a_min: float, a_max: float) -> jnp.ndarray:
        return jnp.clip(array, a_min, a_max)

    @Autocast()
    @partial(jax.jit, static_argnames=["self"])
    def maximum(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return jnp.maximum(a, b)

    @Autocast()
    @partial(jax.jit, static_argnames=["self"])
    def minimum(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return jnp.minimum(a, b)

    @partial(jax.jit, static_argnames=["self"])
    def lgamma(self, array: jnp.ndarray) -> jnp.ndarray:
        return jax.lax.lgamma(array)

    @partial(jax.jit, static_argnames=["self"])
    def conj(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.conj(array)

    @Autocast()
    def pow(self, x: jnp.ndarray, y: float) -> jnp.ndarray:
        return jnp.power(x, y)

    def Categorical(self, probs: jnp.ndarray, name: str):  # pylint: disable=unused-argument
        class Generator:
            def __init__(self, probs):
                self._probs = probs

            def sample(self):
                key = jax.random.PRNGKey(0)
                idx = jnp.arange(len(self._probs))
                return jax.random.choice(key, idx, p=self._probs / jnp.sum(self._probs))

        return Generator(probs)

    @partial(jax.jit, static_argnames=["self", "k"])
    def set_diag(self, array: jnp.ndarray, diag: jnp.ndarray, k: int) -> jnp.ndarray:
        i = jnp.arange(0, array.shape[-2] - abs(k))
        j = jnp.arange(abs(k), array.shape[-1])
        i = jnp.where(k < 0, i - array.shape[-2] + abs(k), i)
        j = jnp.where(k < 0, j - abs(k), j)
        return array.at[..., i, j].set(diag)

    # @partial(jax.jit, static_argnames=['self', 'name', 'dtype', 'bounds'])
    def new_variable(
        self,
        value: jnp.ndarray,
        bounds: tuple[float | None, float | None] | None,
        name: str,  # pylint: disable=unused-argument
        dtype="float64",
    ):
        value = jnp.array(value, dtype=dtype)
        return value

    @partial(jax.jit, static_argnames=["self"])
    def outer(self, array1: jnp.ndarray, array2: jnp.ndarray) -> jnp.ndarray:
        return jnp.tensordot(array1, array2, [[], []])

    @partial(jax.jit, static_argnames=["self", "name", "dtype"])
    def new_constant(self, value, name: str, dtype=None):  # pylint: disable=unused-argument
        dtype = dtype or self.float64
        value = self.astensor(value, dtype)
        return value

    @Autocast()
    @partial(jax.jit, static_argnames=["self", "data_format", "padding"])
    def convolution(
        self,
        array: jnp.ndarray,
        filters: jnp.ndarray,
        padding: str | None = None,
        data_format="NWC",  # pylint: disable=unused-argument
    ) -> jnp.ndarray:
        padding = padding or "VALID"
        return jax.lax.conv(array, filters, (1, 1), padding)

    def tile(self, array: jnp.ndarray, repeats: Sequence[int]) -> jnp.ndarray:
        repeats = tuple(repeats)
        array = jnp.array(array)
        return jnp.tile(array, repeats)

    @Autocast()
    @partial(jax.jit, static_argnames=["self"])
    def update_tensor(
        self, tensor: jnp.ndarray, indices: jnp.ndarray, values: jnp.ndarray
    ) -> jnp.ndarray:
        indices = self.atleast_2d(indices)
        indices = jnp.squeeze(indices, axis=-1)
        return tensor.at[indices].set(values)

    @Autocast()
    @partial(jax.jit, static_argnames=["self"])
    def update_add_tensor(
        self, tensor: jnp.ndarray, indices: jnp.ndarray, values: jnp.ndarray
    ) -> jnp.ndarray:
        indices = self.atleast_2d(indices)
        return tensor.at[tuple(indices.T)].add(values)

    @Autocast()
    @partial(jax.jit, static_argnames=["self"])
    def matvec(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return jnp.matmul(a, b)

    @partial(jax.jit, static_argnames=["self"])
    def cos(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.cos(array)

    @partial(jax.jit, static_argnames=["self"])
    def cosh(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.cosh(array)

    @partial(jax.jit, static_argnames=["self"])
    def det(self, matrix: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.det(matrix)

    # @partial(jax.jit, static_argnames=['self', 'k'])
    def diag(self, array: jnp.ndarray, k: int = 0) -> jnp.ndarray:
        if array.ndim == 0:
            return array
        elif array.ndim in [1, 2]:
            return jnp.diag(array, k=k)
        else:
            # fallback into more complex algorithm
            original_sh = jnp.array(array.shape)

            ravelled_sh = (jnp.prod(original_sh[:-1]), original_sh[-1])
            array = array.ravel().reshape(*ravelled_sh)

            ret = []
            for line in array:
                ret.append(jnp.diag(line, k))

            ret = jnp.array(ret)
            inner_shape = (
                original_sh[-1] + abs(k),
                original_sh[-1] + abs(k),
            )
            return ret.reshape(tuple(original_sh[:-1]) + tuple(inner_shape))

    @partial(jax.jit, static_argnames=["self", "k"])
    def diag_part(self, array: jnp.ndarray, k: int) -> jnp.ndarray:
        return jnp.diagonal(array, offset=k, axis1=-2, axis2=-1)

    @partial(jax.jit, static_argnames=["self", "string"])
    def einsum(self, string: str, *tensors) -> jnp.ndarray:
        if isinstance(string, str):
            return jnp.einsum(string, *tensors)
        return None

    @partial(jax.jit, static_argnames=["self"])
    def exp(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(array)

    @partial(jax.jit, static_argnames=["self", "axis"])
    def expand_dims(self, array: jnp.ndarray, axis: int) -> jnp.ndarray:
        return jnp.expand_dims(array, axis)

    @partial(jax.jit, static_argnames=["self"])
    def expm(self, matrix: jnp.ndarray) -> jnp.ndarray:
        return jsp.linalg.expm(matrix)

    def eye(self, size: int, dtype=None) -> jnp.ndarray:
        dtype = dtype or self.float64
        return jnp.eye(size, dtype=dtype)

    @partial(jax.jit, static_argnames=["self"])
    def eye_like(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.eye(array.shape[-1], dtype=array.dtype)

    # @partial(jax.jit, static_argnames=['self'])
    def from_backend(self, value) -> bool:
        return isinstance(value, jnp.ndarray)

    @partial(jax.jit, static_argnames=["self", "repeats", "axis"])
    def repeat(self, array: jnp.ndarray, repeats: int, axis: int = None) -> jnp.ndarray:
        return jnp.repeat(array, repeats, axis=axis)

    @partial(jax.jit, static_argnames=["self", "axis"])
    def gather(self, array: jnp.ndarray, indices: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
        return jnp.take(array, indices, axis=axis)

    @partial(jax.jit, static_argnames=["self"])
    def imag(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.imag(array)

    @partial(jax.jit, static_argnames=["self"])
    def inv(self, tensor: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.inv(tensor)

    def is_trainable(self, tensor: jnp.ndarray) -> bool:
        if isinstance(tensor, jnp.ndarray):
            return True
        return False

    @partial(jax.jit, static_argnames=["self"])
    def make_complex(self, real: jnp.ndarray, imag: jnp.ndarray) -> jnp.ndarray:
        return real + 1j * imag

    @Autocast()
    @partial(jax.jit, static_argnames=["self"])
    def matmul(self, *matrices: jnp.ndarray) -> jnp.ndarray:
        mat = jnp.linalg.multi_dot(matrices)
        """
        mat = matrices[0]
        for matrix in matrices[1:]:
            mat = jnp.matmul(mat, matrix)
        """
        return mat

    @partial(jax.jit, static_argnames=["self", "old", "new"])
    def moveaxis(
        self, array: jnp.ndarray, old: int | Sequence[int], new: int | Sequence[int]
    ) -> jnp.ndarray:
        return jnp.moveaxis(array, old, new)

    def ones(self, shape: Sequence[int], dtype=None) -> jnp.ndarray:
        dtype = dtype or self.float64
        return jnp.ones(shape, dtype=dtype)

    @partial(jax.jit, static_argnames=["self"])
    def ones_like(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.ones_like(array)

    @partial(jax.jit, static_argnames=["self"])
    def infinity_like(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.full_like(array, jnp.inf, dtype="complex128")

    def conditional(
        self, cond: jnp.ndarray, true_fn: Callable, false_fn: Callable, *args
    ) -> jnp.ndarray:
        return jax.lax.cond(jnp.all(cond), true_fn, false_fn, *args)

    def pad(
        self,
        array: jnp.ndarray,
        paddings: Sequence[tuple[int, int]],
        mode="constant",
        constant_values=0,
    ) -> jnp.ndarray:
        return jnp.pad(array, paddings, mode=mode.lower(), constant_values=constant_values)

    @partial(jax.jit, static_argnames=["self"])
    def pinv(self, matrix: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.pinv(matrix)

    @partial(jax.jit, static_argnames=["self"])
    def real(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.real(array)

    def reshape(self, array: jnp.ndarray, shape: Sequence[int]) -> jnp.ndarray:
        return jnp.reshape(array, shape)

    @partial(jax.jit, static_argnames=["self", "decimals"])
    def round(self, array: jnp.ndarray, decimals: int = 0) -> jnp.ndarray:
        return jnp.round(array, decimals)

    @partial(jax.jit, static_argnames=["self"])
    def sin(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.sin(array)

    @partial(jax.jit, static_argnames=["self"])
    def sinh(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.sinh(array)

    @partial(jax.jit, static_argnames=["self"])
    def solve(self, matrix: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
        if len(rhs.shape) == len(matrix.shape) - 1:
            rhs = jnp.expand_dims(rhs, -1)
            return jnp.linalg.solve(matrix, rhs)[..., 0]
        return jnp.linalg.solve(matrix, rhs)

    @partial(jax.jit, static_argnames=["self", "dtype"])
    def sqrt(self, x: jnp.ndarray, dtype=None) -> jnp.ndarray:
        return jnp.sqrt(self.cast(x, dtype))

    @partial(jax.jit, static_argnames=["self"])
    def kron(self, tensor1: jnp.ndarray, tensor2: jnp.ndarray):
        return jnp.kron(tensor1, tensor2)

    def boolean_mask(self, tensor: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        return tensor[mask]

    @partial(jax.jit, static_argnames=["self", "axes"])
    def sum(self, array: jnp.ndarray, axes: Sequence[int] = None):
        return jnp.sum(array, axis=axes)

    @partial(jax.jit, static_argnames=["self"])
    def norm(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.norm(array)

    def map_fn(self, func, elements):
        return jax.vmap(func)(elements)

    def MultivariateNormalTriL(self, loc: jnp.ndarray, scale_tril: jnp.ndarray, key: int = 0):
        class Generator:
            def __init__(self, mean, cov, key):
                self._mean = mean
                self._cov = cov
                self._rng = jax.random.PRNGKey(key)

            def sample(self, dtype=None):  # pylint: disable=unused-argument
                fn = jax.random.multivariate_normal
                ret = fn(self._rng, self._mean, self._cov)
                return ret

            def prob(self, x):
                return jsp.stats.multivariate_normal.pdf(x, mean=self._mean, cov=self._cov)

        scale_tril = scale_tril @ jnp.transpose(scale_tril)
        return Generator(loc, scale_tril, key)

    @Autocast()
    def tensordot(self, a: jnp.ndarray, b: jnp.ndarray, axes: Sequence[int]) -> jnp.ndarray:
        return jnp.tensordot(a, b, axes)

    @partial(jax.jit, static_argnames=["self", "dtype"])
    def trace(self, array: jnp.ndarray, dtype=None) -> jnp.ndarray:
        return self.cast(jnp.trace(array, axis1=-1, axis2=-2), dtype)

    def transpose(self, a: jnp.ndarray, perm: Sequence[int] = None) -> jnp.ndarray:
        if a is None:
            return None
        return jnp.transpose(a, perm)

    def zeros(self, shape: Sequence[int], dtype=None) -> jnp.ndarray:
        dtype = dtype or self.float64
        return jnp.zeros(shape, dtype=dtype)

    @partial(jax.jit, static_argnames=["self", "dtype"])
    def zeros_like(self, array: jnp.ndarray, dtype: str = "complex128") -> jnp.ndarray:
        return jnp.zeros_like(array, dtype=dtype)

    @partial(jax.jit, static_argnames=["self", "axis"])
    def squeeze(self, tensor: jnp.ndarray, axis=None):
        return jnp.squeeze(tensor, axis=axis)

    @partial(jax.jit, static_argnames=["self"])
    def cholesky(self, input: jnp.ndarray):
        return jnp.linalg.cholesky(input)

    @staticmethod
    @jax.jit
    def eigh(tensor: jnp.ndarray) -> tuple:
        return jnp.linalg.eigh(tensor)

    @partial(jax.jit, static_argnames=["self"])
    def where(self, array: jnp.ndarray, array1: jnp.ndarray, array2: jnp.ndarray):
        return jnp.where(array, array1, array2)

    @property
    def inf(self):
        return jnp.inf

    @staticmethod
    @jax.jit
    def eigvals(tensor: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.eigvals(tensor)

    @partial(jax.jit, static_argnames=["self"])
    def eqigendecomposition_sqrtm(self, tensor: jnp.ndarray) -> jnp.ndarray:
        eigvals, eigvecs = jnp.linalg.eigh(tensor)
        return eigvecs @ jnp.diag(jnp.sqrt(eigvals)) @ jnp.conj(eigvecs.T)

    @partial(jax.jit, static_argnames=["self", "dtype", "rtol", "atol"])
    def sqrtm(self, tensor: jnp.ndarray, dtype, rtol=1e-05, atol=1e-08) -> jnp.ndarray:

        ret = jax.lax.cond(
            jnp.allclose(tensor, 0, rtol=rtol, atol=atol),
            lambda _: self.zeros_like(tensor, dtype="complex128"),
            lambda _: jsp.linalg.sqrtm(tensor),
            None,
        )

        if dtype is None:
            return self.cast(ret, self.complex128)
        return self.cast(ret, dtype)

    # Special functions for optimization
    def DefaultEuclideanOptimizer(self):
        return jax.experimental.optimizers.adam(learning_rate=0.001)

    # @partial(jax.jit, static_argnames=['self'])
    def reorder_AB_bargmann(
        self, A: jnp.ndarray, B: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        r"""In mrmustard.math.numba.compactFock~ dimensions of the Fock representation are ordered like [mode0,mode0,mode1,mode1,...]
        while in mrmustard.physics.bargmann_utils the ordering is [mode0,mode1,...,mode0,mode1,...]. Here we reorder A and B.
        """
        ordering = jnp.arange(2 * A.shape[0] // 2).reshape(2, -1).T.flatten()
        A = self.gather(A, ordering, axis=1)
        A = self.gather(A, ordering)
        B = self.gather(B, ordering, axis=0)
        return A, B

    # ~~~~~~~~~~~~~~~~~
    # hermite_renormalized
    # ~~~~~~~~~~~~~~~~~
    @partial(jax.jit, static_argnames=["self", "shape"])
    def hermite_renormalized(
        self, A: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray, shape: tuple[int]
    ) -> jnp.ndarray:
        function = partial(strategies.vanilla, shape)
        G = jax.pure_callback(
            lambda A, b, c: function(np.array(A), np.array(b), np.array(c)),
            jax.ShapeDtypeStruct(shape, jnp.complex128),
            A,
            b,
            c,
        )
        return G

    # ~~~~~~~~~~~~~~~~~
    # hermite_renormalized_batch
    # ~~~~~~~~~~~~~~~~~

    @partial(jax.jit, static_argnames=["self", "shape"])
    def hermite_renormalized_batch(
        self, A: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray, shape: tuple[int]
    ) -> jnp.ndarray:
        function = partial(strategies.vanilla_batch, tuple(shape))
        G = jax.pure_callback(
            lambda A, b, c: function(np.array(A), np.array(b), np.array(c)),
            jax.ShapeDtypeStruct((b.shape[0],) + shape, jnp.complex128),
            A,
            b,
            c,
        )
        return G

    # ~~~~~~~~~~~~~~~~~
    # hermite_renormalized_diagonal
    # ~~~~~~~~~~~~~~~~~

    @partial(jax.jit, static_argnames=["self", "cutoffs"])
    def hermite_renormalized_diagonal(
        self, A: jnp.ndarray, B: jnp.ndarray, C: jnp.ndarray, cutoffs: tuple[int]
    ) -> jnp.ndarray:
        r"""First, reorder A and B parameters of Bargmann representation to match conventions in mrmustard.math.numba.compactFock~
        Then, calculate the required renormalized multidimensional Hermite polynomial.
        """
        A, B = self.reorder_AB_bargmann(A, B)

        function = partial(hermite_multidimensional_diagonal, cutoffs=tuple(cutoffs))
        poly0 = jax.pure_callback(
            # only first element is of interest
            lambda A, B, C: function(np.array(A), np.array(B), np.array(C))[0],
            jax.ShapeDtypeStruct(cutoffs, jnp.complex128),
            A,
            B,
            C,
        )
        return poly0

    # ~~~~~~~~~~~~~~~~~
    # hermite_renormalized_diagonal_reorderedAB
    # ~~~~~~~~~~~~~~~~~
    @partial(jax.jit, static_argnames=["self", "cutoffs"])
    def hermite_renormalized_diagonal_reorderedAB(
        self, A: jnp.ndarray, B: jnp.ndarray, C: jnp.ndarray, cutoffs: tuple[int]
    ) -> jnp.ndarray:
        r"""Renormalized multidimensional Hermite polynomial given by the "exponential" Taylor
        series of :math:`exp(C + Bx - Ax^2)` at zero, where the series has :math:`sqrt(n!)` at the
        denominator rather than :math:`n!`. Note the minus sign in front of ``A``.

        Calculates the diagonal of the Fock representation (i.e. the PNR detection probabilities of all modes)
        by applying the recursion relation in a selective manner.

        Args:
            A: The A matrix.
            B: The B vector.
            C: The C scalar.
            cutoffs: upper boundary of photon numbers in each mode

        Returns:
            The renormalized Hermite polynomial.
        """
        function = partial(hermite_multidimensional_diagonal, cutoffs=tuple(cutoffs))
        poly0 = jax.pure_callback(
            lambda A, B, C: function(np.array(A), np.array(B), np.array(C))[0],
            jax.ShapeDtypeStruct(cutoffs, jnp.complex128),
            A,
            B,
            C,
        )
        return poly0

    # ~~~~~~~~~~~~~~~~~
    # hermite_renormalized_diagonal_batch
    # ~~~~~~~~~~~~~~~~~

    @partial(jax.jit, static_argnames=["self", "cutoffs"])
    def hermite_renormalized_diagonal_batch(
        self, A: jnp.ndarray, B: jnp.ndarray, C: jnp.ndarray, cutoffs: tuple[int]
    ) -> jnp.ndarray:
        r"""Same as hermite_renormalized_diagonal but works for a batch of different B's."""
        A, B = self.reorder_AB_bargmann(A, B)
        return self.hermite_renormalized_diagonal_reorderedAB_batch(A, B, C, cutoffs=cutoffs)

    # ~~~~~~~~~~~~~~~~~
    # hermite_renormalized_diagonal_reorderedAB_batch
    # ~~~~~~~~~~~~~~~~~

    @partial(jax.jit, static_argnames=["self", "cutoffs"])
    def hermite_renormalized_diagonal_reorderedAB_batch(
        self, A: jnp.ndarray, B: jnp.ndarray, C: jnp.ndarray, cutoffs: tuple[int]
    ) -> jnp.ndarray:
        r"""Same as hermite_renormalized_diagonal_reorderedAB but works for a batch of different B's.

        Args:
            A: The A matrix.
            B: The B vectors.
            C: The C scalar.
            cutoffs: upper boundary of photon numbers in each mode

        Returns:
            The renormalized Hermite polynomial from different B values.
        """
        function = partial(hermite_multidimensional_diagonal_batch, cutoffs=tuple(cutoffs))
        poly0 = jax.pure_callback(
            lambda A, B, C: function(np.array(A), np.array(B), np.array(C))[0],
            jax.ShapeDtypeStruct(cutoffs + (B.shape[1],), jnp.complex128),
            A,
            B,
            C,
        )

        return poly0

    # ~~~~~~~~~~~~~~~~~
    # hermite_renormalized_binomial
    # ~~~~~~~~~~~~~~~~~

    @partial(jax.jit, static_argnames=["self", "shape", "max_l2", "global_cutoff"])
    def hermite_renormalized_binomial(
        self,
        A: jnp.ndarray,
        B: jnp.ndarray,
        C: jnp.ndarray,
        shape: tuple[int],
        max_l2: float | None,
        global_cutoff: int | None,
    ) -> jnp.ndarray:
        r"""Renormalized multidimensional Hermite polynomial given by the "exponential" Taylor
        series of :math:`exp(C + Bx + 1/2*Ax^2)` at zero, where the series has :math:`sqrt(n!)`
        at the denominator rather than :math:`n!`. The computation fills a tensor of given shape
        up to a given L2 norm or global cutoff, whichever applies first. The max_l2 value, if
        not provided, is set to the default value of the AUTOSHAPE_PROBABILITY setting.

        Args:
            A: The A matrix.
            B: The B vector.
            C: The C scalar.
            shape: The shape of the final tensor (local cutoffs).
            max_l2 (float): The maximum squared L2 norm of the tensor.
            global_cutoff (optional int): The global cutoff.

        Returns:
            The renormalized Hermite polynomial of given shape.
        """
        function = partial(strategies.binomial, tuple(shape))
        G = jax.pure_callback(
            lambda A, B, C, max_l2, global_cutoff: function(
                np.array(A), np.array(B), np.array(C), max_l2, global_cutoff
            )[0],
            jax.ShapeDtypeStruct(shape, jnp.complex128),
            A,
            B,
            C,
            max_l2,
            global_cutoff,
        )
        return G

    # ~~~~~~~~~~~~~~~~~
    # hermite_renormalized_1leftoverMode_reorderedAB
    # ~~~~~~~~~~~~~~~~~

    @partial(jax.jit, static_argnames=["self", "cutoffs"])
    def hermite_renormalized_1leftoverMode(self, A, B, C, cutoffs):
        A, B = self.reorder_AB_bargmann(A, B)
        return self.hermite_renormalized_1leftoverMode_reorderedAB(A, B, C, cutoffs=cutoffs)

    # ~~~~~~~~~~~~~~~~~
    # hermite_renormalized_1leftoverMode_reorderedAB
    # ~~~~~~~~~~~~~~~~~

    @partial(jax.jit, static_argnames=["self", "cutoffs"])
    def hermite_renormalized_1leftoverMode_reorderedAB(
        self, A: jnp.ndarray, B: jnp.ndarray, C: jnp.ndarray, cutoffs: tuple[int]
    ) -> jnp.ndarray:
        r"""Renormalized multidimensional Hermite polynomial given by the "exponential" Taylor
        series of :math:`exp(C + Bx - Ax^2)` at zero, where the series has :math:`sqrt(n!)` at the
        denominator rather than :math:`n!`. Note the minus sign in front of ``A``.

        Calculates all possible Fock representations of mode 0,
        where all other modes are PNR detected.
        This is done by applying the recursion relation in a selective manner.

        Args:
            A: The A matrix.
            B: The B vector.
            C: The C scalar.
            cutoffs: upper boundary of photon numbers in each mode

        Returns:
            The renormalized Hermite polynomial.
        """
        function = partial(hermite_multidimensional_1leftoverMode, cutoffs=tuple(cutoffs))
        poly0, _, _, _, _ = jax.pure_callback(
            lambda A, B, C: function(np.array(A), np.array(B), np.array(C))[0],
            jax.ShapeDtypeStruct(cutoffs, jnp.complex128),
            A,
            B,
            C,
        )
        return poly0
