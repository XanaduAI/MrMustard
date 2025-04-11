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

# pylint: disable = missing-function-docstring, missing-class-docstring, fixme, too-many-positional-arguments

from __future__ import annotations  # pragma: no cover
from typing import Callable, Sequence  # pragma: no cover
from functools import partial  # pragma: no cover

import jax  # pragma: no cover
import jax.numpy as jnp  # pragma: no cover
import jax.scipy as jsp  # pragma: no cover
import numpy as np  # pragma: no cover
import equinox as eqx  # pragma: no cover
from jax import tree_util  # pragma: no cover

from .autocast import Autocast  # pragma: no cover
from .backend_base import BackendBase  # pragma: no cover
from .lattice import strategies  # pragma: no cover
from .lattice.strategies.compactFock.inputValidation import (  # pragma: no cover
    hermite_multidimensional_1leftoverMode,
    hermite_multidimensional_diagonal,
    hermite_multidimensional_diagonal_batch,
)

jax.config.update("jax_enable_x64", True)  # pragma: no cover

# pylint: disable=too-many-public-methods
class BackendJax(BackendBase):  # pragma: no cover
    """A JAX backend implementation."""

    int32 = jnp.int32
    int64 = jnp.int64
    float32 = jnp.float32
    float64 = jnp.float64
    complex64 = jnp.complex64
    complex128 = jnp.complex128

    def __init__(self):
        super().__init__(name="jax")

    def __repr__(self) -> str:
        return "BackendJax()"

    def _tree_flatten(self):
        return (), ()

    @classmethod
    def _tree_unflatten(cls, aux, children):
        return cls(*children, *aux)

    @jax.jit
    def abs(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.abs(array)

    @jax.jit
    def any(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.any(array)

    def arange(self, start: int, limit: int = None, delta: int = 1, dtype=None) -> jnp.ndarray:
        dtype = dtype or self.float64
        return jnp.arange(start, limit, delta, dtype=dtype)

    def asnumpy(self, tensor: jnp.ndarray) -> np.ndarray:
        return np.array(tensor)

    @partial(jax.jit, static_argnames=["shape"])
    def broadcast_to(self, array: jnp.ndarray, shape: tuple[int]) -> jnp.ndarray:
        return jnp.broadcast_to(array, shape)

    def broadcast_arrays(self, *arrays: list[jnp.ndarray]) -> list[jnp.ndarray]:
        return jnp.broadcast_arrays(*arrays)

    @partial(jax.jit, static_argnames=["axis"])
    def prod(self, x: jnp.ndarray, axis: int | None):
        return jnp.prod(x, axis=axis)

    @jax.jit
    def assign(self, tensor: jnp.ndarray, value: jnp.ndarray) -> jnp.ndarray:
        tensor = value
        return tensor

    def astensor(self, array: np.ndarray | jnp.ndarray, dtype=None) -> jnp.ndarray:
        return jnp.asarray(array, dtype=dtype)

    @jax.jit
    def log(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.log(array)

    def atleast_nd(self, array: jnp.ndarray, n: int, dtype=None) -> jnp.ndarray:
        return jnp.array(array, ndmin=n, dtype=dtype)

    @jax.jit
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

    @partial(jax.jit, static_argnames=["dtype"])
    def cast(self, array: jnp.ndarray, dtype=None) -> jnp.ndarray:
        if dtype is None:
            return array
        return jnp.asarray(array, dtype=dtype)

    @partial(jax.jit, static_argnames=["axis"])
    def concat(self, values: Sequence[jnp.ndarray], axis: int) -> jnp.ndarray:
        try:
            return jnp.concatenate(values, axis)
        except ValueError:
            return jnp.array(values)

    @partial(jax.jit, static_argnames=["axis"])
    def sort(self, array: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
        return jnp.sort(array, axis)

    @jax.jit
    def allclose(self, array1: jnp.ndarray, array2: jnp.ndarray, atol=1e-9, rtol=1e-5) -> bool:
        return jnp.allclose(array1, array2, atol=atol, rtol=rtol)

    @partial(jax.jit, static_argnames=["a_min", "a_max"])
    def clip(self, array: jnp.ndarray, a_min: float, a_max: float) -> jnp.ndarray:
        return jnp.clip(array, a_min, a_max)

    @Autocast()
    @jax.jit
    def maximum(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return jnp.maximum(a, b)

    @Autocast()
    @jax.jit
    def minimum(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return jnp.minimum(a, b)

    @jax.jit
    def lgamma(self, array: jnp.ndarray) -> jnp.ndarray:
        return jax.lax.lgamma(array)

    @jax.jit
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

    @partial(jax.jit, static_argnames=["k"])
    def set_diag(self, array: jnp.ndarray, diag: jnp.ndarray, k: int) -> jnp.ndarray:
        i = jnp.arange(0, array.shape[-2] - abs(k))
        j = jnp.arange(abs(k), array.shape[-1])
        i = jnp.where(k < 0, i - array.shape[-2] + abs(k), i)
        j = jnp.where(k < 0, j - abs(k), j)
        return array.at[..., i, j].set(diag)

    def new_variable(
        self,
        value: jnp.ndarray,
        bounds: tuple[float | None, float | None] | None,
        name: str,
        dtype="float64",
    ):  # pylint: disable=unused-argument
        value = jnp.array(value, dtype=dtype)
        return value

    @jax.jit
    def outer(self, array1: jnp.ndarray, array2: jnp.ndarray) -> jnp.ndarray:
        return jnp.tensordot(array1, array2, [[], []])

    @partial(jax.jit, static_argnames=["name", "dtype"])
    def new_constant(self, value, name: str, dtype=None):  # pylint: disable=unused-argument
        dtype = dtype or self.float64
        value = self.astensor(value, dtype)
        return value

    @Autocast()
    @partial(jax.jit, static_argnames=["data_format", "padding"])
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
        return jnp.tile(array, repeats)

    @Autocast()
    @jax.jit
    def update_tensor(
        self, tensor: jnp.ndarray, indices: jnp.ndarray, values: jnp.ndarray
    ) -> jnp.ndarray:
        indices = self.atleast_nd(indices, 2)
        indices = jnp.squeeze(indices, axis=-1)
        return tensor.at[indices].set(values)

    @Autocast()
    @jax.jit
    def update_add_tensor(
        self, tensor: jnp.ndarray, indices: jnp.ndarray, values: jnp.ndarray
    ) -> jnp.ndarray:
        indices = self.atleast_nd(indices, 2)
        return tensor.at[tuple(indices.T)].add(values)

    @Autocast()
    @jax.jit
    def matvec(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return jnp.matmul(a, b[..., None])[..., 0]

    @jax.jit
    def cos(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.cos(array)

    @jax.jit
    def cosh(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.cosh(array)

    @jax.jit
    def det(self, matrix: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.det(matrix)

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

    @partial(jax.jit, static_argnames=["k"])
    def diag_part(self, array: jnp.ndarray, k: int) -> jnp.ndarray:
        return jnp.diagonal(array, offset=k, axis1=-2, axis2=-1)

    @partial(jax.jit, static_argnames=["string", "optimize"])
    def einsum(self, string: str, *tensors, optimize: bool | str = False) -> jnp.ndarray:
        return jnp.einsum(string, *tensors, optimize=optimize)

    @jax.jit
    def exp(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.exp(array)

    @partial(jax.jit, static_argnames=["axis"])
    def expand_dims(self, array: jnp.ndarray, axis: int) -> jnp.ndarray:
        return jnp.expand_dims(array, axis)

    @jax.jit
    def expm(self, matrix: jnp.ndarray) -> jnp.ndarray:
        return jsp.linalg.expm(matrix)

    def eye(self, size: int, dtype=None) -> jnp.ndarray:
        dtype = dtype or self.float64
        return jnp.eye(size, dtype=dtype)

    @jax.jit
    def eye_like(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.eye(array.shape[-1], dtype=array.dtype)

    def from_backend(self, value) -> bool:
        return isinstance(value, jnp.ndarray)

    @partial(jax.jit, static_argnames=["repeats", "axis"])
    def repeat(self, array: jnp.ndarray, repeats: int, axis: int = None) -> jnp.ndarray:
        return jnp.repeat(array, repeats, axis=axis)

    @partial(jax.jit, static_argnames=["axis"])
    def gather(self, array: jnp.ndarray, indices: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
        return jnp.take(array, indices, axis=axis)

    @jax.jit
    def imag(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.imag(array)

    @jax.jit
    def inv(self, tensor: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.inv(tensor)

    def is_trainable(self, tensor: jnp.ndarray) -> bool:  # pylint: disable=unused-argument
        return False

    @jax.jit
    def make_complex(self, real: jnp.ndarray, imag: jnp.ndarray) -> jnp.ndarray:
        return real + 1j * imag

    @Autocast()
    @jax.jit
    def matmul(self, *matrices: jnp.ndarray) -> jnp.ndarray:
        mat = jnp.linalg.multi_dot(matrices)
        return mat

    @partial(jax.jit, static_argnames=["old", "new"])
    def moveaxis(
        self, array: jnp.ndarray, old: int | Sequence[int], new: int | Sequence[int]
    ) -> jnp.ndarray:
        return jnp.moveaxis(array, old, new)

    def ones(self, shape: Sequence[int], dtype=None) -> jnp.ndarray:
        dtype = dtype or self.float64
        return jnp.ones(shape, dtype=dtype)

    @jax.jit
    def ones_like(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.ones_like(array)

    @jax.jit
    def infinity_like(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.full_like(array, jnp.inf, dtype="complex128")

    def conditional(
        self, cond: jnp.ndarray, true_fn: Callable, false_fn: Callable, *args
    ) -> jnp.ndarray:
        return jax.lax.cond(jnp.all(cond), true_fn, false_fn, *args)

    def error_if(self, array: jnp.ndarray, condition: jnp.ndarray, msg: str):
        eqx.error_if(array, condition, msg)

    def pad(
        self,
        array: jnp.ndarray,
        paddings: Sequence[tuple[int, int]],
        mode="constant",
        constant_values=0,
    ) -> jnp.ndarray:
        return jnp.pad(array, paddings, mode=mode.lower(), constant_values=constant_values)

    @jax.jit
    def pinv(self, matrix: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.pinv(matrix)

    @jax.jit
    def real(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.real(array)

    def reshape(self, array: jnp.ndarray, shape: Sequence[int]) -> jnp.ndarray:
        return jnp.reshape(array, shape)

    @partial(jax.jit, static_argnames=["decimals"])
    def round(self, array: jnp.ndarray, decimals: int = 0) -> jnp.ndarray:
        return jnp.round(array, decimals)

    @jax.jit
    def sin(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.sin(array)

    @jax.jit
    def sinh(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.sinh(array)

    @jax.jit
    def solve(self, matrix: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
        if len(rhs.shape) == len(matrix.shape) - 1:
            rhs = jnp.expand_dims(rhs, -1)
            return jnp.linalg.solve(matrix, rhs)[..., 0]
        return jnp.linalg.solve(matrix, rhs)

    @partial(jax.jit, static_argnames=["dtype"])
    def sqrt(self, x: jnp.ndarray, dtype=None) -> jnp.ndarray:
        return jnp.sqrt(self.cast(x, dtype))

    @partial(jax.jit, static_argnames=["axis"])
    def stack(self, arrays: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
        return jnp.stack(arrays, axis=axis)

    @jax.jit
    def kron(self, tensor1: jnp.ndarray, tensor2: jnp.ndarray):
        return jnp.kron(tensor1, tensor2)

    def boolean_mask(self, tensor: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        return tensor[mask]

    @partial(jax.jit, static_argnames=["axes"])
    def sum(self, array: jnp.ndarray, axes: Sequence[int] = None):
        return jnp.sum(array, axis=axes)

    @jax.jit
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

    @partial(jax.jit, static_argnames=["dtype"])
    def trace(self, array: jnp.ndarray, dtype=None) -> jnp.ndarray:
        return self.cast(jnp.trace(array, axis1=-1, axis2=-2), dtype)

    def transpose(self, a: jnp.ndarray, perm: Sequence[int] = None) -> jnp.ndarray:
        return jnp.transpose(a, perm)

    def zeros(self, shape: Sequence[int], dtype=None) -> jnp.ndarray:
        dtype = dtype or self.float64
        return jnp.zeros(shape, dtype=dtype)

    @partial(jax.jit, static_argnames=["dtype"])
    def zeros_like(self, array: jnp.ndarray, dtype: str = "complex128") -> jnp.ndarray:
        return jnp.zeros_like(array, dtype=dtype)

    @partial(jax.jit, static_argnames=["axis"])
    def squeeze(self, tensor: jnp.ndarray, axis=None):
        return jnp.squeeze(tensor, axis=axis)

    @jax.jit
    def cholesky(self, input: jnp.ndarray):
        return jnp.linalg.cholesky(input)

    @staticmethod
    @jax.jit
    def eigh(tensor: jnp.ndarray) -> tuple:
        return jnp.linalg.eigh(tensor)

    @staticmethod
    @jax.jit
    def eigvals(tensor: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.eigvals(tensor)

    @jax.jit
    def eqigendecomposition_sqrtm(self, tensor: jnp.ndarray) -> jnp.ndarray:
        eigvals, eigvecs = jnp.linalg.eigh(tensor)
        return eigvecs @ jnp.diag(jnp.sqrt(eigvals)) @ jnp.conj(eigvecs.T)

    @partial(jax.jit, static_argnames=["dtype", "rtol", "atol"])
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

    @jax.jit
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
    # hermite_renormalized_unbatched
    # ~~~~~~~~~~~~~~~~~
    @partial(jax.jit, static_argnames=["shape", "stable"])
    def hermite_renormalized_unbatched(
        self,
        A: jnp.ndarray,
        b: jnp.ndarray,
        c: jnp.ndarray,
        shape: tuple[int],
        stable: bool = False,
    ) -> jnp.ndarray:
        if stable:
            G = jax.pure_callback(
                lambda A, b, c: strategies.stable_numba(
                    shape, np.array(A), np.array(b), np.array(c)
                ),
                jax.ShapeDtypeStruct(shape, jnp.complex128),
                A,
                b,
                c,
            )
        else:
            G = jax.pure_callback(
                lambda A, b, c: strategies.vanilla_numba(
                    shape, np.array(A), np.array(b), np.array(c)
                ),
                jax.ShapeDtypeStruct(shape, jnp.complex128),
                A,
                b,
                c,
            )
        return G

    # ~~~~~~~~~~~~~~~~~
    # hermite_renormalized_batched
    # ~~~~~~~~~~~~~~~~~
    @partial(jax.jit, static_argnames=["shape", "stable"])
    def hermite_renormalized_batched(
        self,
        A: jnp.ndarray,
        b: jnp.ndarray,
        c: jnp.ndarray,
        shape: tuple[int],
        stable: bool = False,
    ) -> jnp.ndarray:
        batch_size = A.shape[0]
        output_shape = (batch_size,) + shape
        G = jax.pure_callback(
            lambda A, b, c: strategies.vanilla_batch_numba(
                shape, np.array(A), np.array(b), np.array(c), stable
            ),
            jax.ShapeDtypeStruct(output_shape, jnp.complex128),
            A,
            b,
            c,
        )
        return G

    # ~~~~~~~~~~~~~~~~~
    # hermite_renormalized_diagonal
    # ~~~~~~~~~~~~~~~~~

    @partial(jax.jit, static_argnames=["cutoffs"])
    def hermite_renormalized_diagonal(
        self, A: jnp.ndarray, B: jnp.ndarray, C: jnp.ndarray, cutoffs: tuple[int]
    ) -> jnp.ndarray:
        r"""First, reorder A and B parameters of Bargmann representation to match conventions in mrmustard.math.numba.compactFock~
        Then, calculate the required renormalized multidimensional Hermite polynomial.
        """
        A, B = self.reorder_AB_bargmann(A, B)
        return self.hermite_renormalized_diagonal_reorderedAB(A, B, C, cutoffs=cutoffs)

    # ~~~~~~~~~~~~~~~~~
    # hermite_renormalized_diagonal_reorderedAB
    # ~~~~~~~~~~~~~~~~~
    @partial(jax.jit, static_argnames=["cutoffs"])
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

    @partial(jax.jit, static_argnames=["cutoffs"])
    def hermite_renormalized_diagonal_batch(
        self, A: jnp.ndarray, B: jnp.ndarray, C: jnp.ndarray, cutoffs: tuple[int]
    ) -> jnp.ndarray:
        r"""Same as hermite_renormalized_diagonal but works for a batch of different B's."""
        A, B = self.reorder_AB_bargmann(A, B)
        return self.hermite_renormalized_diagonal_reorderedAB_batch(A, B, C, cutoffs=cutoffs)

    # ~~~~~~~~~~~~~~~~~
    # hermite_renormalized_diagonal_reorderedAB_batch
    # ~~~~~~~~~~~~~~~~~

    @partial(jax.jit, static_argnames=["cutoffs"])
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

    @partial(jax.jit, static_argnames=["shape", "max_l2", "global_cutoff"])
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

    @partial(jax.jit, static_argnames=["output_cutoff", "pnr_cutoffs"])
    def hermite_renormalized_1leftoverMode(self, A, B, C, output_cutoff, pnr_cutoffs):
        A, B = self.reorder_AB_bargmann(A, B)
        cutoffs = (output_cutoff + 1,) + tuple(p + 1 for p in pnr_cutoffs)
        return self.hermite_renormalized_1leftoverMode_reorderedAB(A, B, C, cutoffs=cutoffs)

    # ~~~~~~~~~~~~~~~~~
    # hermite_renormalized_1leftoverMode_reorderedAB
    # ~~~~~~~~~~~~~~~~~

    @partial(jax.jit, static_argnames=["cutoffs"])
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
        function = partial(hermite_multidimensional_1leftoverMode, cutoffs=cutoffs)
        poly0 = jax.pure_callback(
            lambda A, B, C: function(np.array(A), np.array(B), np.array(C))[0],
            jax.ShapeDtypeStruct((cutoffs[0],) + cutoffs, jnp.complex128),
            A,
            B,
            C,
        )
        return poly0


# defining the pytree node for the JaxBackend.
# This allows to skip specifying `self` in static_argnames.
tree_util.register_pytree_node(
    BackendJax, BackendJax._tree_flatten, BackendJax._tree_unflatten
)  # pragma: no cover
