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

from collections.abc import Callable, Sequence
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import optax
from opt_einsum import contract
from platformdirs import user_cache_dir

from mrmustard.lab import Circuit, CircuitComponent
from mrmustard.physics.ansatz import Ansatz

from .backend_base import BackendBase
from .jax_vjps import (
    beamsplitter_jax,
    displacement_jax,
    hermite_renormalized_batched_jax,
    hermite_renormalized_unbatched_jax,
)
from .lattice import strategies
from .lattice.strategies.compactFock.inputValidation import (
    hermite_multidimensional_1leftoverMode,
    hermite_multidimensional_diagonal,
    hermite_multidimensional_diagonal_batch,
)
from .parameter_set import ParameterSet
from .parameters import Constant, Variable

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", f"{user_cache_dir('mrmustard')}/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")


# ~~~~~~~
# Helpers
# ~~~~~~~


def get_all_subclasses(cls):
    r"""
    Returns all subclasses of a given class.
    """
    all_subclasses = []
    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))
    return all_subclasses


class BackendJax(BackendBase):
    r"""
    A JAX backend implementation.
    """

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

    def all(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.all(array)

    @jax.jit
    def angle(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.angle(array)

    @jax.jit
    def any(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.any(array)

    def arange(
        self,
        start: int,
        limit: int | None = None,
        delta: int = 1,
        dtype=None,
    ) -> jnp.ndarray:
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
        return value

    def astensor(self, array: np.ndarray | jnp.ndarray, dtype=None) -> jnp.ndarray:
        return jnp.asarray(array, dtype=dtype)

    @jax.jit
    def log(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.log(array)

    def atleast_nd(self, array: jnp.ndarray, n: int, dtype=None) -> jnp.ndarray:
        return jnp.array(array, ndmin=n, dtype=dtype)

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
            return jnp.asarray(values)

    @partial(jax.jit, static_argnames=["axis"])
    def sort(self, array: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
        return jnp.sort(array, axis)

    @jax.jit
    def allclose(self, array1: jnp.ndarray, array2: jnp.ndarray, atol=1e-9, rtol=1e-5) -> bool:
        return jnp.allclose(array1, array2, atol=atol, rtol=rtol)

    @partial(jax.jit, static_argnames=["a_min", "a_max"])
    def clip(self, array: jnp.ndarray, a_min: float, a_max: float) -> jnp.ndarray:
        return jnp.clip(array, a_min, a_max)

    @jax.jit
    def conj(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.conj(array)

    def pow(self, x: jnp.ndarray, y: float) -> jnp.ndarray:
        return jnp.power(x, y)

    def new_variable(
        self,
        value: jnp.ndarray,
        bounds: tuple[float | None, float | None] | None,
        name: str,
        dtype="float64",
    ):
        return jnp.array(value, dtype=dtype)

    @jax.jit
    def outer(self, array1: jnp.ndarray, array2: jnp.ndarray) -> jnp.ndarray:
        return self.tensordot(array1, array2, [[], []])

    @partial(jax.jit, static_argnames=["name", "dtype"])
    def new_constant(self, value, name: str, dtype=None):
        dtype = dtype or self.float64
        return self.astensor(value, dtype)

    def tile(self, array: jnp.ndarray, repeats: Sequence[int]) -> jnp.ndarray:
        return jnp.tile(array, repeats)

    def update_tensor(
        self,
        tensor: jnp.ndarray,
        indices: jnp.ndarray,
        values: jnp.ndarray,
    ) -> jnp.ndarray:
        return tensor.at[indices].set(values)

    @jax.jit
    def update_add_tensor(
        self,
        tensor: jnp.ndarray,
        indices: jnp.ndarray,
        values: jnp.ndarray,
    ) -> jnp.ndarray:
        indices = self.atleast_nd(indices, 2)
        return tensor.at[tuple(indices.T)].add(values)

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
        if array.ndim in [1, 2]:
            return jnp.diag(array, k=k)
        # fallback into more complex algorithm
        original_sh = jnp.asarray(array.shape)

        ravelled_sh = (jnp.prod(original_sh[:-1]), original_sh[-1])
        array = array.ravel().reshape(*ravelled_sh)
        ret = jnp.asarray([jnp.diag(line, k) for line in array])
        inner_shape = (
            original_sh[-1] + abs(k),
            original_sh[-1] + abs(k),
        )
        return ret.reshape(tuple(original_sh[:-1]) + tuple(inner_shape))

    @partial(jax.jit, static_argnames=["k"])
    def diag_part(self, array: jnp.ndarray, k: int) -> jnp.ndarray:
        return jnp.diagonal(array, offset=k, axis1=-2, axis2=-1)

    def einsum(self, string: str, *tensors, optimize: bool | str) -> jnp.ndarray:
        return contract(string, *tensors, optimize=optimize, backend="jax")

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

    @partial(jax.jit, static_argnames=["axis"])
    def gather(self, array: jnp.ndarray, indices: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
        return jnp.take(array, indices, axis=axis)

    @jax.jit
    def imag(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.imag(array)

    @jax.jit
    def inv(self, tensor: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.inv(tensor)

    def isnan(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.isnan(array)

    def is_trainable(self, tensor: jnp.ndarray) -> bool:
        return False

    @jax.jit
    def lgamma(self, array: jnp.ndarray) -> jnp.ndarray:
        return jax.lax.lgamma(array)

    @jax.jit
    def make_complex(self, real: jnp.ndarray, imag: jnp.ndarray) -> jnp.ndarray:
        return real + 1j * imag

    @jax.jit
    def matmul(self, *matrices: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.multi_dot(matrices)

    @jax.jit
    def max(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.max(array)

    @jax.jit
    def maximum(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return jnp.maximum(a, b)

    @jax.jit
    def minimum(self, a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return jnp.minimum(a, b)

    @partial(jax.jit, static_argnames=["old", "new"])
    def moveaxis(
        self,
        array: jnp.ndarray,
        old: int | Sequence[int],
        new: int | Sequence[int],
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
        self,
        cond: jnp.ndarray,
        true_fn: Callable,
        false_fn: Callable,
        *args,
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

    @partial(jax.jit, static_argnames=["axes"])
    def sum(self, array: jnp.ndarray, axes: Sequence[int] | None = None):
        return jnp.sum(array, axis=axes)

    def swapaxes(self, array: jnp.ndarray, axis1: int, axis2: int) -> jnp.ndarray:
        return jnp.swapaxes(array, axis1, axis2)

    @jax.jit
    def norm(self, array: jnp.ndarray) -> jnp.ndarray:
        return jnp.linalg.norm(array)

    def map_fn(self, func, elements):
        return jax.vmap(func)(elements)

    def tensordot(self, a: jnp.ndarray, b: jnp.ndarray, axes: Sequence[int]) -> jnp.ndarray:
        return jnp.tensordot(a, b, axes)

    @partial(jax.jit, static_argnames=["dtype"])
    def trace(self, array: jnp.ndarray, dtype=None) -> jnp.ndarray:
        return self.cast(jnp.trace(array, axis1=-1, axis2=-2), dtype)

    def transpose(self, a: jnp.ndarray, perm: Sequence[int] | None = None) -> jnp.ndarray:
        return jnp.transpose(a, perm)

    def zeros(self, shape: Sequence[int], dtype=None) -> jnp.ndarray:
        dtype = dtype or self.float64
        return jnp.zeros(shape, dtype=dtype)

    @partial(jax.jit, static_argnames=["dtype"])
    def zeros_like(self, array: jnp.ndarray, dtype: str = "complex128") -> jnp.ndarray:
        return jnp.zeros_like(array, dtype=dtype)

    def xlogy(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        return jax.scipy.special.xlogy(x, y)

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

    def DefaultEuclideanOptimizer(self):
        return optax.inject_hyperparams(optax.adamw)

    @jax.jit
    def reorder_AB_bargmann(
        self,
        A: jnp.ndarray,
        B: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        r"""In mrmustard.math.numba.compactFock~ dimensions of the Fock representation are ordered like [mode0,mode0,mode1,mode1,...]
        while in mrmustard.physics.bargmann_utils the ordering is [mode0,mode1,...,mode0,mode1,...]. Here we reorder A and B.
        """
        ordering = jnp.arange(2 * A.shape[0] // 2).reshape(2, -1).T.flatten()
        A = self.gather(A, ordering, axis=1)
        A = self.gather(A, ordering)
        B = self.gather(B, ordering, axis=0)
        return A, B

    def hermite_renormalized_unbatched(
        self,
        A: jnp.ndarray,
        b: jnp.ndarray,
        c: jnp.ndarray,
        shape: tuple[int],
        stable: bool = False,
        out: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        if out is not None:
            raise ValueError("The 'out' keyword is not supported in the JAX backend.")
        return hermite_renormalized_unbatched_jax(A, b, c, shape, stable)

    @partial(jax.jit, static_argnames=["shape", "stable"])
    def hermite_renormalized_batched(
        self,
        A: jnp.ndarray,
        b: jnp.ndarray,
        c: jnp.ndarray,
        shape: tuple[int],
        stable: bool = False,
        out: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        if out is not None:
            raise ValueError("The 'out' keyword is not supported in the JAX backend.")
        return hermite_renormalized_batched_jax(A, b, c, shape, stable)

    @partial(jax.jit, static_argnames=["cutoffs"])
    def hermite_renormalized_diagonal(
        self,
        A: jnp.ndarray,
        B: jnp.ndarray,
        C: jnp.ndarray,
        cutoffs: tuple[int],
    ) -> jnp.ndarray:
        r"""First, reorder A and B parameters of Bargmann representation to match conventions in mrmustard.math.numba.compactFock~
        Then, calculate the required renormalized multidimensional Hermite polynomial.
        """
        A, B = self.reorder_AB_bargmann(A, B)
        return self.hermite_renormalized_diagonal_reorderedAB(A, B, C, cutoffs=cutoffs)

    @partial(jax.jit, static_argnames=["cutoffs"])
    def hermite_renormalized_diagonal_reorderedAB(
        self,
        A: jnp.ndarray,
        B: jnp.ndarray,
        C: jnp.ndarray,
        cutoffs: tuple[int],
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
        return jax.pure_callback(
            lambda A, B, C: function(np.asarray(A), np.asarray(B), np.asarray(C))[0],
            jax.ShapeDtypeStruct(cutoffs, jnp.complex128),
            A,
            B,
            C,
        )

    @partial(jax.jit, static_argnames=["cutoffs"])
    def hermite_renormalized_diagonal_batch(
        self,
        A: jnp.ndarray,
        B: jnp.ndarray,
        C: jnp.ndarray,
        cutoffs: tuple[int],
    ) -> jnp.ndarray:
        r"""Same as hermite_renormalized_diagonal but works for a batch of different B's."""
        A, B = self.reorder_AB_bargmann(A, B)
        return self.hermite_renormalized_diagonal_reorderedAB_batch(A, B, C, cutoffs=cutoffs)

    @partial(jax.jit, static_argnames=["cutoffs"])
    def hermite_renormalized_diagonal_reorderedAB_batch(
        self,
        A: jnp.ndarray,
        B: jnp.ndarray,
        C: jnp.ndarray,
        cutoffs: tuple[int],
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
        return jax.pure_callback(
            lambda A, B, C: function(np.asarray(A), np.asarray(B), np.asarray(C))[0],
            jax.ShapeDtypeStruct((*cutoffs, B.shape[1]), jnp.complex128),
            A,
            B,
            C,
        )

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
        return jax.pure_callback(
            lambda A, B, C, max_l2, global_cutoff: function(
                np.asarray(A),
                np.asarray(B),
                np.asarray(C),
                max_l2,
                global_cutoff,
            )[0],
            jax.ShapeDtypeStruct(shape, jnp.complex128),
            A,
            B,
            C,
            max_l2,
            global_cutoff,
        )

    @partial(jax.jit, static_argnames=["output_cutoff", "pnr_cutoffs"])
    def hermite_renormalized_1leftoverMode(self, A, B, C, output_cutoff, pnr_cutoffs):
        A, B = self.reorder_AB_bargmann(A, B)
        cutoffs = (output_cutoff + 1, *tuple(p + 1 for p in pnr_cutoffs))
        return self.hermite_renormalized_1leftoverMode_reorderedAB(A, B, C, cutoffs=cutoffs)

    @partial(jax.jit, static_argnames=["cutoffs"])
    def hermite_renormalized_1leftoverMode_reorderedAB(
        self,
        A: jnp.ndarray,
        B: jnp.ndarray,
        C: jnp.ndarray,
        cutoffs: tuple[int],
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
        return jax.pure_callback(
            lambda A, B, C: function(np.asarray(A), np.asarray(B), np.asarray(C))[0],
            jax.ShapeDtypeStruct((cutoffs[0], *cutoffs), jnp.complex128),
            A,
            B,
            C,
        )

    def displacement(self, x: float, y: float, shape: tuple[int, int], tol: float):
        return displacement_jax(x, y, shape, tol)

    def beamsplitter(self, theta: float, phi: float, shape: tuple[int, int, int, int], method: str):
        return beamsplitter_jax(theta, phi, shape, method)

    def squeezed(self, r: float, phi: float, shape: tuple[int, int]):
        # TODO: implement vjps
        sq_ket = strategies.squeezed(shape, self.asnumpy(r), self.asnumpy(phi))
        return self.astensor(sq_ket, dtype=sq_ket.dtype.name)

    def squeezer(self, r: float, phi: float, shape: tuple[int, int]):
        # TODO: implement vjps
        sq_ket = strategies.squeezer(shape, self.asnumpy(r), self.asnumpy(phi))
        return self.astensor(sq_ket, dtype=sq_ket.dtype.name)


# defining custom pytree nodes
for cls in get_all_subclasses(Ansatz):
    jax.tree_util.register_pytree_node(cls, cls._tree_flatten, cls._tree_unflatten)
jax.tree_util.register_pytree_node(BackendJax, BackendJax._tree_flatten, BackendJax._tree_unflatten)
jax.tree_util.register_pytree_node(Circuit, Circuit._tree_flatten, Circuit._tree_unflatten)
jax.tree_util.register_pytree_node(
    CircuitComponent,
    CircuitComponent._tree_flatten,
    CircuitComponent._tree_unflatten,
)
# register all subclasses of CircuitComponent
for cls in get_all_subclasses(CircuitComponent):
    jax.tree_util.register_pytree_node(cls, cls._tree_flatten, cls._tree_unflatten)
jax.tree_util.register_pytree_node(Constant, Constant._tree_flatten, Constant._tree_unflatten)
jax.tree_util.register_pytree_node(
    ParameterSet,
    ParameterSet._tree_flatten,
    ParameterSet._tree_unflatten,
)
jax.tree_util.register_pytree_node(Variable, Variable._tree_flatten, Variable._tree_unflatten)
