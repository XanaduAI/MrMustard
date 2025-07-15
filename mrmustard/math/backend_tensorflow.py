# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains the tensorflow backend."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from importlib import metadata

import numpy as np
from opt_einsum import contract
from semantic_version import Version

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"


from mrmustard.math.lattice.strategies.compactFock.inputValidation import (
    grad_hermite_multidimensional_1leftoverMode,
    grad_hermite_multidimensional_diagonal,
    hermite_multidimensional_1leftoverMode,
    hermite_multidimensional_diagonal,
    hermite_multidimensional_diagonal_batch,
)

from ..utils.settings import settings
from ..utils.typing import Tensor, Trainable
from .autocast import Autocast
from .backend_base import BackendBase
from .lattice import strategies


class BackendTensorflow(BackendBase):
    r"""
    A base class for backends.
    """

    int32 = tf.int32
    int64 = tf.int64
    float32 = tf.float32
    float64 = tf.float64
    complex64 = tf.complex64
    complex128 = tf.complex128

    def __init__(self):
        tf.experimental.numpy.experimental_enable_numpy_behavior()
        super().__init__(name="tensorflow")

    def __repr__(self) -> str:
        return "BackendTensorflow()"

    def abs(self, array: tf.Tensor) -> tf.Tensor:
        return tf.abs(array)

    def all(self, array: tf.Tensor) -> tf.Tensor:
        return tf.experimental.numpy.all(array)

    def allclose(self, array1: np.array, array2: np.array, atol: float, rtol: float) -> bool:
        return tf.experimental.numpy.allclose(array1, array2, atol=atol, rtol=rtol)

    def angle(self, array: tf.Tensor) -> tf.Tensor:
        return tf.experimental.numpy.angle(array)

    def any(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.reduce_any(array)

    def arange(self, start: int, limit: int | None = None, delta: int = 1, dtype=None) -> tf.Tensor:
        dtype = dtype or self.float64
        return tf.range(start, limit, delta, dtype=dtype)

    def asnumpy(self, tensor: tf.Tensor) -> Tensor:
        return np.array(tensor)

    def assign(self, tensor: tf.Tensor, value: tf.Tensor) -> tf.Tensor:
        tensor.assign(value)
        return tensor

    def astensor(self, array: np.ndarray | tf.Tensor, dtype=None) -> tf.Tensor:
        dtype = dtype or np.array(array).dtype.name
        return tf.cast(tf.convert_to_tensor(array, dtype_hint=dtype), dtype)

    def atleast_nd(self, array: tf.Tensor, n: int, dtype=None) -> tf.Tensor:
        return tf.experimental.numpy.array(array, ndmin=n, dtype=dtype)

    def broadcast_to(self, array: tf.Tensor, shape: tuple[int]) -> tf.Tensor:
        return tf.broadcast_to(array, shape)

    def broadcast_arrays(self, *arrays: list[tf.Tensor]) -> list[tf.Tensor]:
        # TensorFlow doesn't have a direct equivalent to numpy's broadcast_arrays
        # We need to implement it manually
        if not arrays:
            return []

        # Get the broadcasted shape
        shapes = [tf.shape(arr) for arr in arrays]
        broadcasted_shape = shapes[0]
        for shape in shapes[1:]:
            broadcasted_shape = tf.broadcast_dynamic_shape(broadcasted_shape, shape)

        # Broadcast each array to the common shape
        return [tf.broadcast_to(arr, broadcasted_shape) for arr in arrays]

    def cast(self, array: tf.Tensor, dtype=None) -> tf.Tensor:
        if dtype is None:
            return array
        return tf.cast(array, dtype)

    def clip(self, array, a_min, a_max) -> tf.Tensor:
        return tf.clip_by_value(array, a_min, a_max)

    def concat(self, values: Sequence[tf.Tensor], axis: int) -> tf.Tensor:
        if any(tf.rank(v) == 0 for v in values):
            return tf.stack(values, axis)
        return tf.concat(values, axis)

    def conj(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.conj(array)

    @staticmethod
    def constraint_func(bounds: tuple[float | None, float | None]) -> Callable | None:
        bounds = (
            -np.inf if bounds[0] is None else bounds[0],
            np.inf if bounds[1] is None else bounds[1],
        )
        if bounds != (-np.inf, np.inf):

            def constraint(x):
                return tf.clip_by_value(x, bounds[0], bounds[1])

        else:
            constraint = None
        return constraint

    def cos(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.cos(array)

    def cosh(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.cosh(array)

    def det(self, matrix: tf.Tensor) -> tf.Tensor:
        return tf.linalg.det(matrix)

    def diag(self, array: tf.Tensor, k: int = 0) -> tf.Tensor:
        return tf.linalg.diag(array, k=k)

    def diag_part(self, array: tf.Tensor, k: int = 0) -> tf.Tensor:
        return tf.linalg.diag_part(array, k=k)

    @Autocast()
    def einsum(self, string: str, *tensors, optimize: str | bool) -> tf.Tensor:
        return contract(string, *tensors, optimize=optimize, backend="tensorflow")

    def exp(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.exp(array)

    def expand_dims(self, array: tf.Tensor, axis: int) -> tf.Tensor:
        return tf.expand_dims(array, axis)

    def expm(self, matrix: tf.Tensor) -> tf.Tensor:
        return tf.linalg.expm(matrix)

    def eye(self, size: int, dtype=None) -> tf.Tensor:
        dtype = dtype or self.float64
        return tf.eye(size, dtype=dtype)

    def eye_like(self, array: tf.Tensor) -> Tensor:
        return tf.eye(array.shape[-1], dtype=array.dtype)

    def from_backend(self, value) -> bool:
        return isinstance(value, tf.Tensor | tf.Variable)

    def gather(self, array: tf.Tensor, indices: tf.Tensor, axis: int) -> tf.Tensor:
        indices = tf.cast(tf.convert_to_tensor(indices), dtype=tf.int32)
        return tf.gather(array, indices, axis=axis)

    def conditional(
        self,
        cond: tf.Tensor,
        true_fn: Callable,
        false_fn: Callable,
        *args,
    ) -> tf.Tensor:
        if tf.reduce_all(cond):
            return true_fn(*args)
        return false_fn(*args)

    def error_if(self, array: tf.Tensor, condition: tf.Tensor, msg: str):
        if tf.reduce_any(condition):
            raise ValueError(msg)

    def imag(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.imag(array)

    def inv(self, tensor: tf.Tensor) -> tf.Tensor:
        return tf.linalg.inv(tensor)

    def isnan(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.is_nan(array)

    def is_trainable(self, tensor: tf.Tensor) -> bool:
        return isinstance(tensor, tf.Variable)

    def lgamma(self, x: tf.Tensor) -> tf.Tensor:
        return tf.math.lgamma(x)

    def log(self, x: tf.Tensor) -> tf.Tensor:
        return tf.math.log(x)

    @Autocast()
    def matmul(self, *matrices: tf.Tensor) -> tf.Tensor:
        mat = matrices[0]
        for matrix in matrices[1:]:
            mat = tf.matmul(mat, matrix)
        return mat

    @Autocast()
    def matvec(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        return tf.linalg.matvec(a, b)

    def make_complex(self, real: tf.Tensor, imag: tf.Tensor) -> tf.Tensor:
        return tf.complex(real, imag)

    @Autocast()
    def max(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.reduce_max(array)

    @Autocast()
    def maximum(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        return tf.maximum(a, b)

    @Autocast()
    def minimum(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        return tf.minimum(a, b)

    def moveaxis(
        self,
        array: tf.Tensor,
        old: int | Sequence[int],
        new: int | Sequence[int],
    ) -> tf.Tensor:
        return tf.experimental.numpy.moveaxis(array, old, new)

    def new_variable(
        self,
        value,
        bounds: tuple[float | None, float | None] | None,
        name: str,
        dtype=None,
    ):
        bounds = bounds or (None, None)
        dtype = dtype or self.float64
        value = self.astensor(value, dtype)
        return tf.Variable(value, name=name, dtype=dtype, constraint=self.constraint_func(bounds))

    def new_constant(self, value, name: str, dtype=None):
        dtype = dtype or self.float64
        value = self.astensor(value, dtype)
        return tf.constant(value, dtype=dtype, name=name)

    def norm(self, array: tf.Tensor) -> tf.Tensor:
        """Note that the norm preserves the type of array."""
        return tf.linalg.norm(array)

    def ones(self, shape: Sequence[int], dtype=None) -> tf.Tensor:
        dtype = dtype or self.float64
        return tf.ones(shape, dtype=dtype)

    def ones_like(self, array: tf.Tensor) -> tf.Tensor:
        return tf.ones_like(array)

    def infinity_like(self, array: np.ndarray) -> np.ndarray:
        return tf.fill(array.shape, np.inf)

    @Autocast()
    def outer(self, array1: tf.Tensor, array2: tf.Tensor) -> tf.Tensor:
        return self.tensordot(array1, array2, [[], []])

    def pad(
        self,
        array: tf.Tensor,
        paddings: Sequence[tuple[int, int]],
        mode="CONSTANT",
        constant_values=0,
    ) -> tf.Tensor:
        return tf.pad(array, paddings, mode, constant_values)

    @staticmethod
    def pinv(matrix: tf.Tensor) -> tf.Tensor:
        # need to handle complex case on our own
        # https://stackoverflow.com/questions/60025950/tensorflow-pseudo-inverse-doesnt-work-for-complex-matrices
        real_matrix = tf.math.real(matrix)
        imag_matrix = tf.math.imag(matrix)
        r0 = tf.linalg.pinv(real_matrix) @ imag_matrix
        y11 = tf.linalg.pinv(imag_matrix @ r0 + real_matrix)
        y10 = -r0 @ y11
        return tf.cast(tf.complex(y11, y10), dtype=matrix.dtype)

    @Autocast()
    def pow(self, x: tf.Tensor, y: float) -> tf.Tensor:
        return tf.math.pow(x, y)

    def kron(self, tensor1: tf.Tensor, tensor2: tf.Tensor):
        return tf.experimental.numpy.kron(tensor1, tensor2)

    def prod(self, x: tf.Tensor, axis: int | None):
        return tf.math.reduce_prod(x, axis=axis)

    def real(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.real(array)

    def reshape(self, array: tf.Tensor, shape: Sequence[int]) -> tf.Tensor:
        return tf.reshape(array, shape)

    def sin(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.sin(array)

    def sinh(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.sinh(array)

    def solve(self, matrix: tf.Tensor, rhs: tf.Tensor) -> tf.Tensor:
        if len(rhs.shape) == len(matrix.shape) - 1:
            rhs = tf.expand_dims(rhs, -1)
            return tf.linalg.solve(matrix, rhs)[..., 0]
        return tf.linalg.solve(matrix, rhs)

    def sort(self, array: tf.Tensor, axis: int = -1) -> tf.Tensor:
        return tf.sort(array, axis)

    def sqrt(self, x: tf.Tensor, dtype=None) -> tf.Tensor:
        return tf.sqrt(self.cast(x, dtype))

    def stack(self, arrays: tf.Tensor, axis: int = 0) -> tf.Tensor:
        return tf.stack(arrays, axis=axis)

    def sum(self, array: tf.Tensor, axis: int | tuple[int] | None = None):
        return tf.reduce_sum(array, axis)

    def swapaxes(self, array: tf.Tensor, axis1: int, axis2: int) -> tf.Tensor:
        return tf.experimental.numpy.swapaxes(array, axis1, axis2)

    @Autocast()
    def tensordot(self, a: tf.Tensor, b: tf.Tensor, axes: list[int]) -> tf.Tensor:
        return tf.tensordot(a, b, axes)

    def tile(self, array: tf.Tensor, repeats: Sequence[int]) -> tf.Tensor:
        return tf.tile(array, repeats)

    def trace(self, array: tf.Tensor, dtype=None) -> tf.Tensor:
        return self.cast(tf.linalg.trace(array), dtype)

    def transpose(self, a: tf.Tensor, perm: Sequence[int] | None = None) -> tf.Tensor:
        return tf.transpose(a, perm)

    def update_tensor(self, tensor: tf.Tensor, indices: tf.Tensor, values: tf.Tensor) -> tf.Tensor:
        indices = tf.convert_to_tensor([indices], dtype=tf.int32)
        updates = tf.convert_to_tensor([values], dtype=tf.complex64)
        return tf.tensor_scatter_nd_update(tensor, indices, updates)

    def update_add_tensor(
        self,
        tensor: tf.Tensor,
        indices: tf.Tensor,
        values: tf.Tensor,
    ) -> tf.Tensor:
        return tf.tensor_scatter_nd_add(tensor, indices, values)

    def zeros(self, shape: Sequence[int], dtype=None) -> tf.Tensor:
        dtype = dtype or self.float64
        return tf.zeros(shape, dtype=dtype)

    def zeros_like(self, array: tf.Tensor) -> tf.Tensor:
        return tf.zeros_like(array)

    def map_fn(self, func, elements):
        return tf.map_fn(func, elements)

    @staticmethod
    def eigh(tensor: tf.Tensor) -> Tensor:
        return tf.linalg.eigh(tensor)

    @staticmethod
    def eigvals(tensor: tf.Tensor) -> Tensor:
        return tf.linalg.eigvals(tensor)

    @staticmethod
    def xlogy(x: tf.Tensor, y: tf.Tensor) -> Tensor:
        return tf.math.xlogy(x, y)

    def sqrtm(self, tensor: tf.Tensor, dtype, rtol=1e-05, atol=1e-08) -> Tensor:
        # The sqrtm function has issues with matrices that are close to zero, hence we branch
        if np.allclose(tensor, 0, rtol=rtol, atol=atol):
            ret = self.zeros_like(tensor)
        else:
            ret = tf.linalg.sqrtm(tensor)

        if dtype is None:
            return self.cast(ret, self.complex128)
        return self.cast(ret, dtype)

    def DefaultEuclideanOptimizer(self) -> tf.keras.optimizers.legacy.Optimizer:
        if Version(metadata.distribution("tensorflow").version) > Version("2.15.0"):
            os.environ["TF_USE_LEGACY_KERAS"] = "True"
        return tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

    def value_and_gradients(
        self,
        cost_fn: Callable,
        parameters: list[Trainable],
    ) -> tuple[tf.Tensor, list[tf.Tensor]]:
        r"""Computes the loss and gradients of the given cost function.

        Args:
            cost_fn (Callable with no args): The cost function.
            parameters (List[Trainable]): The parameters to optimize.

        Returns:
            tuple(Tensor, List[Tensor]): the loss and the gradients
        """
        with tf.GradientTape() as tape:
            loss = cost_fn()
        gradients = tape.gradient(loss, parameters)
        return loss, gradients

    @tf.custom_gradient
    def hermite_renormalized_unbatched(
        self,
        A: tf.Tensor,
        b: tf.Tensor,
        c: tf.Tensor,
        shape: tuple[int],
        stable: bool,
        out: tf.Tensor | None = None,
    ) -> tuple[tf.Tensor, Callable]:
        A, b, c = self.asnumpy(A), self.asnumpy(b), self.asnumpy(c)
        if out is not None:
            raise ValueError("'out' keyword is not supported in the TensorFlow backend")
        if stable:
            G = strategies.stable_numba(tuple(shape), A, b, c, None)
        else:
            G = strategies.vanilla_numba(tuple(shape), A, b, c, None)

        def grad(dLdGconj):
            dLdA, dLdB, dLdC = strategies.vanilla_vjp_numba(G, c, np.conj(dLdGconj))
            return self.conj(dLdA), self.conj(dLdB), self.conj(dLdC)

        return G, grad

    @tf.custom_gradient
    def hermite_renormalized_batched(
        self,
        A: tf.Tensor,
        b: tf.Tensor,
        c: tf.Tensor,
        shape: tuple[int],
        stable: bool,
        out: tf.Tensor | None = None,
    ) -> tf.Tensor:
        A, b, c = self.asnumpy(A), self.asnumpy(b), self.asnumpy(c)
        if out is not None:
            raise ValueError("'out' keyword is not supported in the TensorFlow backend")
        G = strategies.vanilla_batch_numba(tuple(shape), A, b, c, stable, None)

        def grad(dLdGconj):
            dLdA, dLdB, dLdC = strategies.vanilla_batch_vjp_numba(G, c, np.conj(dLdGconj))
            return self.conj(dLdA), self.conj(dLdB), self.conj(dLdC)

        return G, grad

    @tf.custom_gradient
    def hermite_renormalized_binomial(
        self,
        A: tf.Tensor,
        B: tf.Tensor,
        C: tf.Tensor,
        shape: tuple[int],
        max_l2: float | None,
        global_cutoff: int | None,
    ) -> tf.Tensor:
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
        _A, _B, _C = self.asnumpy(A), self.asnumpy(B), self.asnumpy(C)
        G, _ = strategies.binomial(
            tuple(shape),
            _A,
            _B,
            _C,
            max_l2=max_l2 or settings.AUTOSHAPE_PROBABILITY,
            global_cutoff=global_cutoff or sum(shape) - len(shape) + 1,
        )

        def grad(dLdGconj):
            dLdA, dLdB, dLdC = strategies.vanilla_vjp(G, _C, np.conj(dLdGconj))
            return self.conj(dLdA), self.conj(dLdB), self.conj(dLdC)

        return G, grad

    def reorder_AB_bargmann(self, A: tf.Tensor, B: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        r"""In mrmustard.math.compactFock.compactFock~ dimensions of the Fock representation are ordered like [mode0,mode0,mode1,mode1,...]
        while in mrmustard.physics.bargmann_utils the ordering is [mode0,mode1,...,mode0,mode1,...]. Here we reorder A and B.
        """
        ordering = (
            np.arange(2 * A.shape[0] // 2).reshape(2, -1).T.flatten()
        )  # ordering is [0,2,4,...,1,3,5,...]
        A = tf.gather(A, ordering, axis=1)
        A = tf.gather(A, ordering)
        B = tf.gather(B, ordering, axis=0)
        return A, B

    def hermite_renormalized_diagonal(
        self,
        A: tf.Tensor,
        B: tf.Tensor,
        C: tf.Tensor,
        cutoffs: tuple[int],
    ) -> tf.Tensor:
        A, B = self.reorder_AB_bargmann(A, B)
        return self.hermite_renormalized_diagonal_reorderedAB(A, B, C, cutoffs=cutoffs)

    @tf.custom_gradient
    def hermite_renormalized_diagonal_reorderedAB(
        self,
        A: tf.Tensor,
        B: tf.Tensor,
        C: tf.Tensor,
        cutoffs: tuple[int],
    ) -> tf.Tensor:
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
        A, B, C = self.asnumpy(A), self.asnumpy(B), self.asnumpy(C)

        poly0, poly2, poly1010, poly1001, poly1 = tf.numpy_function(
            hermite_multidimensional_diagonal,
            [A, B, C, cutoffs],
            [A.dtype] * 5,
        )

        def grad(dLdpoly):
            dpoly_dC, dpoly_dA, dpoly_dB = tf.numpy_function(
                grad_hermite_multidimensional_diagonal,
                [A, B, C.item(), poly0, poly2, poly1010, poly1001, poly1],
                [poly0.dtype] * 3,
            )

            ax = tuple(range(dLdpoly.ndim))
            dLdA = self.sum(dLdpoly[..., None, None] * self.conj(dpoly_dA), axis=ax)
            dLdB = self.sum(dLdpoly[..., None] * self.conj(dpoly_dB), axis=ax)
            dLdC = self.sum(dLdpoly * self.conj(dpoly_dC), axis=ax)
            return dLdA, dLdB, dLdC

        return poly0, grad

    def hermite_renormalized_diagonal_batch(
        self,
        A: tf.Tensor,
        B: tf.Tensor,
        C: tf.Tensor,
        cutoffs: tuple[int],
    ) -> tf.Tensor:
        r"""Same as hermite_renormalized_diagonal but works for a batch of different B's."""
        A, B = self.reorder_AB_bargmann(A, B)
        return self.hermite_renormalized_diagonal_reorderedAB_batch(A, B, C, cutoffs=cutoffs)

    def hermite_renormalized_diagonal_reorderedAB_batch(
        self,
        A: tf.Tensor,
        B: tf.Tensor,
        C: tf.Tensor,
        cutoffs: tuple[int],
    ) -> tf.Tensor:
        r"""Same as hermite_renormalized_diagonal_reorderedAB but works for a batch of different B's.

        Args:
            A: The A matrix.
            B: The B vectors.
            C: The C scalar.
            cutoffs: upper boundary of photon numbers in each mode

        Returns:
            The renormalized Hermite polynomial from different B values.
        """
        A, B, C = self.asnumpy(A), self.asnumpy(B), self.asnumpy(C)

        poly0, _, _, _, _ = tf.numpy_function(
            hermite_multidimensional_diagonal_batch,
            [A, B, C, cutoffs],
            [A.dtype] * 5,
        )

        return poly0

    def hermite_renormalized_1leftoverMode(
        self,
        A: tf.Tensor,
        b: tf.Tensor,
        c: tf.Tensor,
        output_cutoff: int,
        pnr_cutoffs: tuple[int, ...],
    ) -> tf.Tensor:
        A, b = self.reorder_AB_bargmann(A, b)
        cutoffs = (output_cutoff + 1, *tuple(p + 1 for p in pnr_cutoffs))
        return self.hermite_renormalized_1leftoverMode_reorderedAB(A, b, c, cutoffs=cutoffs)

    @tf.custom_gradient
    def hermite_renormalized_1leftoverMode_reorderedAB(
        self,
        A: tf.Tensor,
        B: tf.Tensor,
        C: tf.Tensor,
        cutoffs: tuple[int, ...],
    ) -> tf.Tensor:
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
        A, B, C = self.asnumpy(A), self.asnumpy(B), self.asnumpy(C)
        poly0, poly2, poly1010, poly1001, poly1 = tf.numpy_function(
            hermite_multidimensional_1leftoverMode,
            [A, B, C.item(), cutoffs],
            [A.dtype] * 5,
        )

        def grad(dLdpoly):
            dpoly_dC, dpoly_dA, dpoly_dB = tf.numpy_function(
                grad_hermite_multidimensional_1leftoverMode,
                [A, B, C, poly0, poly2, poly1010, poly1001, poly1],
                [poly0.dtype] * 3,
            )

            ax = tuple(range(dLdpoly.ndim))
            dLdA = self.sum(dLdpoly[..., None, None] * self.conj(dpoly_dA), axis=ax)
            dLdB = self.sum(dLdpoly[..., None] * self.conj(dpoly_dB), axis=ax)
            dLdC = self.sum(dLdpoly * self.conj(dpoly_dC), axis=ax)
            return dLdA, dLdB, dLdC

        return poly0, grad

    @tf.custom_gradient
    def displacement(self, x: float, y: float, shape: tuple[int, ...], tol: float):
        alpha = self.asnumpy(x) + 1j * self.asnumpy(y)
        if np.sqrt(x * x + y * y) > tol:
            gate = strategies.displacement(tuple(shape), alpha)
        else:
            gate = self.eye(max(shape), dtype="complex128")[: shape[0], : shape[1]]
        ret = self.astensor(gate, dtype=gate.dtype.name)

        def grad(dL_dDc):
            dD_da, dD_dac = strategies.jacobian_displacement(self.asnumpy(gate), alpha)
            dL_dac = np.sum(np.conj(dL_dDc) * dD_dac + dL_dDc * np.conj(dD_da))
            dLdx = 2 * np.real(dL_dac)
            dLdy = 2 * np.imag(dL_dac)
            return (
                self.astensor(dLdx, dtype=x.dtype),
                self.astensor(dLdy, dtype=y.dtype),
            )

        return ret, grad

    @tf.custom_gradient
    def beamsplitter(self, theta: float, phi: float, shape: tuple[int, int, int, int], method: str):
        t, s = self.asnumpy(theta), self.asnumpy(phi)
        if method == "vanilla":
            bs_unitary = strategies.beamsplitter(shape, t, s)
        elif method == "schwinger":
            bs_unitary = strategies.beamsplitter_schwinger(shape, t, s)
        elif method == "stable":
            bs_unitary = strategies.stable_beamsplitter(shape, t, s)

        ret = self.astensor(bs_unitary, dtype=bs_unitary.dtype.name)

        def vjp(dLdGc):
            dtheta, dphi = strategies.beamsplitter_vjp(
                self.asnumpy(bs_unitary),
                self.asnumpy(self.conj(dLdGc)),
                self.asnumpy(theta),
                self.asnumpy(phi),
            )
            return (
                self.astensor(dtheta, dtype=theta.dtype),
                self.astensor(dphi, dtype=phi.dtype),
            )

        return ret, vjp

    @tf.custom_gradient
    def squeezed(self, r: float, phi: float, shape: tuple[int, int]):
        sq_ket = strategies.squeezed(shape, self.asnumpy(r), self.asnumpy(phi))
        ret = self.astensor(sq_ket, dtype=sq_ket.dtype.name)

        def vjp(dLdGc):
            dr, dphi = strategies.squeezed_vjp(
                self.asnumpy(sq_ket),
                self.asnumpy(self.conj(dLdGc)),
                self.asnumpy(r),
                self.asnumpy(phi),
            )
            return self.astensor(dr, dtype=r.dtype), self.astensor(dphi, phi.dtype)

        return ret, vjp

    @tf.custom_gradient
    def squeezer(self, r: float, phi: float, shape: tuple[int, int]):
        sq_unitary = strategies.squeezer(shape, self.asnumpy(r), self.asnumpy(phi))
        ret = self.astensor(sq_unitary, dtype=sq_unitary.dtype.name)

        def vjp(dLdGc):
            dr, dphi = strategies.squeezer_vjp(
                self.asnumpy(sq_unitary),
                self.asnumpy(self.conj(dLdGc)),
                self.asnumpy(r),
                self.asnumpy(phi),
            )
            return self.astensor(dr, dtype=r.dtype), self.astensor(dphi, phi.dtype)

        return ret, vjp
