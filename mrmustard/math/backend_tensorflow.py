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

# pylint: disable = missing-function-docstring, missing-class-docstring, wrong-import-position

from __future__ import annotations
from typing import Callable, Sequence

from importlib import metadata
import os

import numpy as np
from semantic_version import Version
import tensorflow_probability as tfp

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


# pylint: disable=too-many-public-methods
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

    def arange(self, start: int, limit: int = None, delta: int = 1, dtype=None) -> tf.Tensor:
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

    def block_diag(self, mat1: tf.Tensor, mat2: tf.Tensor) -> tf.Tensor:
        Za = self.zeros((mat1.shape[-2], mat2.shape[-1]), dtype=mat1.dtype)
        Zb = self.zeros((mat2.shape[-2], mat1.shape[-1]), dtype=mat1.dtype)
        return self.concat(
            [self.concat([mat1, Za], axis=-1), self.concat([Zb, mat2], axis=-1)],
            axis=-2,
        )

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

    def boolean_mask(self, tensor: tf.Tensor, mask: tf.Tensor) -> Tensor:
        return tf.boolean_mask(tensor, mask)

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

    def constraint_func(self, bounds: tuple[float | None, float | None]) -> Callable | None:
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

    # pylint: disable=arguments-differ
    @Autocast()
    def convolution(
        self,
        array: tf.Tensor,
        filters: tf.Tensor,
        padding: str | None = None,
        data_format="NWC",
    ) -> tf.Tensor:
        padding = padding or "VALID"
        return tf.nn.convolution(array, filters=filters, padding=padding, data_format=data_format)

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
        if optimize is False:
            optimize = "greedy"
        return tf.einsum(string, *tensors, optimize=optimize)

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
        return isinstance(value, (tf.Tensor, tf.Variable))

    def gather(self, array: tf.Tensor, indices: tf.Tensor, axis: int) -> tf.Tensor:
        indices = tf.cast(tf.convert_to_tensor(indices), dtype=tf.int32)
        return tf.gather(array, indices, axis=axis)

    def conditional(
        self, cond: tf.Tensor, true_fn: Callable, false_fn: Callable, *args
    ) -> tf.Tensor:
        if tf.reduce_all(cond):
            return true_fn(*args)
        else:
            return false_fn(*args)

    def error_if(
        self, array: tf.Tensor, condition: tf.Tensor, msg: str
    ):  # pylint: disable=unused-argument
        if tf.reduce_any(condition):
            raise ValueError(msg)

    def imag(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.imag(array)

    def inv(self, tensor: tf.Tensor) -> tf.Tensor:
        return tf.linalg.inv(tensor)

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
    def maximum(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        return tf.maximum(a, b)

    @Autocast()
    def minimum(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        return tf.minimum(a, b)

    def moveaxis(
        self, array: tf.Tensor, old: int | Sequence[int], new: int | Sequence[int]
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
        return tf.tensordot(array1, array2, [[], []])

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
        return tf.linalg.pinv(matrix)

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

    def repeat(self, array: tf.Tensor, repeats: int, axis: int = None) -> tf.Tensor:
        return tf.repeat(array, repeats, axis=axis)

    def round(self, array: tf.Tensor, decimals: int = 0) -> tf.Tensor:
        return tf.round(10**decimals * array) / 10**decimals

    def set_diag(self, array: tf.Tensor, diag: tf.Tensor, k: int) -> tf.Tensor:
        return tf.linalg.set_diag(array, diag, k=k)

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

    @Autocast()
    def tensordot(self, a: tf.Tensor, b: tf.Tensor, axes: list[int]) -> tf.Tensor:
        return tf.tensordot(a, b, axes)

    def tile(self, array: tf.Tensor, repeats: Sequence[int]) -> tf.Tensor:
        return tf.tile(array, repeats)

    def trace(self, array: tf.Tensor, dtype=None) -> tf.Tensor:
        return self.cast(tf.linalg.trace(array), dtype)

    def transpose(self, a: tf.Tensor, perm: Sequence[int] | None = None) -> tf.Tensor:
        return tf.transpose(a, perm)

    @Autocast()
    def update_tensor(self, tensor: tf.Tensor, indices: tf.Tensor, values: tf.Tensor) -> tf.Tensor:
        return tf.tensor_scatter_nd_update(tensor, indices, values)

    @Autocast()
    def update_add_tensor(
        self, tensor: tf.Tensor, indices: tf.Tensor, values: tf.Tensor
    ) -> tf.Tensor:
        return tf.tensor_scatter_nd_add(tensor, indices, values)

    def zeros(self, shape: Sequence[int], dtype=None) -> tf.Tensor:
        dtype = dtype or self.float64
        return tf.zeros(shape, dtype=dtype)

    def zeros_like(self, array: tf.Tensor) -> tf.Tensor:
        return tf.zeros_like(array)

    def map_fn(self, func, elements):
        return tf.map_fn(func, elements)

    def squeeze(self, tensor, axis=None):
        return tf.squeeze(tensor, axis=axis or [])

    def cholesky(self, input: Tensor):
        return tf.linalg.cholesky(input)

    def Categorical(self, probs: Tensor, name: str):
        return tfp.distributions.Categorical(probs=probs, name=name)

    def MultivariateNormalTriL(self, loc: Tensor, scale_tril: Tensor):
        return tfp.distributions.MultivariateNormalTriL(loc=loc, scale_tril=scale_tril)

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

    # ~~~~~~~~~~~~~~~~~
    # Special functions
    # ~~~~~~~~~~~~~~~~~

    def DefaultEuclideanOptimizer(self) -> tf.keras.optimizers.legacy.Optimizer:
        if Version(metadata.distribution("tensorflow").version) > Version("2.15.0"):
            os.environ["TF_USE_LEGACY_KERAS"] = "True"
        return tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

    def value_and_gradients(
        self, cost_fn: Callable, parameters: list[Trainable]
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
        self, A: tf.Tensor, B: tf.Tensor, C: tf.Tensor, cutoffs: tuple[int]
    ) -> tf.Tensor:
        A, B = self.reorder_AB_bargmann(A, B)
        return self.hermite_renormalized_diagonal_reorderedAB(A, B, C, cutoffs=cutoffs)

    @tf.custom_gradient
    def hermite_renormalized_diagonal_reorderedAB(
        self, A: tf.Tensor, B: tf.Tensor, C: tf.Tensor, cutoffs: tuple[int]
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
            hermite_multidimensional_diagonal, [A, B, C, cutoffs], [A.dtype] * 5
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
        self, A: tf.Tensor, B: tf.Tensor, C: tf.Tensor, cutoffs: tuple[int]
    ) -> tf.Tensor:
        r"""Same as hermite_renormalized_diagonal but works for a batch of different B's."""
        A, B = self.reorder_AB_bargmann(A, B)
        return self.hermite_renormalized_diagonal_reorderedAB_batch(A, B, C, cutoffs=cutoffs)

    def hermite_renormalized_diagonal_reorderedAB_batch(
        self, A: tf.Tensor, B: tf.Tensor, C: tf.Tensor, cutoffs: tuple[int]
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
            hermite_multidimensional_diagonal_batch, [A, B, C, cutoffs], [A.dtype] * 5
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
        cutoffs = (output_cutoff + 1,) + tuple(p + 1 for p in pnr_cutoffs)
        return self.hermite_renormalized_1leftoverMode_reorderedAB(A, b, c, cutoffs=cutoffs)

    @tf.custom_gradient
    def hermite_renormalized_1leftoverMode_reorderedAB(
        self, A: tf.Tensor, B: tf.Tensor, C: tf.Tensor, cutoffs: tuple[int, ...]
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
    def getitem(tensor, *, key):
        """A differentiable pure equivalent of numpy's ``value = tensor[key]``."""
        value = np.array(tensor)[key]

        def grad(dy):
            dL_dtensor = np.zeros_like(tensor)
            dL_dtensor[key] = dy
            return dL_dtensor

        return value, grad

    @tf.custom_gradient
    def setitem(tensor, value, *, key):
        """A differentiable pure equivalent of numpy's ``tensor[key] = value``."""
        _tensor = np.array(tensor)
        value = np.array(value)
        _tensor[key] = value

        def grad(dy):
            dL_dtensor = np.array(dy)
            dL_dtensor[key] = 0.0
            # unbroadcasting the gradient
            implicit_broadcast = list(range(_tensor.ndim - value.ndim))
            explicit_broadcast = [
                _tensor.ndim - value.ndim + j for j in range(value.ndim) if value.shape[j] == 1
            ]
            dL_dvalue = np.sum(
                np.array(dy)[key], axis=tuple(implicit_broadcast + explicit_broadcast)
            )
            dL_dvalue = np.expand_dims(
                dL_dvalue, [i - len(implicit_broadcast) for i in explicit_broadcast]
            )
            return dL_dtensor, dL_dvalue

        return _tensor, grad
