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

"""This module contains the Tensorflow implementation of the :class:`Math` interface."""

from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from mrmustard import settings
from mrmustard.math.autocast import Autocast
from mrmustard.math.lattice import strategies
from mrmustard.math.numba.compactFock_inputValidation import (
    grad_hermite_multidimensional_1leftoverMode,
    grad_hermite_multidimensional_diagonal,
    hermite_multidimensional_1leftoverMode,
    hermite_multidimensional_diagonal,
)
from mrmustard.typing import Tensor, Trainable

from .math_interface import MathInterface


# pylint: disable=too-many-public-methods,no-self-argument,arguments-differ
class TFMath(MathInterface):
    r"""Tensorflow implemantion of the :class:`Math` interface."""

    float64 = tf.float64
    float32 = tf.float32
    complex64 = tf.complex64
    complex128 = tf.complex128

    def __getattr__(self, name):
        return getattr(tf, name)

    # ~~~~~~~~~
    # Basic ops
    # ~~~~~~~~~

    def abs(self, array: tf.Tensor) -> tf.Tensor:
        return tf.abs(array)

    def any(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.reduce_any(array)

    def arange(self, start: int, limit: int = None, delta: int = 1, dtype=tf.float64) -> tf.Tensor:
        return tf.range(start, limit, delta, dtype=dtype)

    def asnumpy(self, tensor: tf.Tensor) -> Tensor:
        return np.array(tensor)

    def assign(self, tensor: tf.Tensor, value: tf.Tensor) -> tf.Tensor:
        tensor.assign(value)
        return tensor

    def astensor(self, array: Union[np.ndarray, tf.Tensor], dtype=None) -> tf.Tensor:
        return tf.convert_to_tensor(array, dtype=dtype)

    def atleast_1d(self, array: tf.Tensor, dtype=None) -> tf.Tensor:
        return self.cast(tf.reshape(array, [-1]), dtype)

    def cast(self, array: tf.Tensor, dtype=None) -> tf.Tensor:
        if dtype is None:
            return array
        return tf.cast(array, dtype)

    def clip(self, array, a_min, a_max) -> tf.Tensor:
        return tf.clip_by_value(array, a_min, a_max)

    def concat(self, values: Sequence[tf.Tensor], axis: int) -> tf.Tensor:
        if any(tf.rank(v) == 0 for v in values):
            return tf.stack(values, axis)
        else:
            return tf.concat(values, axis)

    def conj(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.conj(array)

    def constraint_func(
        self, bounds: Tuple[Optional[float], Optional[float]]
    ) -> Optional[Callable]:
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
        strides: Optional[List[int]] = None,
        padding="VALID",
        data_format="NWC",
        dilations: Optional[List[int]] = None,
    ) -> tf.Tensor:
        return tf.nn.convolution(array, filters, strides, padding, data_format, dilations)

    def cos(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.cos(array)

    def cosh(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.cosh(array)

    def atan2(self, y: tf.Tensor, x: tf.Tensor) -> tf.Tensor:
        return tf.math.atan2(y, x)

    def make_complex(self, real: tf.Tensor, imag: tf.Tensor) -> tf.Tensor:
        return tf.complex(real, imag)

    def det(self, matrix: tf.Tensor) -> tf.Tensor:
        return tf.linalg.det(matrix)

    def diag(self, array: tf.Tensor, k: int = 0) -> tf.Tensor:
        return tf.linalg.diag(array, k=k)

    def diag_part(self, array: tf.Tensor, k: int = 0) -> tf.Tensor:
        return tf.linalg.diag_part(array, k=k)

    def einsum(self, string: str, *tensors) -> tf.Tensor:
        if type(string) is str:
            return tf.einsum(string, *tensors)
        return None  # provide same functionality as numpy.einsum or upgrade to opt_einsum

    def exp(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.exp(array)

    def expand_dims(self, array: tf.Tensor, axis: int) -> tf.Tensor:
        return tf.expand_dims(array, axis)

    def expm(self, matrix: tf.Tensor) -> tf.Tensor:
        return tf.linalg.expm(matrix)

    def eye(self, size: int, dtype=tf.float64) -> tf.Tensor:
        return tf.eye(size, dtype=dtype)

    def eye_like(self, array: tf.Tensor) -> Tensor:
        return tf.eye(array.shape[-1], dtype=array.dtype)

    def from_backend(self, value) -> bool:
        return isinstance(value, (tf.Tensor, tf.Variable))

    def gather(self, array: tf.Tensor, indices: tf.Tensor, axis: int = None) -> tf.Tensor:
        return tf.gather(array, indices, axis=axis)

    def hash_tensor(self, tensor: tf.Tensor) -> int:
        try:
            REF = tensor.ref()
        except AttributeError as e:
            raise TypeError("Cannot hash tensor") from e
        return hash(REF)

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
    def matmul(
        self,
        a: tf.Tensor,
        b: tf.Tensor,
        transpose_a=False,
        transpose_b=False,
        adjoint_a=False,
        adjoint_b=False,
    ) -> tf.Tensor:
        return tf.linalg.matmul(a, b, transpose_a, transpose_b, adjoint_a, adjoint_b)

    @Autocast()
    def matvec(self, a: tf.Tensor, b: tf.Tensor, transpose_a=False, adjoint_a=False) -> tf.Tensor:
        return tf.linalg.matvec(a, b, transpose_a, adjoint_a)

    @Autocast()
    def maximum(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        return tf.maximum(a, b)

    @Autocast()
    def minimum(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        return tf.minimum(a, b)

    def new_variable(
        self,
        value,
        bounds: Union[Tuple[Optional[float], Optional[float]], None],
        name: str,
        dtype=tf.float64,
    ):
        bounds = bounds or (None, None)
        value = self.convert_to_tensor(value, dtype)
        return tf.Variable(value, name=name, dtype=dtype, constraint=self.constraint_func(bounds))

    def new_constant(self, value, name: str, dtype=tf.float64):
        value = self.convert_to_tensor(value, dtype)
        return tf.constant(value, dtype=dtype, name=name)

    def norm(self, array: tf.Tensor) -> tf.Tensor:
        """Note that the norm preserves the type of array."""
        return tf.linalg.norm(array)

    def ones(self, shape: Sequence[int], dtype=tf.float64) -> tf.Tensor:
        return tf.ones(shape, dtype=dtype)

    def ones_like(self, array: tf.Tensor) -> tf.Tensor:
        return tf.ones_like(array)

    @Autocast()
    def outer(self, array1: tf.Tensor, array2: tf.Tensor) -> tf.Tensor:
        return tf.tensordot(array1, array2, [[], []])

    def pad(
        self,
        array: tf.Tensor,
        paddings: Sequence[Tuple[int, int]],
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

    def real(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.real(array)

    def reshape(self, array: tf.Tensor, shape: Sequence[int]) -> tf.Tensor:
        return tf.reshape(array, shape)

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

    def sqrt(self, x: tf.Tensor, dtype=None) -> tf.Tensor:
        return tf.sqrt(self.cast(x, dtype))

    def sum(self, array: tf.Tensor, axes: Sequence[int] = None):
        return tf.reduce_sum(array, axes)

    @Autocast()
    def tensordot(self, a: tf.Tensor, b: tf.Tensor, axes: List[int]) -> tf.Tensor:
        return tf.tensordot(a, b, axes)

    def tile(self, array: tf.Tensor, repeats: Sequence[int]) -> tf.Tensor:
        return tf.tile(array, repeats)

    def trace(self, array: tf.Tensor, dtype=None) -> tf.Tensor:
        return self.cast(tf.linalg.trace(array), dtype)

    def transpose(self, a: tf.Tensor, perm: Sequence[int] = None) -> tf.Tensor:
        if a is None:
            return None  # TODO: remove and address None inputs where tranpose is used
        return tf.transpose(a, perm)

    @Autocast()
    def update_tensor(self, tensor: tf.Tensor, indices: tf.Tensor, values: tf.Tensor):
        return tf.tensor_scatter_nd_update(tensor, indices, values)

    @Autocast()
    def update_add_tensor(self, tensor: tf.Tensor, indices: tf.Tensor, values: tf.Tensor):
        return tf.tensor_scatter_nd_add(tensor, indices, values)

    def unique_tensors(self, lst: List[Tensor]) -> List[Tensor]:
        hash_dict = {}
        for tensor in lst:
            try:
                if (hash := self.hash_tensor(tensor)) not in hash_dict:
                    hash_dict[hash] = tensor
            except TypeError:
                continue
        return list(hash_dict.values())

    def zeros(self, shape: Sequence[int], dtype=tf.float64) -> tf.Tensor:
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

    # ~~~~~~~~~~~~~~~~~
    # Special functions
    # ~~~~~~~~~~~~~~~~~

    # TODO: is a wrapper class better?
    @staticmethod
    def DefaultEuclideanOptimizer() -> tf.keras.optimizers.legacy.Optimizer:
        r"""Default optimizer for the Euclidean parameters."""
        return tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

    def value_and_gradients(
        self, cost_fn: Callable, parameters: List[Trainable]
    ) -> Tuple[tf.Tensor, List[tf.Tensor]]:
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
    def hermite_renormalized(
        self, A: tf.Tensor, B: tf.Tensor, C: tf.Tensor, shape: Tuple[int]
    ) -> tf.Tensor:
        r"""Renormalized multidimensional Hermite polynomial given by the "exponential" Taylor
        series of :math:`exp(C + Bx + 1/2*Ax^2)` at zero, where the series has :math:`sqrt(n!)`
        at the denominator rather than :math:`n!`. It computes all the amplitudes within the
        tensor of given shape.

        Args:
            A: The A matrix.
            B: The B vector.
            C: The C scalar.
            shape: The shape of the final tensor.

        Returns:
            The renormalized Hermite polynomial of given shape.
        """
        _A, _B, _C = self.asnumpy(A), self.asnumpy(B), self.asnumpy(C)
        G = strategies.vanilla(tuple(shape), _A, _B, _C)

        def grad(dLdGconj):
            dLdA, dLdB, dLdC = strategies.vanilla_vjp(G, _C, np.conj(dLdGconj))
            return self.conj(dLdA), self.conj(dLdB), self.conj(dLdC)

        return G, grad

    @tf.custom_gradient
    def hermite_renormalized_binomial(
        self,
        A: tf.Tensor,
        B: tf.Tensor,
        C: tf.Tensor,
        shape: Tuple[int],
        max_l2: Optional[float],
        global_cutoff: Optional[int],
    ) -> tf.Tensor:
        r"""Renormalized multidimensional Hermite polynomial given by the "exponential" Taylor
        series of :math:`exp(C + Bx + 1/2*Ax^2)` at zero, where the series has :math:`sqrt(n!)`
        at the denominator rather than :math:`n!`. The computation fills a tensor of given shape
        up to a given L2 norm or global cutoff, whichever applies first. The max_l2 value, if
        not provided, is set to the default value of the AUTOCUTOFF_PROBABILITY setting.

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
            max_l2=max_l2 or settings.AUTOCUTOFF_PROBABILITY,
            global_cutoff=global_cutoff or sum(shape) - len(shape) + 1,
        )

        def grad(dLdGconj):
            dLdA, dLdB, dLdC = strategies.vanilla_vjp(G, _C, np.conj(dLdGconj))
            return self.conj(dLdA), self.conj(dLdB), self.conj(dLdC)

        return G, grad

    def reorder_AB_bargmann(self, A: tf.Tensor, B: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        r"""In mrmustard.math.numba.compactFock~ dimensions of the Fock representation are ordered like [mode0,mode0,mode1,mode1,...]
        while in mrmustard.physics.bargmann the ordering is [mode0,mode1,...,mode0,mode1,...]. Here we reorder A and B.
        """
        ordering = np.arange(2 * A.shape[0] // 2).reshape(2, -1).T.flatten()
        A = tf.gather(A, ordering, axis=1)
        A = tf.gather(A, ordering)
        B = tf.gather(B, ordering)
        return A, B

    def hermite_renormalized_diagonal(
        self, A: tf.Tensor, B: tf.Tensor, C: tf.Tensor, cutoffs: Tuple[int]
    ) -> tf.Tensor:
        r"""First, reorder A and B parameters of Bargmann representation to match conventions in mrmustard.math.numba.compactFock~
        Then, calculate the required renormalized multidimensional Hermite polynomial.
        """
        A, B = self.reorder_AB_bargmann(A, B)
        return self.hermite_renormalized_diagonal_reorderedAB(A, B, C, cutoffs=cutoffs)

    @tf.custom_gradient
    def hermite_renormalized_diagonal_reorderedAB(
        self, A: tf.Tensor, B: tf.Tensor, C: tf.Tensor, cutoffs: Tuple[int]
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
        poly0, poly2, poly1010, poly1001, poly1 = tf.numpy_function(
            hermite_multidimensional_diagonal, [A, B, C, cutoffs], [A.dtype] * 5
        )

        def grad(dLdpoly):
            dpoly_dC, dpoly_dA, dpoly_dB = tf.numpy_function(
                grad_hermite_multidimensional_diagonal,
                [A, B, C, poly0, poly2, poly1010, poly1001, poly1],
                [poly0.dtype] * 3,
            )
            ax = tuple(range(dLdpoly.ndim))
            dLdA = self.sum(dLdpoly[..., None, None] * self.conj(dpoly_dA), axes=ax)
            dLdB = self.sum(dLdpoly[..., None] * self.conj(dpoly_dB), axes=ax)
            dLdC = self.sum(dLdpoly * self.conj(dpoly_dC), axes=ax)
            return dLdA, dLdB, dLdC

        return poly0, grad

    def hermite_renormalized_1leftoverMode(
        self, A: tf.Tensor, B: tf.Tensor, C: tf.Tensor, cutoffs: Tuple[int]
    ) -> tf.Tensor:
        r"""First, reorder A and B parameters of Bargmann representation to match conventions in mrmustard.math.numba.compactFock~
        Then, calculate the required renormalized multidimensional Hermite polynomial.
        """
        A, B = self.reorder_AB_bargmann(A, B)
        return self.hermite_renormalized_1leftoverMode_reorderedAB(A, B, C, cutoffs=cutoffs)

    @tf.custom_gradient
    def hermite_renormalized_1leftoverMode_reorderedAB(
        self, A: tf.Tensor, B: tf.Tensor, C: tf.Tensor, cutoffs: Tuple[int]
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
        poly0, poly2, poly1010, poly1001, poly1 = tf.numpy_function(
            hermite_multidimensional_1leftoverMode, [A, B, C, cutoffs], [A.dtype] * 5
        )

        def grad(dLdpoly):
            dpoly_dC, dpoly_dA, dpoly_dB = tf.numpy_function(
                grad_hermite_multidimensional_1leftoverMode,
                [A, B, C, poly0, poly2, poly1010, poly1001, poly1],
                [poly0.dtype] * 3,
            )
            ax = tuple(range(dLdpoly.ndim))
            dLdA = self.sum(dLdpoly[..., None, None] * self.conj(dpoly_dA), axes=ax)
            dLdB = self.sum(dLdpoly[..., None] * self.conj(dpoly_dB), axes=ax)
            dLdC = self.sum(dLdpoly * self.conj(dpoly_dC), axes=ax)
            return dLdA, dLdB, dLdC

        return poly0, grad

    @staticmethod
    def eigvals(tensor: tf.Tensor) -> Tensor:
        """Returns the eigenvalues of a matrix."""
        return tf.linalg.eigvals(tensor)

    @staticmethod
    def eigvalsh(tensor: tf.Tensor) -> Tensor:
        """Returns the eigenvalues of a Real Symmetric or Hermitian matrix."""
        return tf.linalg.eigvalsh(tensor)

    @staticmethod
    def svd(tensor: tf.Tensor) -> Tensor:
        """Returns the Singular Value Decomposition of a matrix."""
        return tf.linalg.svd(tensor)

    @staticmethod
    def xlogy(x: tf.Tensor, y: tf.Tensor) -> Tensor:
        """Returns 0 if ``x == 0,`` and ``x * log(y)`` otherwise, elementwise."""
        return tf.math.xlogy(x, y)

    @staticmethod
    def eigh(tensor: tf.Tensor) -> Tensor:
        """Returns the eigenvalues and eigenvectors of a matrix."""
        return tf.linalg.eigh(tensor)

    def sqrtm(self, tensor: tf.Tensor, rtol=1e-05, atol=1e-08) -> Tensor:
        """Returns the matrix square root of a square matrix, such that ``sqrt(A) @ sqrt(A) = A``."""

        # The sqrtm function has issues with matrices that are close to zero, hence we branch
        if np.allclose(tensor, 0, rtol=rtol, atol=atol):
            return self.zeros_like(tensor)
        return tf.linalg.sqrtm(tensor)

    @staticmethod
    def boolean_mask(tensor: tf.Tensor, mask: tf.Tensor) -> Tensor:
        """Returns a tensor based on the truth value of the boolean mask."""
        return tf.boolean_mask(tensor, mask)

    @staticmethod
    def custom_gradient(func, *args, **kwargs):
        """Decorator to define a function with a custom gradient."""
        return tf.custom_gradient(func, *args, **kwargs)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Extras (not in the Interface)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
