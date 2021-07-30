import numpy as np
import tensorflow as tf
from scipy.special import binom
from scipy.stats import unitary_group
from itertools import product
from functools import lru_cache
from mrmustard.backends import BackendInterface
from thewalrus._hermite_multidimensional import hermite_multidimensional_numba, grad_hermite_multidimensional_numba
from mrmustard.typing import *

#  NOTE: the reason why we have a class with methods and not a namespace with functions
#  is that we want to enforce the interface, in order to ensure compatibility
#  of new backends with the rest of the codebase.


class Backend(BackendInterface):

    dtype_order = (tf.float16, float, tf.float32, tf.float64, complex, tf.complex64, tf.complex128)
    no_cast = (int, tf.int8, tf.uint8, tf.int16, tf.uint16, tf.int32, tf.uint32, tf.int64, tf.uint64)

    # ~~~~~~~~~
    # Basic ops
    # ~~~~~~~~~

    def astensor(self, array: Union[np.ndarray, tf.Tensor]) -> tf.Tensor:
        return tf.convert_to_tensor(array)

    def conj(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.conj(array)

    def real(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.real(array)

    def imag(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.imag(array)

    def exp(self, array: tf.Tensor) -> tf.Tensor:
        return tf.math.exp(array)

    def lgamma(self, x: tf.Tensor) -> tf.Tensor:
        return tf.math.lgamma(x)

    def log(self, x: tf.Tensor) -> tf.Tensor:
        return tf.math.log(x)

    def maximum(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        return tf.maximum(a, b)

    def minimum(self, a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        return tf.minimum(a, b)

    def abs(self, array: tf.Tensor) -> tf.Tensor:
        return tf.abs(array)

    def norm(self, array: tf.Tensor) -> tf.Tensor:
        'Note that the norm preserves the type of array'
        return tf.linalg.norm(array)

    def matmul(self, a: tf.Tensor, b: tf.Tensor, transpose_a=False, transpose_b=False, adjoint_a=False, adjoint_b=False)  -> tf.Tensor:
        return tf.linalg.matmul(a, b, transpose_a, transpose_b, adjoint_a, adjoint_b)

    def matvec(self, a: tf.Tensor, b: tf.Tensor, transpose_a=False, adjoint_a=False) -> tf.Tensor:
        return tf.linalg.matvec(a, b, transpose_a, adjoint_a)

    def tensordot(self, a: tf.Tensor, b: tf.Tensor, axes: List[int]) -> tf.Tensor:
        return tf.tensordot(a, b, axes)

    def einsum(self, string: str, *tensors) -> tf.Tensor:
        return tf.einsum(string, *tensors)

    def inv(self, a: tf.Tensor) -> tf.Tensor:
        return tf.linalg.inv(a)

    def pinv(self, array: tf.Tensor) -> tf.Tensor:
        return tf.linalg.pinv(array)

    def det(self, a: tf.Tensor) -> tf.Tensor:
        return tf.linalg.det(a)

    def tile(self, array: tf.Tensor, repeats: Sequence[int]) -> tf.Tensor:
        return tf.tile(array, repeats)

    def diag(self, array: tf.Tensor) -> tf.Tensor:
        return tf.linalg.diag(array)

    def diag_part(self, array: tf.Tensor) -> tf.Tensor:
        return tf.linalg.diag_part(array)

    def pad(self, array: tf.Tensor, paddings: Sequence[Tuple[int, int]], mode='CONSTANT', constant_values=0) -> tf.Tensor:
        return tf.pad(array, paddings, mode, constant_values)

    def convolution(self, array: tf.Tensor, filters: tf.Tensor, strides: List[int], padding='VALID', data_format='NWC', dilations: Optional[List[int]] = None) -> tf.Tensor:
        return tf.nn.convolution(array, filters, strides, padding, data_format, dilations)

    def transpose(self, a: tf.Tensor, perm: List[int]):
        return tf.transpose(a, perm)

    def reshape(self, array: tf.Tensor, shape: Sequence[int]) -> tf.Tensor:
        return tf.reshape(array, shape)

    def sum(self, array: tf.Tensor, axes: Sequence[int]=None):
        return tf.reduce_sum(array, axes)

    def arange(self, start: int, limit: int = None, delta: int = 1) -> tf.Tensor:
        return tf.range(start, limit, delta, dtype=tf.float64)

    def outer(self, array1: tf.Tensor, array2: tf.Tensor) -> tf.Tensor:
        return tf.tensordot(array1, array2, [[], []])

    def eye(self, size: int, dtype=tf.float64) -> tf.Tensor:
        return tf.eye(size, dtype=dtype)

    def zeros(self, shape: Sequence[int], dtype=tf.float64) -> tf.Tensor:
        return tf.zeros(shape, dtype=dtype)

    def zeros_like(self, array: tf.Tensor) -> tf.Tensor:
        return tf.zeros_like(array)

    def ones(self, shape: Sequence[int], dtype=tf.float64) -> tf.Tensor:
        return tf.ones(shape, dtype=dtype)

    def ones_like(self, array: tf.Tensor) -> tf.Tensor:
        return tf.ones_like(array)

    def gather(self, array: tf.Tensor, indices: tf.Tensor, axis: tf.Tensor) -> tf.Tensor:
        return tf.gather(array, indices, axes)

    def trace(self, array: tf.Tensor) -> tf.Tensor:
        return tf.linalg.trace(array)

    def concat(self, values, axis):
        return tf.concat(values, axis)

    def update_tensor(self, tensor: tf.Tensor, indices: tf.Tensor, values: tf.Tensor):
        return tf.tensor_scatter_nd_update(tensor, indices, values)

    def update_add_tensor(self, tensor: tf.Tensor, indices: tf.Tensor, values: tf.Tensor):
        return tf.tensor_scatter_nd_add(tensor, indices, values)

    def constraint_func(self, bounds: Tuple[Optional[float], Optional[float]]) -> Optional[Callable]:
        bounds = (-np.inf if bounds[0] is None else bounds[0], np.inf if bounds[1] is None else bounds[1])
        if not bounds == (-np.inf, np.inf):
            constraint: Optional[Callable] = lambda x: tf.clip_by_value(x, bounds[0], bounds[1])
        else:
            constraint = None
        return constraint

    def new_variable(self, value, bounds: Tuple[Optional[float], Optional[float]], name: str):
        return tf.Variable(value, dtype=tf.float64, name=name, constraint=self.constraint_func(bounds))

    def new_constant(self, value, name: str):
        return tf.constant(value, dtype=tf.float64, name=name)

    @tf.custom_gradient
    def hermite_renormalized(self, A: tf.Tensor, B: tf.Tensor, C: tf.Tensor, shape: Sequence[int]) -> tf.Tensor:  # TODO this is not ready
        r"""
        Renormalized multidimensional Hermite polynomial given by the Taylor series of exp(Ax^2 + Bx + C) at zero.
        
        Args:
            A: The A matrix.
            B: The B vector.
            C: The C scalar.
            shape: The shape of the final tensor.
        Returns:
            The Fock state.
        """
        poly = hermite_multidimensional_numba(-A, shape, B, C)

        def grad(dLdpoly):
            dpoly_dC, dpoly_dA, dpoly_dB = grad_hermite_multidimensional_numba(poly, -A, shape, B, C)
            ax = tuple(range(dLdpoly.ndim))
            dLdA = self.sum(dLdpoly[..., None, None] * self.conj(dpoly_dA), axis=ax)
            dLdB = self.sum(dLdpoly[..., None] * self.conj(dpoly_dB), axis=ax))
            dLdC = self.sum(dLdpoly * self.conj(dpoly_dC), axis=ax))
            return dLdA, dLdB, dLdC

        return poly, grad