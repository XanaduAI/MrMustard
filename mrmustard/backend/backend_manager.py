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

import importlib.util
import numpy as np
import sys
from functools import lru_cache
from itertools import product
from scipy.special import binom
from scipy.stats import ortho_group, unitary_group

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from .backend_numpy import BackendNumpy
from ..utils.settings import settings
from ..utils.typing import (
    Matrix,
    Scalar,
    Tensor,
    Trainable,
    Vector,
)

# ~~~~~~~
# Helpers
# ~~~~~~~


def lazy_import(module_name: str):
    r"""
    Returns module and loader for lazy import.

    Args:
        module_name: The name of the module to import.
    """
    try:
        return sys.modules[module_name], None
    except KeyError:
        spec = importlib.util.find_spec(module_name)
        module = importlib.util.module_from_spec(spec)
        loader = importlib.util.LazyLoader(spec.loader)
        return module, loader


# lazy import for numpy
module_name_np = "mrmustard.backend.backend_numpy"
module_np, loader_np = lazy_import(module_name_np)

# lazy import for tensorflow
module_name_tf = "mrmustard.backend.backend_tensorflow"
module_tf, loader_tf = lazy_import(module_name_tf)

all_modules = {
    "numpy": {"module": module_np, "loader": loader_np, "object": "BackendNumpy"},
    "tensorflow": {"module": module_tf, "loader": loader_tf, "object": "BackendTensorflow"},
}

# ~~~~~~~
# Classes
# ~~~~~~~


class BackendManager:
    r"""
    A class to manage backends.
    """

    @property
    def backend(self):
        r"""
        The backend that is being used.
        """
        backend = settings.BACKEND
        module = all_modules[backend]["module"]
        object = all_modules[backend]["object"]
        try:
            ret = getattr(module, object)()
        except:
            loader = all_modules[backend]["loader"]
            loader.exec_module(module)
            ret = getattr(module, object)()
        return ret

    def __new__(cls):
        # singleton
        if not hasattr(cls, "instance"):
            cls.instance = super(BackendManager, cls).__new__(cls)
        return cls.instance

    def _apply(self, fn: str, args: Optional[Sequence[any]] = ()):
        r"""
        Applies a function ``fn`` from the backend in use to the given ``args``.
        """
        try:
            return getattr(self.backend, fn)(*args)
        except AttributeError:
            msg = f"Function ``{fn}`` not implemented for backend ``{self.backend.name}``."
            raise NotImplementedError(msg)

    # ~~~~~~~
    # Methods
    # ~~~~~~~
    # Below are the methods supported by the various backends.

    def hello(self):
        r"""A function to say hello."""
        self._apply("hello")

    def sum(self, x, y):
        r"""A function to sum two numbers."""
        return self._apply("sum", (x, y))

    def abs(self, array: Tensor) -> Tensor:
        r"""The absolute value of array.

        Args:
            array: The array to take the absolute value of.

        Returns:
            The absolute value of the given ``array``.
        """
        return self._apply("abs", (array,))

    def any(self, array: Tensor) -> bool:
        r"""Returns ``True`` if any element of array is ``True``, ``False`` otherwise.

        Args:
            array (array): array to check

        Returns:
            bool: True if any element of array is True
        """
        return self._apply("any", (array,))

    def arange(self, start: int, limit: int = None, delta: int = 1, dtype: Any = None) -> Tensor:
        r"""Returns an array of evenly spaced values within a given interval.

        Args:
            start: start of the interval
            limit: end of the interval
            delta: step size
            dtype: dtype of the returned array

        Returns:
            array: array of evenly spaced values
        """
        # NOTE: is float64 by default
        return self._apply("arange", (start, limit, delta, dtype))

    def asnumpy(self, tensor: Tensor) -> Tensor:
        r"""Converts an array to a numpy array.

        Args:
            tensor: The tensor to convert.

        Returns:
            The corrsponidng numpy array.
        """
        return self._apply("asnumpy", (tensor,))

    def assign(self, tensor: Tensor, value: Tensor) -> Tensor:
        r"""Assigns value to tensor.

        Args:
            tensor: The tensor to assign to.
            value: The value to assign.

        Returns:
            The tensor with value assigned
        """
        return self._apply("assign", (tensor, value))

    def astensor(self, array: Tensor, dtype: str):
        r"""Converts a numpy array to a tensor.

        Args:
            array: The numpy array to convert.
            dtype: The dtype of the tensor.

        Returns:
            The tensor with dtype.
        """
        return self._apply("astensor", (array, dtype))

    def atleast_1d(self, array: Tensor, dtype: str = None) -> Tensor:
        r"""Returns an array with at least one dimension.

        Args:
            array: The array to convert.
            dtype: The data type of the array.

        Returns:
            array: The array with at least one dimension.
        """
        return self._apply("atleast_1d", (array, dtype))

    def cast(self, array: Tensor, dtype) -> Tensor:
        r"""Casts ``array`` to ``dtype``.

        Args:
            array: The array to cast.
            dtype: The data type to cast to.

        Returns:
            The array cast to dtype.
        """
        return self._apply("cast", (array, dtype))

    def clip(self, array: Tensor, a_min: float, a_max: float) -> Tensor:
        r"""Clips array to the interval ``[a_min, a_max]``.

        Args:
            array: The array to clip.
            a_min: The minimum value.
            a_max: The maximum value.

        Returns:
            The clipped array.
        """
        return self._apply("clip", (array, a_min, a_max))

    def concat(self, values: Sequence[Tensor], axis: int) -> Tensor:
        r"""Concatenates values along the given axis.

        Args:
            values: The values to concatenate.
            axis: The axis along which to concatenate.

        Returns:
            The concatenated values.
        """
        return self._apply("concat", (values, axis))

    def conj(self, array: Tensor) -> Tensor:
        r"""The complex conjugate of array.

        Args:
            array: The array to take the complex conjugate of.

        Returns:
            The complex conjugate of the given ``array``.
        """
        return self._apply("conj", (array))

    def constraint_func(
        self, bounds: Tuple[Optional[float], Optional[float]]
    ) -> Optional[Callable]:
        r"""Returns a constraint function for the given bounds.

        A constraint function will clip the value to the interval given by the bounds.

        .. note::

            The upper and/or lower bounds can be ``None``, in which case the constraint
            function will not clip the value.

        Args:
            bounds: The bounds of the constraint.

        Returns:
            The constraint function.
        """
        return self._apply("constraint_func", (bounds))

    def convolution(
        self,
        array: Tensor,
        filters: Tensor,
        padding="VALID",
        data_format="NWC",
    ) -> Tensor:  # TODO: remove strides and data_format?
        r"""Performs a convolution on array with filters.

        Args:
            array: The array to convolve.
            filters: The filters to convolve with.
            padding: The padding mode.
            data_format: The data format of the array.

        Returns:
            The convolved array.
        """
        return self._apply("convolution", (array, filters, padding, data_format))

    def cos(self, array: Tensor) -> Tensor:
        r"""The cosine of array.

        Args:
            array (array): array to take the cosine of

        Returns:
            array: cosine of array
        """
        return self._apply("cos", (array))

    def cosh(self, array: Tensor) -> Tensor:
        r"""The hyperbolic cosine of array.

        Args:
            array (array): array to take the hyperbolic cosine of

        Returns:
            array: hyperbolic cosine of array
        """
        return self._apply("cosh", (array))

    def make_complex(self, real: Tensor, imag: Tensor) -> Tensor:
        """Given two real tensors representing the real and imaginary part of a complex number,
        this operation returns a complex tensor. The input tensors must have the same shape.

        Args:
            real (array): real part of the complex number
            imag (array): imaginary part of the complex number

        Returns:
            array: complex array ``real + 1j * imag``
        """
        return self._apply("make_complex", (real, imag))

    def atan2(self, y: Tensor, x: Tensor) -> Tensor:
        r"""Computes the trignometric inverse tangent of y/x element-wise.

        Args:
            y (array): numerator array
            x (array): denominator array

        Returns:
            array: arctan of y/x
        """
        return self._apply("atan2", (y, x))

    def det(self, matrix: Tensor) -> Tensor:
        r"""The determinant of matrix.

        Args:
            matrix (matrix): matrix to take the determinant of

        Returns:
            determinant of matrix
        """
        return self._apply("det", (matrix))

    def diag(self, array: Tensor, k: int) -> Tensor:
        r"""The array made by inserting the given array along the :math:`k`-th diagonal.

        Args:
            array (array): array to insert
            k (int): kth diagonal to insert array into

        Returns:
            array: array with array inserted into the kth diagonal
        """
        return self._apply("diag", (array, k))

    def diag_part(self, array: Tensor, k: int) -> Tensor:
        r"""The array of the main diagonal of array.

        Args:
            array (array): array to extract the main diagonal of
            k (int): diagonal to extract

        Returns:
            array: array of the main diagonal of array
        """
        return self._apply("diag_part", (array, k))

    def eigvals(self, tensor: Tensor) -> Tensor:
        r"""The eigenvalues of a matrix."""
        return self._apply("eigvals", (tensor,))

    def einsum(self, string: str, *tensors) -> Tensor:
        r"""The result of the Einstein summation convention on the tensors.

        Args:
            string (str): string of the Einstein summation convention
            tensors (array): tensors to perform the Einstein summation on

        Returns:
            array: result of the Einstein summation convention
        """
        return self._apply("einsum", (string, *tensors))

    def exp(self, array: Tensor) -> Tensor:
        r"""The exponential of array element-wise.

        Args:
            array (array): array to take the exponential of

        Returns:
            array: exponential of array
        """
        return self._apply("exp", (array))

    def expand_dims(self, array: Tensor, axis: int) -> Tensor:
        r"""The array with an additional dimension inserted at the given axis.

        Args:
            array (array): array to expand
            axis (int): axis to insert the new dimension

        Returns:
            array: array with an additional dimension inserted at the given axis
        """
        return self._apply("expand_dims", (array, axis))

    def expm(self, matrix: Tensor) -> Tensor:
        r"""The matrix exponential of matrix.

        Args:
            matrix (matrix): matrix to take the exponential of

        Returns:
            matrix: exponential of matrix
        """
        return self._apply("expm", (matrix,))

    def eye(self, size: int, dtype) -> Tensor:
        r"""The identity matrix of size.

        Args:
            size (int): size of the identity matrix
            dtype (dtype): data type of the identity matrix

        Returns:
            matrix: identity matrix
        """
        return self._apply("eye", (size, dtype))

    def eye_like(self, array: Tensor) -> Tensor:
        r"""The identity matrix of the same shape and dtype as array.

        Args:
            array (array): array to create the identity matrix of

        Returns:
            matrix: identity matrix
        """
        return self._apply("eye_like", (array,))

    def from_backend(self, value: Any) -> bool:
        r"""Whether the given tensor is a tensor of the concrete backend."""
        return self._apply("from_backend", (value,))

    def gather(self, array: Tensor, indices: Tensor, axis: int) -> Tensor:
        r"""The values of the array at the given indices.

        Args:
            array (array): array to gather values from
            indices (array): indices to gather values from
            axis (int): axis to gather values from

        Returns:
            array: values of the array at the given indices
        """
        return self._apply(
            "gather",
            (
                array,
                indices,
                axis,
            ),
        )

    def hash_tensor(self, tensor: Tensor) -> int:
        r"""The hash of the given tensor.

        Args:
            tensor (array): tensor to hash

        Returns:
            int: hash of the given tensor
        """
        return self._apply("hash_tensor", (tensor,))

    def hermite_renormalized(self, A: Matrix, B: Vector, C: Scalar, shape: Sequence[int]) -> Tensor:
        r"""The array of hermite renormalized polynomials of the given coefficients.

        Args:
            A (array): Matrix coefficient of the hermite polynomial
            B (array): Vector coefficient of the hermite polynomial
            C (array): Scalar coefficient of the hermite polynomial
            shape (tuple): shape of the hermite polynomial

        Returns:
            array: renormalized hermite polynomials
        """
        return self._apply(
            "hermite_renormalized",
            (
                A,
                B,
                C,
                shape,
            ),
        )

    def imag(self, array: Tensor) -> Tensor:
        r"""The imaginary part of array.

        Args:
            array (array): array to take the imaginary part of

        Returns:
            array: imaginary part of array
        """
        return self._apply("imag", (array,))

    def inv(self, tensor: Tensor) -> Tensor:
        r"""The inverse of tensor.

        Args:
            tensor (array): tensor to take the inverse of

        Returns:
            array: inverse of tensor
        """
        return self._apply("inv", (tensor,))

    def is_trainable(self, tensor: Tensor) -> bool:
        r"""Whether the given tensor is trainable.

        Args:
            tensor: The tensor to train.

        Returns:
            Whether the given tensor can be trained.
        """
        return self._apply("is_trainable", (tensor,))

    def lgamma(self, x: Tensor) -> Tensor:
        r"""The natural logarithm of the gamma function of ``x``.

        Args:
            x (array): array to take the natural logarithm of the gamma function of

        Returns:
            array: natural logarithm of the gamma function of ``x``
        """
        return self._apply("lgamma", (x,))

    def log(self, x: Tensor) -> Tensor:
        r"""The natural logarithm of ``x``.

        Args:
            x (array): array to take the natural logarithm of

        Returns:
            array: natural logarithm of ``x``
        """
        return self._apply("log", (x,))

    def matmul(
        self,
        a: Tensor,
        b: Tensor,
        transpose_a=False,
        transpose_b=False,
        adjoint_a=False,
        adjoint_b=False,
    ) -> Tensor:
        r"""The matrix product of ``a`` and ``b``.

        Args:
            a (array): first matrix to multiply
            b (array): second matrix to multiply
            transpose_a (bool): whether to transpose ``a``
            transpose_b (bool): whether to transpose ``b``
            adjoint_a (bool): whether to adjoint ``a``
            adjoint_b (bool): whether to adjoint ``b``

        Returns:
            array: matrix product of ``a`` and ``b``
        """
        return self._apply("matmul", (a, b, transpose_a, transpose_b, adjoint_a, adjoint_b))

    def matvec(self, a: Matrix, b: Vector, transpose_a=False, adjoint_a=False) -> Tensor:
        r"""The matrix vector product of ``a`` (matrix) and ``b`` (vector).

        Args:
            a (array): matrix to multiply
            b (array): vector to multiply
            transpose_a (bool): whether to transpose ``a``
            adjoint_a (bool): whether to adjoint ``a``

        Returns:
            array: matrix vector product of ``a`` and ``b``
        """
        return self._apply("matvec", (a, b, transpose_a, adjoint_a))

    def maximum(self, a: Tensor, b: Tensor) -> Tensor:
        r"""The element-wise maximum of ``a`` and ``b``.

        Args:
            a (array): first array to take the maximum of
            b (array): second array to take the maximum of

        Returns:
            array: element-wise maximum of ``a`` and ``b``
        """
        return self._apply(
            "maximum",
            (
                a,
                b,
            ),
        )

    def minimum(self, a: Tensor, b: Tensor) -> Tensor:
        r"""The element-wise minimum of ``a`` and ``b``.

        Args:
            a (array): first array to take the minimum of
            b (array): second array to take the minimum of

        Returns:
            array: element-wise minimum of ``a`` and ``b``
        """
        return self._apply(
            "minimum",
            (
                a,
                b,
            ),
        )

    def new_variable(
        self, value: Tensor, bounds: Tuple[Optional[float], Optional[float]], name: str, dtype: Any
    ) -> Tensor:
        r"""Returns a new variable with the given value and bounds.

        Args:
            value (array): value of the new variable
            bounds (tuple): bounds of the new variable
            name (str): name of the new variable
            dtype (type): dtype of the array
        Returns:
            array: new variable
        """
        return self._apply("new_variable", (value, bounds, name, dtype))

    def new_constant(self, value: Tensor, name: str, dtype: Any) -> Tensor:
        r"""Returns a new constant with the given value.

        Args:
            value (array): value of the new constant
            name (str): name of the new constant
            dtype (type): dtype of the array

        Returns:
            array: new constant
        """
        return self._apply("new_constant", (value, name, dtype))

    def norm(self, array: Tensor) -> Tensor:
        r"""The norm of array.

        Args:
            array (array): array to take the norm of

        Returns:
            array: norm of array
        """
        return self._apply("norm", (array,))

    def ones(self, shape: Sequence[int], dtype) -> Tensor:
        r"""Returns an array of ones with the given ``shape`` and ``dtype``.

        Args:
            shape (tuple): shape of the array
            dtype (type): dtype of the array

        Returns:
            array: array of ones
        """
        # NOTE : should be float64 by default
        return self._apply("ones", (shape, dtype))

    def ones_like(self, array: Tensor) -> Tensor:
        r"""Returns an array of ones with the same shape and ``dtype`` as ``array``.

        Args:
            array (array): array to take the shape and dtype of

        Returns:
            array: array of ones
        """
        return self._apply("ones_like", (array,))

    def outer(self, array1: Tensor, array2: Tensor) -> Tensor:
        r"""The outer product of ``array1`` and ``array2``.

        Args:
            array1 (array): first array to take the outer product of
            array2 (array): second array to take the outer product of

        Returns:
            array: outer product of array1 and array2
        """
        return self._apply("outer", (array1, array2))

    def pad(
        self, array: Tensor, paddings: Sequence[Tuple[int, int]], mode="CONSTANT", constant_values=0
    ) -> Tensor:
        r"""The padded array.

        Args:
            array (array): array to pad
            paddings (tuple): paddings to apply
            mode (str): mode to apply the padding
            constant_values (int): constant values to use for padding

        Returns:
            array: padded array
        """
        return self._apply("pad", (array, paddings, mode, constant_values))

    def pinv(self, matrix: Tensor) -> Tensor:
        r"""The pseudo-inverse of matrix.

        Args:
            matrix (array): matrix to take the pseudo-inverse of

        Returns:
            array: pseudo-inverse of matrix
        """
        return self._apply("pinv", (matrix,))

    def pow(self, x: Tensor, y: Tensor) -> Tensor:
        r"""Returns :math:`x^y`. Broadcasts ``x`` and ``y`` if necessary.
        Args:
            x (array): base
            y (array): exponent

        Returns:
            array: :math:`x^y`
        """
        return self._apply("pow", (x, y))

    def real(self, array: Tensor) -> Tensor:
        r"""The real part of ``array``.

        Args:
            array (array): array to take the real part of

        Returns:
            array: real part of ``array``
        """
        return self._apply("real", (array,))

    def reshape(self, array: Tensor, shape: Sequence[int]) -> Tensor:
        r"""The reshaped array.

        Args:
            array (array): array to reshape
            shape (tuple): shape to reshape the array to

        Returns:
            array: reshaped array
        """
        return self._apply("reshape", (array, shape))

    def set_diag(self, array: Tensor, diag: Tensor, k: int) -> Tensor:
        r"""The array with the diagonal set to ``diag``.

        Args:
            array (array): array to set the diagonal of
            diag (array): diagonal to set
            k (int): diagonal to set

        Returns:
            array: array with the diagonal set to ``diag``
        """
        return self._apply("set_diag", (array, diag, k))

    def sin(self, array: Tensor) -> Tensor:
        r"""The sine of ``array``.

        Args:
            array (array): array to take the sine of

        Returns:
            array: sine of ``array``
        """
        return self._apply("sin", (array,))

    def sinh(self, array: Tensor) -> Tensor:
        r"""The hyperbolic sine of ``array``.

        Args:
            array (array): array to take the hyperbolic sine of

        Returns:
            array: hyperbolic sine of ``array``
        """
        return self._apply("sinh", (array,))

    def solve(self, matrix: Tensor, rhs: Tensor) -> Tensor:
        r"""The solution of the linear system :math:`Ax = b`.

        Args:
            matrix (array): matrix :math:`A`
            rhs (array): vector :math:`b`

        Returns:
            array: solution :math:`x`
        """
        return self._apply("solve", (matrix, rhs))

    def sqrt(self, x: Tensor, dtype=None) -> Tensor:
        r"""The square root of ``x``.

        Args:
            x (array): array to take the square root of
            dtype (type): ``dtype`` of the output array

        Returns:
            array: square root of ``x``
        """
        return self._apply("sqrt", (x, dtype))

    def sqrtm(self, tensor: Tensor) -> Tensor:
        r"""The matrix square root."""
        return self._apply("sqrtm", (tensor,))

    def sum(self, array: Tensor, axes: Sequence[int] = None):
        r"""The sum of array.

        Args:
            array (array): array to take the sum of
            axes (tuple): axes to sum over

        Returns:
            array: sum of array
        """
        return self._apply("sum", (array, axes))

    def tensordot(self, a: Tensor, b: Tensor, axes: Sequence[int]) -> Tensor:
        r"""The tensordot product of ``a`` and ``b``.

        Args:
            a (array): first array to take the tensordot product of
            b (array): second array to take the tensordot product of
            axes (tuple): axes to take the tensordot product over

        Returns:
            array: tensordot product of ``a`` and ``b``
        """
        return self._apply("tensordot", (a, b, axes))

    def tile(self, array: Tensor, repeats: Sequence[int]) -> Tensor:
        r"""The tiled array.

        Args:
            array (array): array to tile
            repeats (tuple): number of times to tile the array along each axis

        Returns:
            array: tiled array
        """
        return self._apply("tile", (array, repeats))

    def trace(self, array: Tensor, dtype: Any = None) -> Tensor:
        r"""The trace of array.

        Args:
            array (array): array to take the trace of
            dtype (type): ``dtype`` of the output array

        Returns:
            array: trace of array
        """
        return self._apply("trace", (array, dtype))

    def transpose(self, a: Tensor, perm: Sequence[int] = None):
        r"""The transposed arrays.

        Args:
            a (array): array to transpose
            perm (tuple): permutation to apply to the array

        Returns:
            array: transposed array
        """
        return self._apply("transpose", (a, perm))

    def unique_tensors(self, lst: List[Tensor]) -> List[Tensor]:
        r"""The tensors in ``lst`` without duplicates and non-tensors.

        Args:
            lst (list): list of tensors to remove duplicates and non-tensors from.

        Returns:
            list: list of tensors without duplicates and non-tensors.
        """
        return self._apply("unique_tensors", (lst,))

    def update_tensor(self, tensor: Tensor, indices: Tensor, values: Tensor) -> Tensor:
        r"""Updates a tensor in place with the given values.

        Args:
            tensor (array): tensor to update
            indices (array): indices to update
            values (array): values to update
        """
        return self._apply("update_tensor", (tensor, indices, values))

    def update_add_tensor(self, tensor: Tensor, indices: Tensor, values: Tensor) -> Tensor:
        r"""Updates a tensor in place by adding the given values.

        Args:
            tensor (array): tensor to update
            indices (array): indices to update
            values (array): values to add
        """
        return self._apply("update_add_tensor", (tensor, indices, values))

    def value_and_gradients(
        self, cost_fn: Callable, parameters: Dict[str, List[Trainable]]
    ) -> Tuple[Tensor, Dict[str, List[Tensor]]]:
        r"""The loss and gradients of the given cost function.

        Args:
            cost_fn (callable): cost function to compute the loss and gradients of
            parameters (dict): parameters to compute the loss and gradients of

        Returns:
            tuple: loss and gradients (dict) of the given cost function
        """
        return self._apply("value_and_gradients", (cost_fn, parameters))

    def zeros(self, shape: Sequence[int], dtype) -> Tensor:
        r"""Returns an array of zeros with the given shape and ``dtype``.

        Args:
            shape (tuple): shape of the array
            dtype (type): dtype of the array

        Returns:
            array: array of zeros
        """
        return self._apply("zeros", (shape,))

    def zeros_like(self, array: Tensor) -> Tensor:
        r"""Returns an array of zeros with the same shape and ``dtype`` as ``array``.

        Args:
            array (array): array to take the shape and ``dtype`` of

        Returns:
            array: array of zeros
        """
        return self._apply("zeros_like", (array,))

    def map_fn(self, fn: Callable, elements: Tensor) -> Tensor:
        """Transforms elems by applying fn to each element unstacked on axis 0.

        Args:
            fn (func): The callable to be performed. It accepts one argument,
                which will have the same (possibly nested) structure as elems.
            elements (Tensor): A tensor or (possibly nested) sequence of tensors,
                each of which will be unstacked along their first dimension.
                ``func`` will be applied to the nested sequence of the resulting slices.

        Returns:
            Tensor: applied ``func`` on ``elements``
        """
        return self._apply("map_fn", (fn, elements))

    def squeeze(self, tensor: Tensor, axis: Optional[List[int]]) -> Tensor:
        """Removes dimensions of size 1 from the shape of a tensor.

        Args:
            tensor (Tensor): the tensor to squeeze
            axis (Optional[List[int]]): if specified, only squeezes the
                dimensions listed, defaults to []

        Returns:
            Tensor: tensor with one or more dimensions of size 1 removed
        """
        return self._apply("squeeze", (tensor, axis))

    def cholesky(self, input: Tensor) -> Tensor:
        """Computes the Cholesky decomposition of square matrices.

        Args:
            input (Tensor)

        Returns:
            Tensor: tensor with the same type as input
        """
        return self._apply("cholesky", (input,))

    def Categorical(self, probs: Tensor, name: str):
        """Categorical distribution over integers.

        Args:
            probs (Tensor): tensor representing the probabilities of a set of Categorical
                distributions.
            name (str): name prefixed to operations created by this class

        Returns:
            tfp.distributions.Categorical: instance of ``tfp.distributions.Categorical`` class
        """
        return self._apply("Categorical", (probs, name))

    def MultivariateNormalTriL(self, loc: Tensor, scale_tril: Tensor):
        """Multivariate normal distribution on `R^k` and parameterized by a (batch of) length-k loc
        vector (aka "mu") and a (batch of) k x k scale matrix; covariance = scale @ scale.T
        where @ denotes matrix-multiplication.

        Args:
            loc (Tensor): if this is set to None, loc is implicitly 0
            scale_tril: lower-triangular Tensor with non-zero diagonal elements

        Returns:
            tfp.distributions.MultivariateNormalTriL: instance of ``tfp.distributions.MultivariateNormalTriL``
        """
        return self._apply("MultivariateNormalTriL", (loc, scale_tril))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Methods that build on the basic ops and don't need to be overridden in the backend implementation
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @property
    def euclidean_opt(self):
        r"""The configured Euclidean optimizer."""
        if not self._euclidean_opt:
            self._euclidean_opt = self.DefaultEuclideanOptimizer()
        return self._euclidean_opt

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Methods that build on the basic ops and don't need to be overridden in the backend implementation
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def block(self, blocks: List[List[Tensor]], axes=(-2, -1)) -> Tensor:
        r"""Returns a matrix made from the given blocks.

        Args:
            blocks (list): list of lists of compatible blocks
            axes (tuple): axes to stack the blocks along

        Returns:
            array: matrix made of blocks
        """
        rows = [self.concat(row, axis=axes[1]) for row in blocks]
        return self.concat(rows, axis=axes[0])

    def dagger(self, array: Tensor) -> Tensor:
        """The adjoint of ``array``. This operation swaps the first
        and second half of the indexes and then conjugates the matrix.

        Args:
            array (array): array to take the adjoint of

        Returns:
            array: adjoint of ``array``
        """
        N = len(array.shape) // 2
        perm = list(range(N, 2 * N)) + list(range(0, N))
        return self.conj(self.transpose(array, perm=perm))

    def unitary_to_orthogonal(self, U):
        r"""Unitary to orthogonal mapping.

        Args:
            U (array): unitary matrix in ``U(n)``

        Returns:
            array: orthogonal matrix in :math:`O(2n)`
        """
        X = self.real(U)
        Y = self.imag(U)
        return self.block([[X, -Y], [Y, X]])

    def random_symplectic(self, num_modes: int, max_r: float = 1.0) -> Tensor:
        r"""A random symplectic matrix in ``Sp(2*num_modes)``.

        Squeezing is sampled uniformly from 0.0 to ``max_r`` (1.0 by default).
        """
        if num_modes == 1:
            W = np.exp(1j * settings.rng.uniform(size=(1, 1)))
            V = np.exp(1j * settings.rng.uniform(size=(1, 1)))
        else:
            W = unitary_group.rvs(dim=num_modes, random_state=settings.rng)
            V = unitary_group.rvs(dim=num_modes, random_state=settings.rng)
        r = settings.rng.uniform(low=0.0, high=max_r, size=num_modes)
        OW = self.unitary_to_orthogonal(W)
        OV = self.unitary_to_orthogonal(V)
        dd = self.diag(self.concat([self.exp(-r), np.exp(r)], axis=0), k=0)
        return OW @ dd @ OV

    @staticmethod
    def random_orthogonal(N: int) -> Tensor:
        """A random orthogonal matrix in :math:`O(N)`."""
        if N == 1:
            return np.array([[1.0]])
        return ortho_group.rvs(dim=N, random_state=settings.rng)

    def random_unitary(self, N: int) -> Tensor:
        """a random unitary matrix in :math:`U(N)`"""
        if N == 1:
            return self.exp(1j * settings.rng.uniform(size=(1, 1)))
        return unitary_group.rvs(dim=N, random_state=settings.rng)

    def single_mode_to_multimode_vec(self, vec, num_modes: int):
        r"""Apply the same 2-vector (i.e. single-mode) to a larger number of modes."""
        if vec.shape[-1] != 2:
            raise ValueError("vec must be 2-dimensional (i.e. single-mode)")
        x, y = vec[..., -2], vec[..., -1]
        vec = self.concat([self.tile([x], [num_modes]), self.tile([y], [num_modes])], axis=-1)
        return vec

    def single_mode_to_multimode_mat(self, mat: Tensor, num_modes: int):
        r"""Apply the same :math:`2\times 2` matrix (i.e. single-mode) to a larger number of modes."""
        if mat.shape[-2:] != (2, 2):
            raise ValueError("mat must be a single-mode (2x2) matrix")
        mat = self.diag(
            self.tile(self.expand_dims(mat, axis=-1), (1, 1, num_modes)), k=0
        )  # shape [2,2,N,N]
        mat = self.reshape(self.transpose(mat, (0, 2, 1, 3)), [2 * num_modes, 2 * num_modes])
        return mat

    @staticmethod
    @lru_cache()
    def Xmat(num_modes: int):
        r"""The matrix :math:`X_n = \begin{bmatrix}0 & I_n\\ I_n & 0\end{bmatrix}.`

        Args:
            num_modes (int): positive integer

        Returns:
            array: :math:`2N\times 2N` array
        """
        I = np.identity(num_modes)
        O = np.zeros((num_modes, num_modes))
        return np.block([[O, I], [I, O]])

    @staticmethod
    @lru_cache()
    def rotmat(num_modes: int):
        "Rotation matrix from quadratures to complex amplitudes."
        I = np.identity(num_modes)
        return np.sqrt(0.5) * np.block([[I, 1j * I], [I, -1j * I]])

    @staticmethod
    @lru_cache()
    def J(num_modes: int):
        """Symplectic form."""
        I = np.identity(num_modes)
        O = np.zeros_like(I)
        return np.block([[O, I], [-I, O]])

    def add_at_modes(
        self, old: Tensor, new: Optional[Tensor], modes: Sequence[int]
    ) -> Tensor:  # NOTE: To be deprecated (XPTensor)
        """Adds two phase-space tensors (cov matrices, displacement vectors, etc..) on the specified modes."""
        if new is None:
            return old
        N = old.shape[-1] // 2
        indices = modes + [m + N for m in modes]
        return self.update_add_tensor(
            old, list(product(*[indices] * len(new.shape))), self.reshape(new, -1)
        )

    def left_matmul_at_modes(
        self, a_partial: Tensor, b_full: Tensor, modes: Sequence[int]
    ) -> Tensor:  # NOTE: To be deprecated (XPTensor)
        r"""Left matrix multiplication of a partial matrix and a full matrix.

        It assumes that that ``a_partial`` is a matrix operating on M modes and that ``modes`` is a
        list of ``M`` integers, i.e., it will apply ``a_partial`` on the corresponding ``M`` modes
        of ``b_full`` from the left.

        Args:
            a_partial (array): :math:`2M\times 2M` array
            b_full (array): :math:`2N\times 2N` array
            modes (list): list of ``M`` modes to perform the multiplication on

        Returns:
            array: :math:`2N\times 2N` array
        """
        if a_partial is None:
            return b_full
        N = b_full.shape[-1] // 2
        indices = self.astensor(modes + [m + N for m in modes], dtype="int32")
        b_rows = self.gather(b_full, indices, axis=0)
        b_rows = self.matmul(a_partial, b_rows)
        return self.update_tensor(b_full, indices[:, None], b_rows)

    def right_matmul_at_modes(
        self, a_full: Tensor, b_partial: Tensor, modes: Sequence[int]
    ) -> Tensor:  # NOTE: To be deprecated (XPTensor)
        r"""Right matrix multiplication of a full matrix and a partial matrix.

        It assumes that that ``b_partial`` is a matrix operating on ``M`` modes and that ``modes``
        is a list of ``M`` integers, i.e., it will apply ``b_partial`` on the corresponding M modes
        of ``a_full`` from the right.

        Args:
            a_full (array): :math:`2N\times 2N` array
            b_partial (array): :math:`2M\times 2M` array
            modes (list): list of `M` modes to perform the multiplication on

        Returns:
            array: :math:`2N\times 2N` array
        """
        return self.transpose(
            self.left_matmul_at_modes(self.transpose(b_partial), self.transpose(a_full), modes)
        )

    def matvec_at_modes(
        self, mat: Optional[Tensor], vec: Tensor, modes: Sequence[int]
    ) -> Tensor:  # NOTE: To be deprecated (XPTensor)
        """Matrix-vector multiplication between a phase-space matrix and a vector in the specified modes."""
        if mat is None:
            return vec
        N = vec.shape[-1] // 2
        indices = self.astensor(modes + [m + N for m in modes], dtype="int32")
        updates = self.matvec(mat, self.gather(vec, indices, axis=0))
        return self.update_tensor(vec, indices[:, None], updates)

    def all_diagonals(self, rho: Tensor, real: bool) -> Tensor:
        """Returns all the diagonals of a density matrix."""
        cutoffs = rho.shape[: rho.ndim // 2]
        rho = self.reshape(rho, (int(np.prod(cutoffs)), int(np.prod(cutoffs))))
        diag = self.diag_part(rho)
        if real:
            return self.real(self.reshape(diag, cutoffs))

        return self.reshape(diag, cutoffs)

    def poisson(self, max_k: int, rate: Tensor) -> Tensor:
        """Poisson distribution up to ``max_k``."""
        k = self.arange(max_k)
        rate = self.cast(rate, k.dtype)
        return self.exp(k * self.log(rate + 1e-9) - rate - self.lgamma(k + 1.0))

    def binomial_conditional_prob(self, success_prob: Tensor, dim_out: int, dim_in: int):
        """:math:`P(out|in) = binom(in, out) * (1-success_prob)**(in-out) * success_prob**out`."""
        in_ = self.arange(dim_in)[None, :]
        out_ = self.arange(dim_out)[:, None]
        return (
            self.cast(binom(in_, out_), in_.dtype)
            * self.pow(success_prob, out_)
            * self.pow(1.0 - success_prob, self.maximum(in_ - out_, 0.0))
        )

    def convolve_probs_1d(self, prob: Tensor, other_probs: List[Tensor]) -> Tensor:
        """Convolution of a joint probability with a list of single-index probabilities."""

        if prob.ndim > 3 or len(other_probs) > 3:
            raise ValueError("cannot convolve arrays with more than 3 axes")
        if not all((q.ndim == 1 for q in other_probs)):
            raise ValueError("other_probs must contain 1d arrays")
        if not all((len(q) == s for q, s in zip(other_probs, prob.shape))):
            raise ValueError("The length of the 1d prob vectors must match shape of prob")

        q = other_probs[0]
        for q_ in other_probs[1:]:
            q = q[..., None] * q_[(None,) * q.ndim + (slice(None),)]

        return self.convolve_probs(prob, q)

    def convolve_probs(self, prob: Tensor, other: Tensor) -> Tensor:
        r"""Convolve two probability distributions (up to 3D) with the same shape.

        Note that the output is not guaranteed to be a complete joint probability,
        as it's computed only up to the dimension of the base probs.
        """

        if prob.ndim > 3 or other.ndim > 3:
            raise ValueError("cannot convolve arrays with more than 3 axes")
        if not prob.shape == other.shape:
            raise ValueError("prob and other must have the same shape")

        prob_padded = self.pad(prob, [(s - 1, 0) for s in other.shape])
        other_reversed = other[(slice(None, None, -1),) * other.ndim]
        return self.convolution(
            prob_padded[None, ..., None],
            other_reversed[..., None, None],
            padding="VALID",  # TODO: do we need to specify this?
            data_format="N"
            + ("HD"[: other.ndim - 1])[::-1]
            + "WC",  # TODO: rewrite this to be more readable (do we need it?)
        )[0, ..., 0]

    def euclidean_to_symplectic(self, S: Matrix, dS_euclidean: Matrix) -> Matrix:
        r"""Convert the Euclidean gradient to a Riemannian gradient on the
        tangent bundle of the symplectic manifold.

        Implemented from:
            Wang J, Sun H, Fiori S. A Riemannian‐steepest‐descent approach
            for optimization on the real symplectic group.
            Mathematical Methods in the Applied Sciences. 2018 Jul 30;41(11):4273-86.

        Args:
            S (Matrix): symplectic matrix
            dS_euclidean (Matrix): Euclidean gradient tensor

        Returns:
            Matrix: symplectic gradient tensor
        """
        Jmat = self.J(S.shape[-1] // 2)
        Z = self.matmul(self.transpose(S), dS_euclidean)
        return 0.5 * (Z + self.matmul(self.matmul(Jmat, self.transpose(Z)), Jmat))

    def euclidean_to_unitary(self, U: Matrix, dU_euclidean: Matrix) -> Matrix:
        r"""Convert the Euclidean gradient to a Riemannian gradient on the
        tangent bundle of the unitary manifold.

        Implemented from:
            Y Yao, F Miatto, N Quesada - arXiv preprint arXiv:2209.06069, 2022.

        Args:
            U (Matrix): unitary matrix
            dU_euclidean (Matrix): Euclidean gradient tensor

        Returns:
            Matrix: unitary gradient tensor
        """
        Z = self.matmul(self.conj(self.transpose(U)), dU_euclidean)
        return 0.5 * (Z - self.conj(self.transpose(Z)))
