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

"""This module contains the backend manager."""


import importlib.util
import sys
from functools import lru_cache
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import binom
from scipy.stats import ortho_group, unitary_group

from ..utils.settings import settings
from ..utils.typing import (
    Batch,
    Matrix,
    Tensor,
    Trainable,
    Vector,
)
from .backend_base import BackendBase
from .backend_numpy import BackendNumpy

__all__ = [
    "BackendManager",
]

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
module_name_np = "mrmustard.math.backend_numpy"
module_np, loader_np = lazy_import(module_name_np)

# lazy import for tensorflow
module_name_tf = "mrmustard.math.backend_tensorflow"
module_tf, loader_tf = lazy_import(module_name_tf)

all_modules = {
    "numpy": {"module": module_np, "loader": loader_np, "object": "BackendNumpy"},
    "tensorflow": {
        "module": module_tf,
        "loader": loader_tf,
        "object": "BackendTensorflow",
    },
}


class BackendManager:  # pylint: disable=too-many-public-methods, fixme
    r"""
    A class to manage the different backends supported by Mr Mustard.
    """

    # the backend in use, which is numpy by default
    _backend = BackendNumpy()

    # the configured Euclidean optimizer.
    _euclidean_opt: Optional[type] = None

    # whether or not the backend can be changed
    _is_immutable = False

    def __init__(self) -> None:
        # binding types and decorators of numpy backend
        self._bind()

    def _apply(self, fn: str, args: Optional[Sequence[Any]] = ()) -> Any:
        r"""
        Applies a function ``fn`` from the backend in use to the given ``args``.
        """
        try:
            attr = getattr(self.backend, fn)
        except AttributeError:
            msg = f"Function ``{fn}`` not implemented for backend ``{self.backend_name}``."
            # pylint: disable=raise-missing-from
            raise NotImplementedError(msg)
        return attr(*args)

    def _bind(self) -> None:
        r"""
        Binds the types and decorators of this backend manager to those of the given ``self._backend``.
        """
        for name in [
            "int32",
            "float32",
            "float64",
            "complex64",
            "complex128",
            "hermite_renormalized",
            "hermite_renormalized_binomial",
            "hermite_renormalized_diagonal_reorderedAB",
            "hermite_renormalized_1leftoverMode_reorderedAB",
        ]:
            setattr(self, name, getattr(self._backend, name))

    def __new__(cls):
        # singleton
        try:
            return cls.instance
        except AttributeError:
            cls.instance = super(BackendManager, cls).__new__(cls)
            return cls.instance

    def __repr__(self) -> str:
        return f"Backend({self.backend_name})"

    @property
    def backend(self) -> BackendBase:
        r"""
        The backend that is being used.
        """
        self._is_immutable = True
        return self._backend

    @property
    def backend_name(self) -> str:
        r"""
        The name of the backend in use.
        """
        return self._backend.name

    def change_backend(self, name: str) -> None:
        r"""
        Changes the backend to a different one.

        Args:
            name: The name of the new backend.
        """
        if name not in ["numpy", "tensorflow"]:
            msg = "Backend must be either ``numpy`` or ``tensorflow``"
            raise ValueError(msg)

        if self.backend_name != name:
            if self._is_immutable:
                msg = "Can no longer change the backend in this session."
                raise ValueError(msg)

            module = all_modules[name]["module"]
            object = all_modules[name]["object"]
            try:
                backend = getattr(module, object)()
            except AttributeError:
                # lazy import
                loader = all_modules[name]["loader"]
                loader.exec_module(module)
                backend = getattr(module, object)()

            # switch backend
            self._backend = backend

            # bind
            self._bind()

    # ~~~~~~~
    # Methods
    # ~~~~~~~
    # Below are the methods supported by the various backends.

    def abs(self, array: Tensor) -> Tensor:
        r"""The absolute value of array.

        Args:
            array: The array to take the absolute value of.

        Returns:
            The absolute value of the given ``array``.
        """
        return self._apply("abs", (array,))

    def allclose(self, array1: Tensor, array2: Tensor, atol=1e-9) -> bool:
        r"""
        Whether two arrays are equal within tolerance.

        The two arrays are compaired element-wise.

        Args:
            array1: An array.
            array2: Another array.
            atol: The absolute tolerance.

        Returns:
            Whether two arrays are equal within tolerance.

        Raises:
            ValueError: If the shape of the two arrays do not match.
        """
        return self._apply("allclose", (array1, array2, atol))

    def any(self, array: Tensor) -> bool:
        r"""Returns ``True`` if any element of array is ``True``, ``False`` otherwise.

        Args:
            array: The array to check.

        Returns:
            ``True`` if any element of array is ``True``, ``False`` otherwise.
        """
        return self._apply("any", (array,))

    def arange(self, start: int, limit: int = None, delta: int = 1, dtype: Any = None) -> Tensor:
        r"""Returns an array of evenly spaced values within a given interval.

        Args:
            start: The start of the interval.
            limit: The end of the interval.
            delta: The step size.
            dtype: The dtype of the returned array.

        Returns:
            The array of evenly spaced values.
        """
        # NOTE: is float64 by default
        return self._apply("arange", (start, limit, delta, dtype))

    def asnumpy(self, tensor: Tensor) -> Tensor:
        r"""Converts an array to a numpy array.

        Args:
            tensor: The tensor to convert.

        Returns:
            The corresponidng numpy array.
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

    def astensor(self, array: Tensor, dtype=None):
        r"""Converts a numpy array to a tensor.

        Args:
            array: The numpy array to convert.
            dtype: The dtype of the tensor.  If ``None``, the returned tensor
                is of type ``float``.

        Returns:
            The tensor with dtype.
        """
        return self._apply("astensor", (array, dtype))

    def atleast_1d(self, array: Tensor, dtype=None) -> Tensor:
        r"""Returns an array with at least one dimension.

        Args:
            array: The array to convert.
            dtype: The data type of the array. If ``None``, the returned array
                is of the same type as the given one.

        Returns:
            The array with at least one dimension.
        """
        return self._apply("atleast_1d", (array, dtype))

    def atleast_2d(self, array: Tensor, dtype=None) -> Tensor:
        r"""Returns an array with at least two dimensions.

        Args:
            array: The array to convert.
            dtype: The data type of the array. If ``None``, the returned array
                is of the same type as the given one.

        Returns:
            The array with at least two dimensions.
        """
        return self._apply("atleast_2d", (array, dtype))

    def atleast_3d(self, array: Tensor, dtype=None) -> Tensor:
        r"""Returns an array with at least three dimensions by eventually inserting
        new axes at the beginning. Note this is not the way atleast_3d works in numpy
        and tensorflow, where it adds at the beginning and/or end.

        Args:
            array: The array to convert.
            dtype: The data type of the array. If ``None``, the returned array
                is of the same type as the given one.

        Returns:
            The array with at least three dimensions.
        """
        return self._apply("atleast_3d", (array, dtype))

    def block_diag(self, mat1: Matrix, mat2: Matrix) -> Matrix:
        r"""Returns a block diagonal matrix from the given matrices.

        Args:
            mat1: A matrix.
            mat2: A matrix.

        Returns:
            A block diagonal matrix from the given matrices.
        """
        return self._apply("block_diag", (mat1, mat2))

    def boolean_mask(self, tensor: Tensor, mask: Tensor) -> Tensor:
        """
        Returns a tensor based on the truth value of the boolean mask.

        Args:
            tensor: A tensor.
            mask: A boolean mask.

        Returns:
            A tensor based on the truth value of the boolean mask.
        """
        return self._apply("boolean_mask", (tensor, mask))

    def block(self, blocks: List[List[Tensor]], axes=(-2, -1)) -> Tensor:
        r"""Returns a matrix made from the given blocks.

        Args:
            blocks: A list of lists of compatible blocks.
            axes: The axes to stack the blocks along.

        Returns:
            The matrix made of blocks.
        """
        return self._apply("block", (blocks, axes))

    def cast(self, array: Tensor, dtype=None) -> Tensor:
        r"""Casts ``array`` to ``dtype``.

        Args:
            array: The array to cast.
            dtype: The data type to cast to. If ``None``, the returned array
                is the same as the given one.

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
        return self._apply("conj", (array,))

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
        padding: Optional[str] = None,
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
        r"""The cosine of an array.

        Args:
            array: The array to take the cosine of.

        Returns:
            The cosine of ``array``.
        """
        return self._apply("cos", (array,))

    def cosh(self, array: Tensor) -> Tensor:
        r"""The hyperbolic cosine of array.

        Args:
            array: The array to take the hyperbolic cosine of.

        Returns:
            The hyperbolic cosine of ``array``.
        """
        return self._apply("cosh", (array,))

    def det(self, matrix: Tensor) -> Tensor:
        r"""The determinant of matrix.

        Args:
            matrix: The matrix to take the determinant of

        Returns:
            The determinant of ``matrix``.
        """
        return self._apply("det", (matrix,))

    def diag(self, array: Tensor, k: int = 0) -> Tensor:
        r"""The array made by inserting the given array along the :math:`k`-th diagonal.

        Args:
            array: The array to insert.
            k: The ``k``-th diagonal to insert array into.

        Returns:
            The array with ``array`` inserted into the ``k``-th diagonal.
        """
        return self._apply("diag", (array, k))

    def diag_part(self, array: Tensor, k: int = 0) -> Tensor:
        r"""The array of the main diagonal of array.

        Args:
            array: The array to extract the main diagonal of.
            k: The diagonal to extract.

        Returns:
            The array of the main diagonal of ``array``.
        """
        return self._apply("diag_part", (array, k))

    def eigvals(self, tensor: Tensor) -> Tensor:
        r"""The eigenvalues of a tensor.

        Args:
            tensor: The tensor to calculate the eigenvalues of.

        Returns:
            The eigenvalues of ``tensor``.
        """
        return self._apply("eigvals", (tensor,))

    def eigh(self, tensor: Tensor) -> Tensor:
        """
        The eigenvalues and eigenvectors of a matrix.

        Args:
            tensor: The tensor to calculate the eigenvalues and eigenvectors of.

        Returns:
            The eigenvalues and eigenvectors of ``tensor``.
        """
        return self._apply("eigh", (tensor,))

    def einsum(self, string: str, *tensors) -> Tensor:
        r"""The result of the Einstein summation convention on the tensors.

        Args:
            string: The string of the Einstein summation convention.
            tensors: The tensors to perform the Einstein summation on.

        Returns:
            The result of the Einstein summation convention.
        """
        return self._apply("einsum", (string, *tensors))

    def exp(self, array: Tensor) -> Tensor:
        r"""The exponential of array element-wise.

        Args:
            array: The array to take the exponential of.

        Returns:
            The exponential of array.
        """
        return self._apply("exp", (array,))

    def expand_dims(self, array: Tensor, axis: int) -> Tensor:
        r"""The array with an additional dimension inserted at the given axis.

        Args:
            array: The array to expand.
            axis: The axis to insert the new dimension.

        Returns:
            The array with an additional dimension inserted at the given axis.
        """
        return self._apply("expand_dims", (array, axis))

    def expm(self, matrix: Tensor) -> Tensor:
        r"""The matrix exponential of matrix.

        Args:
            matrix: The matrix to take the exponential of.

        Returns:
            The exponential of ``matrix``.
        """
        return self._apply("expm", (matrix,))

    def eye(self, size: int, dtype=None) -> Tensor:
        r"""The identity matrix of size.

        Args:
            size: The size of the identity matrix
            dtype: The data type of the identity matrix. If ``None``,
                the returned matrix is of type ``float``.

        Returns:
            The identity matrix.
        """
        return self._apply("eye", (size, dtype))

    def eye_like(self, array: Tensor) -> Tensor:
        r"""The identity matrix of the same shape and dtype as array.

        Args:
            array: The array to create the identity matrix of.

        Returns:
            The identity matrix.
        """
        return self._apply("eye_like", (array,))

    def from_backend(self, value: Any) -> bool:
        r"""Whether the given tensor is a tensor of the concrete backend.

        Args:
            value: A value.

        Returns:
            Whether given ``value`` is a tensor of the concrete backend.
        """
        return self._apply("from_backend", (value,))

    def gather(self, array: Tensor, indices: Batch[int], axis: Optional[int] = None) -> Tensor:
        r"""The values of the array at the given indices.

        Args:
            array: The array to gather values from.
            indices: The indices to gather values from.
            axis: The axis to gather values from.

        Returns:
            The values of the array at the given indices.
        """
        return self._apply(
            "gather",
            (
                array,
                indices,
                axis,
            ),
        )

    def hermite_renormalized_batch(
        self, A: Tensor, B: Tensor, C: Tensor, shape: Tuple[int]
    ) -> Tensor:
        r"""Renormalized multidimensional Hermite polynomial given by the "exponential" Taylor
        series of :math:`exp(C + Bx + 1/2*Ax^2)` at zero, where the series has :math:`sqrt(n!)`
        at the denominator rather than :math:`n!`. It computes all the amplitudes within the
        tensor of given shape in case of B is a batched vector with a batched diemnsion on the
        last index.

        Args:
            A: The A matrix.
            B: The batched B vector with its batch dimension on the last index.
            C: The C scalar.
            shape: The shape of the final tensor.

        Returns:
            The batched Hermite polynomial of given shape.
        """
        return self._apply("hermite_renormalized_batch", (A, B, C, shape))

    def hermite_renormalized_diagonal(
        self, A: Tensor, B: Tensor, C: Tensor, cutoffs: Tuple[int]
    ) -> Tensor:
        r"""Firsts, reorder A and B parameters of Bargmann representation to match conventions in mrmustard.math.compactFock~
        Then, calculates the required renormalized multidimensional Hermite polynomial.
        """
        return self._apply("hermite_renormalized_diagonal", (A, B, C, cutoffs))

    def hermite_renormalized_diagonal_batch(
        self, A: Tensor, B: Tensor, C: Tensor, cutoffs: Tuple[int]
    ) -> Tensor:
        r"""First, reorder A and B parameters of Bargmann representation to match conventions in mrmustard.math.compactFock~
        Then, calculates the required renormalized multidimensional Hermite polynomial.
        Same as hermite_renormalized_diagonal but works for a batch of different B's."""
        return self._apply("hermite_renormalized_diagonal_batch", (A, B, C, cutoffs))

    def hermite_renormalized_1leftoverMode(
        self, A: Tensor, B: Tensor, C: Tensor, cutoffs: Tuple[int]
    ) -> Tensor:
        r"""First, reorder A and B parameters of Bargmann representation to match conventions in mrmustard.math.compactFock~
        Then, calculate the required renormalized multidimensional Hermite polynomial.
        """
        return self._apply("hermite_renormalized_1leftoverMode", (A, B, C, cutoffs))

    def imag(self, array: Tensor) -> Tensor:
        r"""The imaginary part of array.

        Args:
            array: The array to take the imaginary part of

        Returns:
            The imaginary part of array
        """
        return self._apply("imag", (array,))

    def inv(self, tensor: Tensor) -> Tensor:
        r"""The inverse of tensor.

        Args:
            tensor: The tensor to take the inverse of

        Returns:
            The inverse of tensor
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
            x: The array to take the natural logarithm of the gamma function of

        Returns:
            The natural logarithm of the gamma function of ``x``
        """
        return self._apply("lgamma", (x,))

    def log(self, x: Tensor) -> Tensor:
        r"""The natural logarithm of ``x``.

        Args:
            x: The array to take the natural logarithm of

        Returns:
            The natural logarithm of ``x``
        """
        return self._apply("log", (x,))

    def make_complex(self, real: Tensor, imag: Tensor) -> Tensor:
        """Given two real tensors representing the real and imaginary part of a complex number,
        this operation returns a complex tensor. The input tensors must have the same shape.

        Args:
            real: The real part of the complex number.
            imag: The imaginary part of the complex number.

        Returns:
            The complex array ``real + 1j * imag``.
        """
        return self._apply("make_complex", (real, imag))

    def matmul(self, *matrices: Matrix) -> Tensor:
        r"""The matrix product of the given matrices.

        Args:
            matrices: The matrices to multiply.

        Returns:
            The matrix product
        """
        return self._apply("matmul", matrices)

    def matvec(self, a: Matrix, b: Vector) -> Tensor:
        r"""The matrix vector product of ``a`` (matrix) and ``b`` (vector).

        Args:
            a: The matrix to multiply
            b: The vector to multiply

        Returns:
            The matrix vector product of ``a`` and ``b``
        """
        return self._apply("matvec", (a, b))

    def maximum(self, a: Tensor, b: Tensor) -> Tensor:
        r"""The element-wise maximum of ``a`` and ``b``.

        Args:
            a: The first array to take the maximum of
            b: The second array to take the maximum of

        Returns:
            The element-wise maximum of ``a`` and ``b``
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
            a: The first array to take the minimum of
            b: The second array to take the minimum of

        Returns:
            The element-wise minimum of ``a`` and ``b``
        """
        return self._apply(
            "minimum",
            (
                a,
                b,
            ),
        )

    def moveaxis(self, array: Tensor, old: Tensor, new: Tensor) -> Tensor:
        r"""
        Moves the axes of an array to a new position.

        Args:
            array: The array to move the axes of.
            old: The old index position
            new: The new index position


        Returns:
            The updated array
        """
        return self._apply(
            "moveaxis",
            (
                array,
                old,
                new,
            ),
        )

    def new_variable(
        self,
        value: Tensor,
        bounds: Tuple[Optional[float], Optional[float]],
        name: str,
        dtype=None,
    ) -> Tensor:
        r"""Returns a new variable with the given value and bounds.

        Args:
            value: The value of the new variable.
            bounds: The bounds of the new variable.
            name: The name of the new variable.
            dtype: dtype of the new variable. If ``None``, casts it to float.
        Returns:
            The new variable.
        """
        return self._apply("new_variable", (value, bounds, name, dtype))

    def new_constant(self, value: Tensor, name: str, dtype=None) -> Tensor:
        r"""Returns a new constant with the given value.

        Args:
            value: The value of the new constant
            name (str): name of the new constant
            dtype (type): dtype of the array

        Returns:
            The new constant
        """
        return self._apply("new_constant", (value, name, dtype))

    def norm(self, array: Tensor) -> Tensor:
        r"""The norm of array.

        Args:
            array: The array to take the norm of

        Returns:
            The norm of array
        """
        return self._apply("norm", (array,))

    def ones(self, shape: Sequence[int], dtype=None) -> Tensor:
        r"""Returns an array of ones with the given ``shape`` and ``dtype``.

        Args:
            shape (tuple): shape of the array
            dtype (type): dtype of the array. If ``None``, the returned array is
                of type ``float``.

        Returns:
            The array of ones
        """
        # NOTE : should be float64 by default
        return self._apply("ones", (shape, dtype))

    def ones_like(self, array: Tensor) -> Tensor:
        r"""Returns an array of ones with the same shape and ``dtype`` as ``array``.

        Args:
            array: The array to take the shape and dtype of

        Returns:
            The array of ones
        """
        return self._apply("ones_like", (array,))

    def outer(self, array1: Tensor, array2: Tensor) -> Tensor:
        r"""The outer product of ``array1`` and ``array2``.

        Args:
            array1: The first array to take the outer product of
            array2: The second array to take the outer product of

        Returns:
            The outer product of array1 and array2
        """
        return self._apply("outer", (array1, array2))

    def pad(
        self,
        array: Tensor,
        paddings: Sequence[Tuple[int, int]],
        mode="CONSTANT",
        constant_values=0,
    ) -> Tensor:
        r"""The padded array.

        Args:
            array: The array to pad
            paddings (tuple): paddings to apply
            mode (str): mode to apply the padding
            constant_values (int): constant values to use for padding

        Returns:
            The padded array
        """
        return self._apply("pad", (array, paddings, mode, constant_values))

    def pinv(self, matrix: Tensor) -> Tensor:
        r"""The pseudo-inverse of matrix.

        Args:
            matrix: The matrix to take the pseudo-inverse of

        Returns:
            The pseudo-inverse of matrix
        """
        return self._apply("pinv", (matrix,))

    def pow(self, x: Tensor, y: Tensor) -> Tensor:
        r"""Returns :math:`x^y`. Broadcasts ``x`` and ``y`` if necessary.
        Args:
            x: The base
            y: The exponent

        Returns:
            The :math:`x^y`
        """
        return self._apply("pow", (x, y))

    def kron(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
        r"""
        The Kroenecker product of the given tensors.

        Args:
            tensor1: A tensor.
            tensor2: Another tensor.

        Returns:
            The Kroenecker product.
        """
        return self._apply("kron", (tensor1, tensor2))

    def prod(self, array: Tensor, axis=None) -> Tensor:
        r"""
        The product of all elements in ``array``.

        Args:
            array: The array of elements to calculate the product of.
            axis: The axis along which a product is performed. If ``None``, it calculates
                the product of all elements in ``array``.

        Returns:
            The product of the elements in ``array``.
        """
        return self._apply("prod", (array, axis))

    def real(self, array: Tensor) -> Tensor:
        r"""The real part of ``array``.

        Args:
            array: The array to take the real part of

        Returns:
            The real part of ``array``
        """
        return self._apply("real", (array,))

    def reshape(self, array: Tensor, shape: Sequence[int]) -> Tensor:
        r"""The reshaped array.

        Args:
            array: The array to reshape
            shape (tuple): shape to reshape the array to

        Returns:
            The reshaped array
        """
        return self._apply("reshape", (array, shape))

    def round(self, array: Tensor, decimals: int) -> Tensor:
        r"""The array rounded to the nearest integer.

        Args:
            array: The array to round
            decimals: number of decimals to round to

        Returns:
            The array rounded to the nearest integer
        """
        return self._apply("round", (array, decimals))

    def set_diag(self, array: Tensor, diag: Tensor, k: int) -> Tensor:
        r"""The array with the diagonal set to ``diag``.

        Args:
            array: The array to set the diagonal of
            diag: The diagonal to set
            k (int): diagonal to set

        Returns:
            The array with the diagonal set to ``diag``
        """
        return self._apply("set_diag", (array, diag, k))

    def sin(self, array: Tensor) -> Tensor:
        r"""The sine of ``array``.

        Args:
            array: The array to take the sine of

        Returns:
            The sine of ``array``
        """
        return self._apply("sin", (array,))

    def sinh(self, array: Tensor) -> Tensor:
        r"""The hyperbolic sine of ``array``.

        Args:
            array: The array to take the hyperbolic sine of

        Returns:
            The hyperbolic sine of ``array``
        """
        return self._apply("sinh", (array,))

    def solve(self, matrix: Tensor, rhs: Tensor) -> Tensor:
        r"""The solution of the linear system :math:`Ax = b`.

        Args:
            matrix: The matrix :math:`A`
            rhs: The vector :math:`b`

        Returns:
            The solution :math:`x`
        """
        return self._apply("solve", (matrix, rhs))

    def sort(self, array: Tensor, axis: int = -1) -> Tensor:
        r"""Sort the array along an axis.

        Args:
            array: The array to sort
            axis: (optional) The axis to sort along. Defaults to last axis.

        Returns:
            A sorted version of the array in acending order.
        """
        return self._apply("sort", (array, axis))

    def sqrt(self, x: Tensor, dtype=None) -> Tensor:
        r"""The square root of ``x``.

        Args:
            x: The array to take the square root of
            dtype: ``dtype`` of the output array.

        Returns:
            The square root of ``x``
        """
        return self._apply("sqrt", (x, dtype))

    def sqrtm(self, tensor: Tensor, dtype=None) -> Tensor:
        r"""The matrix square root.

        Args:
            tensor: The tensor to take the matrix square root of.
            dtype: The ``dtype`` of the output tensor. If ``None``, the output
                is of type ``math.complex128``.

        Returns:
            The square root of ``x``"""
        return self._apply("sqrtm", (tensor, dtype))

    def sum(self, array: Tensor, axes: Sequence[int] = None):
        r"""The sum of array.

        Args:
            array: The array to take the sum of
            axes (tuple): axes to sum over

        Returns:
            The sum of array
        """
        if axes is not None:
            neg = [a for a in axes if a < 0]
            pos = [a for a in axes if a >= 0]
            axes = sorted(neg) + sorted(pos)[::-1]
        return self._apply("sum", (array, axes))

    def tensordot(self, a: Tensor, b: Tensor, axes: Sequence[int]) -> Tensor:
        r"""The tensordot product of ``a`` and ``b``.

        Args:
            a: The first array to take the tensordot product of
            b: The second array to take the tensordot product of
            axes: The axes to take the tensordot product over

        Returns:
            The tensordot product of ``a`` and ``b``
        """
        return self._apply("tensordot", (a, b, axes))

    def tile(self, array: Tensor, repeats: Sequence[int]) -> Tensor:
        r"""The tiled array.

        Args:
            array: The array to tile
            repeats (tuple): number of times to tile the array along each axis

        Returns:
            The tiled array
        """
        return self._apply("tile", (array, repeats))

    def trace(self, array: Tensor, dtype=None) -> Tensor:
        r"""The trace of array.

        Args:
            array: The array to take the trace of
            dtype (type): ``dtype`` of the output array

        Returns:
            The trace of array
        """
        return self._apply("trace", (array, dtype))

    def transpose(self, a: Tensor, perm: Sequence[int] = None):
        r"""The transposed arrays.

        Args:
            a: The array to transpose
            perm (tuple): permutation to apply to the array

        Returns:
            The transposed array
        """
        return self._apply("transpose", (a, perm))

    def update_tensor(self, tensor: Tensor, indices: Tensor, values: Tensor) -> Tensor:
        r"""Updates a tensor in place with the given values.

        Args:
            tensor: The tensor to update
            indices: The indices to update
            values: The values to update

        Returns:
            The updated tensor
        """
        return self._apply("update_tensor", (tensor, indices, values))

    def update_add_tensor(self, tensor: Tensor, indices: Tensor, values: Tensor) -> Tensor:
        r"""Updates a tensor in place by adding the given values.

        Args:
            tensor: The tensor to update
            indices: The indices to update
            values: The values to add

        Returns:
            The updated tensor
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

    def xlogy(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Returns ``0`` if ``x == 0`` elementwise and ``x * log(y)`` otherwise.
        """
        return self._apply("xlogy", (x, y))

    def zeros(self, shape: Sequence[int], dtype=None) -> Tensor:
        r"""Returns an array of zeros with the given shape and ``dtype``.

        Args:
            shape: The shape of the array.
            dtype: The dtype of the array. If ``None``, the returned array is
                of type ``float``.

        Returns:
            The array of zeros.
        """
        return self._apply("zeros", (shape, dtype))

    def zeros_like(self, array: Tensor) -> Tensor:
        r"""Returns an array of zeros with the same shape and ``dtype`` as ``array``.

        Args:
            array: The array to take the shape and ``dtype`` of.

        Returns:
            The array of zeros.
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
            probs: The unnormalized probabilities of a set of Categorical distributions.
            name: The name prefixed to operations created by this class.

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

    def custom_gradient(self, func):
        r"""
        A decorator to define a function with a custom gradient.
        """

        def wrapper(*args, **kwargs):
            if self.backend_name == "numpy":
                return func(*args, **kwargs)
            else:
                from tensorflow import (  # pylint: disable=import-outside-toplevel
                    custom_gradient,
                )

                return custom_gradient(func)(*args, **kwargs)

        return wrapper

    def DefaultEuclideanOptimizer(self):
        r"""Default optimizer for the Euclidean parameters."""
        return self._apply("DefaultEuclideanOptimizer")

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

    def dagger(self, array: Tensor) -> Tensor:
        """The adjoint of ``array``. This operation swaps the first
        and second half of the indexes and then conjugates the matrix.

        Args:
            array: The array to take the adjoint of

        Returns:
            The adjoint of ``array``
        """
        N = len(array.shape) // 2
        perm = list(range(N, 2 * N)) + list(range(0, N))
        return self.conj(self.transpose(array, perm=perm))

    def unitary_to_orthogonal(self, U):
        r"""Unitary to orthogonal mapping.

        Args:
            U: The unitary matrix in ``U(n)``

        Returns:
            The orthogonal matrix in :math:`O(2n)`
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
            The :math:`2N\times 2N` array
        """
        I = np.identity(num_modes)
        O = np.zeros((num_modes, num_modes))
        return np.block([[O, I], [I, O]])

    @staticmethod
    @lru_cache()
    def Zmat(num_modes: int):
        r"""The matrix :math:`Z_n = \begin{bmatrix}I_n & 0\\ 0 & -I_n\end{bmatrix}.`

        Args:
            num_modes: A positive integer representing the number of modes.

        Returns:
            The :math:`2N\times 2N` array
        """
        I = np.identity(num_modes)
        O = np.zeros((num_modes, num_modes))
        return np.block([[I, O], [O, -I]])

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
        shape = getattr(old, "shape", ())
        N = (shape[-1] if shape != () else 0) // 2
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
            a_partial: The :math:`2M\times 2M` array
            b_full: The :math:`2N\times 2N` array
            modes: A list of ``M`` modes to perform the multiplication on

        Returns:
            The :math:`2N\times 2N` array
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
            a_full: The :math:`2N\times 2N` array
            b_partial: The :math:`2M\times 2M` array
            modes: A list of `M` modes to perform the multiplication on

        Returns:
            The :math:`2N\times 2N` array
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
