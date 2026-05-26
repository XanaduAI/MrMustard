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

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any, Literal, cast, overload

import numpy as np
from numpy.typing import ArrayLike, DTypeLike
from opt_einsum import contract
from opt_einsum.typing import BackendType, OptimizeKind
from scipy.stats import ortho_group, unitary_group

from mrmustard import settings

from ..utils.typing import (
    BoolScalar,
    ComplexMatrix,
    ComplexScalar,
    ComplexTensor,
    ComplexVector,
    IntArrayLike,
    IntScalarValue,
    IntTensor,
    Matrix,
    RealScalar,
    RealScalarValue,
    RealTensor,
    Scalar,
    ScalarValue,
    Tensor,
    Trainable,
    Vector,
)
from .backend_base import BackendBase
from .backend_numpy import BackendNumpy
from .utils import compute_collapsed_shape, strip_parentheses

__all__ = [
    "BackendManager",
]

# ~~~~~~~
# Helpers
# ~~~~~~~


def lazy_import(module_name: str):
    r"""Returns module and loader for lazy import.

    Args:
        module_name: The name of the module to import.

    Raises:
        ValueError: If the spec or spec loader are not found.
    """
    try:
        return sys.modules[module_name], None
    except KeyError:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            raise ValueError(f"Spec {module_name} not found!") from None
        module = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            raise ValueError(f"Spec loader {module_name} not found!") from None
        loader = importlib.util.LazyLoader(spec.loader)
        return module, loader


# lazy import for numpy
module_name_np = "mrmustard.math.backend_numpy"
module_np, loader_np = lazy_import(module_name_np)

# lazy import for jax
module_name_jax = "mrmustard.math.backend_jax"
module_jax, loader_jax = lazy_import(module_name_jax)

all_modules = {
    "numpy": {"module": module_np, "loader": loader_np, "object": "BackendNumpy"},
    "jax": {
        "module": module_jax,
        "loader": loader_jax,
        "object": "BackendJax",
    },
}


class BackendManager:
    r"""A class to manage the different backends supported by Mr Mustard."""

    # defaults to help with type-hinting, e.g. so math.complex128 is available for static analysis
    int32 = np.int32
    int64 = np.int64
    float32 = np.float32
    float64 = np.float64
    complex64 = np.complex64
    complex128 = np.complex128

    # the backend in use, which is numpy by default
    _backend = BackendNumpy()

    def __init__(self) -> None:
        # binding types and decorators of numpy backend
        self._bind()

    def _apply(
        self,
        fn: str,
        args: Sequence[Any] = (),
        kwargs: dict | None = None,
        backend_name: str | None = None,
    ) -> Any:
        r"""Applies a function ``fn`` from the backend in use to the given ``args`` and ``kwargs``.

        Args:
            fn: The function to apply.
            args: The arguments to pass to the function.
            kwargs: The keyword arguments to pass to the function.
            backend_name: The name of the backend to use. If ``None``, the set backend is used.

        Returns:
            The result of the function application.
        """
        kwargs = kwargs or {}
        backend = self.get_backend(backend_name) if backend_name else self.backend
        try:
            attr = getattr(backend, fn)
        except AttributeError:
            raise NotImplementedError(
                f"Function ``{fn}`` not implemented for backend ``{backend.name}``.",
            ) from None
        return attr(*args, **kwargs)

    def _bind(self) -> None:
        r"""Binds the types and decorators of this backend manager to those of the given ``self._backend``."""
        for name in [
            "int32",
            "int64",
            "float32",
            "float64",
            "complex64",
            "complex128",
        ]:
            setattr(self, name, getattr(self._backend, name))

    def __new__(cls):
        # singleton
        try:
            return cls.instance
        except AttributeError:
            cls.instance = super().__new__(cls)
            return cls.instance

    def __repr__(self) -> str:
        return f"Backend({self.backend_name})"

    @property
    def backend(self) -> BackendBase:
        r"""The backend that is being used."""
        return self._backend

    @property
    def backend_name(self) -> str:
        r"""The name of the backend in use."""
        return self._backend.name

    @property
    def BackendError(self):
        r"""The error class for backend specific errors.

        Note that currently this only applies to the case where
        ``auto_shape`` is jitted  via the ``jax`` backend.
        """
        return self._apply("BackendError")

    def change_backend(self, name: str) -> None:
        r"""Changes the backend to a different one.

        Args:
            name: The name of the new backend.
        """
        if self.backend_name != name:
            # switch backend
            self._backend = self.get_backend(name)
            # bind
            self._bind()

    def get_backend(self, name: str | None = None) -> BackendBase:
        r"""Returns the backend with the given name.

        Args:
            name: The name of the backend.

        Returns:
            The backend with the given name.

        Raises:
            ValueError: If the backend name is not a supported one.
        """
        if name not in ["numpy", "jax"]:
            raise ValueError("Backend must be either ``numpy`` or ``jax``.")

        if self.backend_name != name:
            module = all_modules[name]["module"]
            obj = all_modules[name]["object"]
            try:
                backend = getattr(module, obj)()
            except AttributeError:
                # lazy import
                loader = all_modules[name]["loader"]
                loader.exec_module(module)
                backend = getattr(module, obj)()
        else:
            backend = self.backend

        return backend

    # ~~~~~~~
    # Methods
    # ~~~~~~~
    # Below are the methods supported by the various backends.

    def abs(self, array: ArrayLike) -> RealTensor:
        r"""The absolute value of array.

        Args:
            array: The array to take the absolute value of.

        Returns:
            The absolute value of the given ``array``.
        """
        return self._apply("abs", (array,))

    def all(self, array: ArrayLike) -> bool:
        r"""Returns ``True`` if all elements of array are ``True``, ``False`` otherwise.

        Args:
            array: The array to check.

        Returns:
            ``True`` if all elements of array are ``True``, ``False`` otherwise.
        """
        return self._apply("all", (array,))

    def allclose(
        self, array1: ArrayLike, array2: ArrayLike, atol: float = 1e-9, rtol: float = 1e-5
    ) -> bool:
        r"""Whether two arrays are equal within tolerance.

        The two arrays are compaired element-wise.

        Args:
            array1: An array.
            array2: Another array.
            atol: The absolute tolerance.
            rtol: The relative tolerance.

        Returns:
            Whether two arrays are equal within tolerance.

        Raises:
            ValueError: If the shape of the two arrays do not match.
        """
        return self._apply("allclose", (array1, array2, atol, rtol))

    def angle(self, array: ArrayLike) -> RealScalarValue | RealTensor:
        r"""The complex phase of ``array``.

        Args:
            array: The array to take the complex phase of.

        Returns:
            The complex phase of ``array``.
        """
        return self._apply("angle", (array,))

    def any(self, array: ArrayLike) -> bool:
        r"""Returns ``True`` if any element of array is ``True``, ``False`` otherwise.

        Args:
            array: The array to check.

        Returns:
            ``True`` if any element of array is ``True``, ``False`` otherwise.
        """
        return self._apply("any", (array,))

    def arange(
        self,
        start: int,
        limit: int | None = None,
        delta: int = 1,
        dtype: Any = None,
    ) -> IntTensor:
        r"""Returns an array of evenly spaced values within a given interval.

        Args:
            start: The start of the interval.
            limit: The end of the interval.
            delta: The step size.
            dtype: The dtype of the returned array.

        Returns:
            The array of evenly spaced values.
        """
        return self._apply("arange", (start, limit, delta, dtype))

    def argmax(self, array: ArrayLike, axis: int | None = None) -> IntScalarValue | IntTensor:
        r"""The indices of the maximum values along an axis.

        Args:
            array: The array to find the maximum indices of
            axis: The axis along which to find the maximum indices. If ``None``, the array is flattened.

        Returns:
            The indices of the maximum values
        """
        return self._apply("argmax", (array, axis))

    def argmin(self, array: ArrayLike, axis: int | None = None) -> IntScalarValue | IntTensor:
        r"""The indices of the minimum values along an axis.

        Args:
            array: The array to find the minimum indices of
            axis: The axis along which to find the minimum indices. If ``None``, the array is flattened.

        Returns:
            The indices of the minimum values
        """
        return self._apply("argmin", (array, axis))

    def argsort(self, array: ArrayLike, axis: int | None = None) -> IntTensor:
        r"""The indices that would sort an array along an axis.

        Args:
            array: The array to sort.
            axis: The axis along which to sort. If ``None``, the array is flattened before sorting.

        Returns:
            The indices that sort the array.
        """
        return self._apply("argsort", (array, axis))

    @overload
    def asnumpy[Arr: np.ndarray](self, tensor: Arr) -> Arr: ...
    @overload
    def asnumpy(self, tensor: ArrayLike) -> Tensor: ...
    def asnumpy(self, tensor: ArrayLike) -> Tensor:
        r"""Converts an array to a numpy array.

        Args:
            tensor: The tensor to convert.

        Returns:
            The corresponding numpy array.
        """
        return self._apply("asnumpy", (tensor,))

    @overload
    def astensor(self, array: ArrayLike, dtype: type[np.floating]) -> RealTensor: ...
    @overload
    def astensor(self, array: ArrayLike, dtype: type[np.complexfloating]) -> ComplexTensor: ...
    @overload
    def astensor(self, array: ArrayLike, dtype: type[np.signedinteger]) -> IntTensor: ...
    @overload
    def astensor(self, array: ArrayLike, dtype: DTypeLike | None = None) -> Tensor: ...
    def astensor(self, array, dtype=None):
        r"""Converts a scalar or array-like input to a tensor.

        Args:
            array: The scalar or array-like input to convert.
            dtype: The dtype of the tensor.  If ``None``, the returned tensor
                is of type ``float``.

        Returns:
            The tensor with dtype.
        """
        return self._apply("astensor", (array, dtype))

    @overload
    def atleast_nd(self, array: ArrayLike, n: int, dtype: type[np.floating]) -> RealTensor: ...
    @overload
    def atleast_nd(
        self, array: ArrayLike, n: int, dtype: type[np.complexfloating]
    ) -> ComplexTensor: ...
    @overload
    def atleast_nd(self, array: ArrayLike, n: int, dtype: type[np.signedinteger]) -> IntTensor: ...
    @overload
    def atleast_nd(self, array: ArrayLike, n: int, dtype: DTypeLike | None = None) -> Tensor: ...
    def atleast_nd(self, array: ArrayLike, n: int, dtype: DTypeLike | None = None) -> Tensor:
        r"""Returns an array with at least n dimensions. Note that dimensions are
        prepended to meet the minimum number of dimensions.

        Args:
            array: The array to convert.
            n: The minimum number of dimensions.
            dtype: The data type of the array. If ``None``, the returned array
                is of the same type as the given one.

        Returns:
            The array with at least n dimensions.
        """
        return self._apply("atleast_nd", (array, n, dtype))

    def block(self, blocks: list[list[ArrayLike]], axes: tuple[int, ...] = (-2, -1)) -> Tensor:
        r"""Returns a matrix made from the given blocks.

        Args:
            blocks: A list of lists of compatible blocks.
            axes: The axes to stack the blocks along.

        Returns:
            The matrix made of blocks.
        """
        rows = [self.concat(row, axis=axes[-1]) for row in blocks]
        return self.concat(rows, axis=axes[-2])

    def broadcast_arrays(self, *arrays: list[ArrayLike]) -> list[Tensor]:
        r"""Broadcast arrays to a common shape.

        Args:
            *arrays: The arrays to broadcast.

        Returns:
            A list of broadcasted arrays.
        """
        return self._apply("broadcast_arrays", arrays)

    def broadcast_to(self, array: ArrayLike, shape: tuple[int, ...]) -> Tensor:
        r"""Broadcasts an array to a new shape.

        Args:
            array: The array to broadcast.
            shape: The shape to broadcast to.

        Returns:
            The broadcasted array.
        """
        return self._apply("broadcast_to", (array, shape))

    @overload
    def cast(self, array: ArrayLike, dtype: type[np.floating]) -> RealTensor: ...
    @overload
    def cast(self, array: ArrayLike, dtype: type[np.complexfloating]) -> ComplexTensor: ...
    @overload
    def cast(self, array: ArrayLike, dtype: type[np.signedinteger]) -> IntTensor: ...
    @overload
    def cast(self, array: ArrayLike, dtype: DTypeLike | None = None) -> Tensor: ...
    def cast(self, array: ArrayLike, dtype: DTypeLike | None = None) -> Tensor:
        r"""Casts ``array`` to ``dtype``.

        Args:
            array: The array to cast.
            dtype: The data type to cast to. If ``None``, the returned array
                is the same as the given one.

        Returns:
            The array cast to dtype.
        """
        return self._apply("cast", (array, dtype))

    @overload
    def clip[Arr: np.ndarray](
        self, array: Arr, a_min: ArrayLike | None, a_max: ArrayLike | None = None
    ) -> Arr: ...
    @overload
    def clip(
        self, array: ArrayLike, a_min: ArrayLike | None, a_max: ArrayLike | None = None
    ) -> Tensor: ...
    def clip(
        self, array: ArrayLike, a_min: ArrayLike | None = None, a_max: ArrayLike | None = None
    ) -> Tensor:
        r"""Clips array to the interval ``[a_min, a_max]``.

        Args:
            array: The array to clip.
            a_min: The minimum value.
            a_max: The maximum value.

        Returns:
            The clipped array.
        """
        return self._apply("clip", (array, a_min, a_max))

    def complex_gaussian_integral_1(
        self,
        A: ComplexMatrix,
        b: ComplexVector,
        idx12: ArrayLike,
        A_out: ComplexMatrix | None = None,
        b_out: ComplexVector | None = None,
        log_c_out: ComplexTensor | None = None,
    ) -> tuple[ComplexMatrix, ComplexVector, ComplexTensor]:
        r"""Computes the complex Gaussian integral.

        In particular,

        .. math::
            \int_{C^m} d\mu(z) \exp(\frac{1}{2}(z,z^*,\beta)^T A (z,z^*,\beta) + (z,z^*,\beta)^T b),

        where :math:`z\in\mathbb{C}^{m}`, :math:`\beta\in\mathbb{C}^{N}` and the integration measure is given by
        :math:`d\mu(z) = \exp(-|z|^2) \frac{d^{2m}z}{\pi^m} = \frac{1}{\pi^m}\exp(-|z|^2) d\mathrm{Re}(z) d\mathrm{Im}(z)`.

        If we partition A and b into idx12 blocks and the remaining blocks:

        .. math::
            A = \begin{pmatrix} B & C^T \\ C & D \end{pmatrix},\quad
            b = \begin{pmatrix} g \\ h \end{pmatrix},

        the result is given by:

        .. math::
            A_{\mathrm{out}} = D - C M^{-1} C^T, \\
            b_{\mathrm{out}} = h - C M^{-1} g, \\
            \log c_{\mathrm{out}} = -\frac{1}{2} g^T M^{-1} g + \frac{1}{2} \log(\det(iM^{-1}))

        where :math:`M = B - X`, :math:`X` is the block with zeros on the diagonal and identities on the off-diagonal.

        This function supports broadcasted/batched inputs, however note that this function is about one order of magnitude faster when
        the inputs are not batched versus being batched with size 1, so it's recommended to squeeze the batch dimension before calling this function.

        Arguments:
            A: The A matrix.
            b: The b vector.
            idx12: the indices of the z variables to integrate over. 
                The first half is the z variables and the second half is the z* variables.
            A_out: The (optional) output A matrix.
            b_out: The (optional) output b vector.
            log_c_out: The (optional) output log_c vector.

        Returns:
            The ``(A_out, b_out, log_c_out)`` triple which parametrizes the result of the integral with eventual batch dimensions.
        """
        idx12 = np.array(idx12, dtype=np.int64)
        if A.ndim == 2 and b.ndim == 1:
            return self._apply(
                "complex_gaussian_integral_1_single", (A, b, idx12, A_out, b_out, log_c_out)
            )
        return self._apply(
            "complex_gaussian_integral_1_batched", (A, b, idx12, A_out, b_out, log_c_out)
        )

    def complex_gaussian_integral_2(
        self,
        A1: ComplexMatrix,
        b1: ComplexVector,
        A2: ComplexMatrix,
        b2: ComplexVector,
        idx1: ArrayLike,
        idx2: ArrayLike,
        A_out: ComplexMatrix | None = None,
        b_out: ComplexVector | None = None,
        log_c_out: ComplexTensor | None = None,
    ) -> tuple[ComplexMatrix, ComplexVector, ComplexTensor]:
        r"""Computes the complex Gaussian integral.

        In particular:

        .. math::
            \int_{C^m} d\mu(z) 
            \exp\!\left(\tfrac{1}{2}(z,\beta)^T A_1 (z,\beta) + (z,\beta)^T b_1\right)
            \exp\!\left(\tfrac{1}{2}(z^*,\gamma)^T A_2 (z^*,\gamma) + (z^*,\gamma)^T b_2\right),

        where :math:`z\in\mathbb{C}^{m}`, :math:`\beta\in\mathbb{C}^{N_1}`, :math:`\gamma\in\mathbb{C}^{N_2}`
        and the integration measure is given by
        :math:`d\mu(z) = \exp(-|z|^2) \frac{d^{2m}z}{\pi^m} = \frac{1}{\pi^m}\exp(-|z|^2) d\mathrm{Re}(z) d\mathrm{Im}(z)`.

        If we partition :math:`A_1, b_1` and :math:`A_2, b_2` into :math:`\mathrm{idx1}, \mathrm{idx2}` blocks and the remaining blocks:

        .. math::
            A_1 = \begin{pmatrix} A & C^T \\ C & B \end{pmatrix},\quad
            b_1 = \begin{pmatrix} g \\ h \end{pmatrix},\qquad
            A_2 = \begin{pmatrix} D & F^T \\ F & E \end{pmatrix},\quad
            b_2 = \begin{pmatrix} i \\ j \end{pmatrix},

        the result is given by:

        .. math::
            A_{\mathrm{out}} = \begin{pmatrix}
                B - C\,D\,L\,C^T & -F\,L\,C^T \\
                -F\,L\,C^T & E - F\,L\,A\,F^T
            \end{pmatrix}, \\
            b_{\mathrm{out}} = \begin{pmatrix}
                h - C\,(D\,L^T g + L\, i) \\
                j - F\,(A\,L\, i + L^T g)
            \end{pmatrix}, \\
            \log c_{\mathrm{out}} = -\frac{1}{2}\big[g^T D L^T g + 2 g^T L i + i^T A L i\big] + \frac{1}{2} \log(\det(-L)),

        where :math:`L = (A D - I)^{-1}`.

        This function supports broadcasted/batched inputs, however note that this function is about one order of magnitude faster when
        the inputs are not batched versus being batched with size 1, so it's recommended to squeeze the batch dimension before calling this function.

        Arguments:
            A1: The first A matrix.
            b1: The first b vector.
            A2: The second A matrix.
            b2: The second b vector.
            idx1: The indices in the first tuple to integrate over. 
                The first half is the z variables and the second half is the z* variables.
            idx2: The indices in the second tuple to integrate over. 
                The first half is the z variables and the second half is the z* variables.
            A_out: The (optional) output A matrix.
            b_out: The (optional) output b vector.
            log_c_out: The (optional) output log_c vector.

        Returns:
            The ``(A_out, b_out, log_c_out)`` triple which parametrizes the result of the integral with eventual batch dimensions.
        """
        idx1 = np.array(idx1, dtype=np.int64)
        idx2 = np.array(idx2, dtype=np.int64)
        if A1.ndim == 2 and A2.ndim == 2 and b1.ndim == 1 and b2.ndim == 1:
            return self._apply(
                "complex_gaussian_integral_2_single",
                (A1, b1, A2, b2, idx1, idx2, A_out, b_out, log_c_out),
            )
        return self._apply(
            "complex_gaussian_integral_2_batched",
            (A1, b1, A2, b2, idx1, idx2, A_out, b_out, log_c_out),
        )

    def concat(self, values: Sequence[ArrayLike], axis: int = 0) -> Tensor:
        r"""Concatenates values along the given axis.

        Args:
            values: The values to concatenate.
            axis: The axis along which to concatenate.

        Returns:
            The concatenated values.
        """
        return self._apply("concat", (values, axis))

    @overload
    def conj[Arr: np.ndarray](self, array: Arr) -> Arr: ...
    @overload
    def conj(self, array: ArrayLike) -> Tensor: ...
    def conj(self, array: ArrayLike) -> Tensor:
        r"""The complex conjugate of array.

        Args:
            array: The array to take the complex conjugate of.

        Returns:
            The complex conjugate of the given ``array``.
        """
        return self._apply("conj", (array,))

    def cos(self, array: ArrayLike) -> Tensor:
        r"""The cosine of an array.

        Args:
            array: The array to take the cosine of.

        Returns:
            The cosine of ``array``.
        """
        return self._apply("cos", (array,))

    def cosh(self, array: ArrayLike) -> Tensor:
        r"""The hyperbolic cosine of array.

        Args:
            array: The array to take the hyperbolic cosine of.

        Returns:
            The hyperbolic cosine of ``array``.
        """
        return self._apply("cosh", (array,))

    def det(self, matrix: ArrayLike) -> ScalarValue | Tensor:
        r"""The determinant of matrix.

        Args:
            matrix: The matrix to take the determinant of

        Returns:
            The determinant of ``matrix``.
        """
        return self._apply("det", (matrix,))

    def diagonal(self, array: ArrayLike, offset: int = 0, axis1: int = 0, axis2: int = 1) -> Tensor:
        r"""Return specified diagonals of array.

        Args:
            array: The array to take the diagonal of.
            offset: The offset of the diagonal.
            axis1: The first axis to take the diagonal of.
            axis2: The second axis to take the diagonal of.

        Returns:
            The diagonal of ``array``.
        """
        return self._apply("diagonal", (array, offset, axis1, axis2))

    def diag(self, array: ArrayLike, k: int = 0) -> Tensor:
        r"""The array made by inserting the given array along the :math:`k`-th diagonal.

        Args:
            array: The array to insert.
            k: The ``k``-th diagonal to insert array into.

        Returns:
            The array with ``array`` inserted into the ``k``-th diagonal.
        """
        return self._apply("diag", (array, k))

    def diag_part(self, array: ArrayLike, k: int = 0) -> Tensor:
        r"""The array of the main diagonal of array.

        Args:
            array: The array to extract the main diagonal of.
            k: The diagonal to extract.

        Returns:
            The array of the main diagonal of ``array``.
        """
        return self.diagonal(array, offset=k, axis1=-2, axis2=-1)

    def eigvals(self, tensor: ArrayLike) -> Tensor:
        r"""The eigenvalues of a tensor.

        Args:
            tensor: The tensor to calculate the eigenvalues of.

        Returns:
            The eigenvalues of ``tensor``.
        """
        return self._apply("eigvals", (tensor,))

    def eigh(self, tensor: ArrayLike) -> Tensor:
        """The eigenvalues and eigenvectors of a matrix.

        Args:
            tensor: The tensor to calculate the eigenvalues and eigenvectors of.

        Returns:
            The eigenvalues and eigenvectors of ``tensor``.
        """
        return self._apply("eigh", (tensor,))

    def einsum(
        self,
        *operands: str | ArrayLike | list[int],
        optimize: OptimizeKind = "greedy",
        memory_limit: int | Literal["max_input"] | None = None,
        backend: BackendType | None = None,
    ) -> Tensor:
        r"""The result of the Einstein summation convention on the operands.

        Similar to ``np.einsum``, two signatures are supported:
            - Subscript style:
                The first operand is a string where the subscripts for summation are a comma separated list
                of subscript labels with explicit output indices following a `->` indicator.
            - Sublist style:
                The operands must be in the form ``tensor0, labels0, tensor1, labels1, ..., output`` where ``labels``
                are a list of integers labeling indices for the preceding tensor in the list of operands
                and output is a list of integers labeling output indices for the resulting tensor.

        Note:
            In subscript style, parentheses are supported in the output string to group indices
            that should be vectorized/flattened. For example, ``"ij,jk->h(ik)"`` will vectorize
            indices ``i`` and ``k``. Ellipsis notation (``...``) cannot be combined with parenthesized
            groups.

        Args:
            operands: The operands to perform the Einstein summation on in either subscript style or sublist style.
            optimize: Optional flag whether to optimize the contraction order.
                Allowed values are True, False, "greedy", "optimal" or "auto".
                Note the TF backend does not support False and converts it to "greedy".
                If None, ``settings.EINSUM_OPTIMIZE`` is used.
            memory_limit: The memory limit for the contraction. If ``None``, the memory limit is set to the default value.
            backend: The name of the backend to use. If ``None``, the set backend is used.

        Returns:
            The result of the Einstein summation convention.
        """
        optimize = optimize or settings.EINSUM_OPTIMIZE
        backend_ = cast(BackendType, self.backend_name if backend is None else backend)

        if isinstance(operands[0], str):
            string = operands[0]
            tensors = operands[1:]
            string, groups = strip_parentheses(string)

            result = contract(
                string, *tensors, optimize=optimize, memory_limit=memory_limit, backend=backend_
            )

            if groups:
                new_shape = compute_collapsed_shape(tuple(result.shape), groups)
                result = self.reshape(result, new_shape)
        else:
            result = contract(
                *operands, optimize=optimize, memory_limit=memory_limit, backend=backend_
            )

        return result

    def exp(self, array: ArrayLike) -> Tensor:
        r"""The exponential of array element-wise.

        Args:
            array: The array to take the exponential of.

        Returns:
            The exponential of array.
        """
        return self._apply("exp", (array,))

    def expand_dims(self, array: ArrayLike, axis: int) -> Tensor:
        r"""The array with an additional dimension inserted at the given axis.

        Args:
            array: The array to expand.
            axis: The axis to insert the new dimension.

        Returns:
            The array with an additional dimension inserted at the given axis.
        """
        return self._apply("expand_dims", (array, axis))

    def squeeze(self, array: ArrayLike, axis: ArrayLike | None = None) -> Tensor:
        r"""Remove axes of length one from the array.

        Args:
            array: The array to squeeze.
            axis: The axis or axes to squeeze. If ``None``, all axes of length
                one are removed. If an axis is specified, it will only be removed
                if its length is one, otherwise an error is raised.

        Returns:
            The squeezed array with the specified axes removed.
        """
        return self._apply("squeeze", (array, axis))

    def expm(self, matrix: ArrayLike) -> Tensor:
        r"""The matrix exponential of matrix.

        Args:
            matrix: The matrix to take the exponential of.

        Returns:
            The exponential of ``matrix``.
        """
        return self._apply("expm", (matrix,))

    @overload
    def eye(self, size: int, dtype: type[np.floating]) -> RealTensor: ...
    @overload
    def eye(self, size: int, dtype: type[np.complexfloating]) -> ComplexTensor: ...
    @overload
    def eye(self, size: int, dtype: type[np.signedinteger]) -> IntTensor: ...
    @overload
    def eye(self, size: int, dtype: DTypeLike | None = None) -> Tensor: ...
    def eye(self, size: int, dtype: DTypeLike | None = None) -> Tensor:
        r"""The identity matrix of size.

        Args:
            size: The size of the identity matrix
            dtype: The data type of the identity matrix. If ``None``,
                the returned matrix is of type ``float``.

        Returns:
            The identity matrix.
        """
        return self._apply("eye", (size, dtype))

    @overload
    def eye_like[Arr: np.ndarray](self, array: Arr) -> Arr: ...
    @overload
    def eye_like(self, array: ArrayLike) -> Tensor: ...
    def eye_like(self, array: ArrayLike) -> Tensor:
        r"""The identity matrix of the same shape and dtype as array.

        Args:
            array: The array to create the identity matrix of.

        Returns:
            The identity matrix.
        """
        return self._apply("eye_like", (array,))

    def equal(self, a: ArrayLike, b: ArrayLike) -> BoolScalar:
        r"""Returns the element-wise equality of two arrays.

        Args:
            a: The first array.
            b: The second array.

        Returns:
            The element-wise equality of two arrays.
        """
        return self._apply("equal", (a, b))

    @overload
    def gather[Arr: np.ndarray](
        self, array: Arr, indices: IntArrayLike, axis: int | None = None
    ) -> Arr: ...
    @overload
    def gather(
        self, array: ArrayLike, indices: IntArrayLike, axis: int | None = None
    ) -> Tensor: ...
    def gather(self, array: ArrayLike, indices: IntArrayLike, axis: int | None = None) -> Tensor:
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

    def hermite_renormalized(
        self,
        A: ArrayLike,
        b: ArrayLike,
        c: ArrayLike,
        shape: tuple[int, ...],
        stable: bool | None = None,
        out: ComplexTensor | None = None,
    ) -> ComplexTensor:
        r"""Renormalized multidimensional Hermite polynomial given by the "exponential" Taylor
        series of :math:`exp(c + bx + 1/2*Ax^2)` at zero, where the series has :math:`sqrt(n!)`
        at the denominator rather than :math:`n!`. It computes all the amplitudes within the
        tensor of given shape.

        This method automatically selects the appropriate calculation method based on input dimensions:
        1. If A.ndim = 2, b.ndim = 1, c is scalar: Uses vanilla strategy (unbatched)
        2. If A.ndim = 2, b.ndim > 1, c is scalar: Uses vanilla_full_batch strategy with broadcasting
        3. If A.ndim > 2, b.ndim > 1, c.ndim > 0: Uses vanilla_full_batch strategy (fully batched)

        Args:
            A: The A matrix. Can be unbatched (shape D×D) or batched (shape B×D×D).
            b: The b vector. Can be unbatched (shape D) or batched (shape B×D).
            c: The c scalar. Can be scalar or batched (shape B).
            shape: The shape of the final tensor (excluding batch dimensions).
            stable: Whether to use the numerically stable version of the algorithm (also slower).
                If ``None``, uses ``settings.STABLE_FOCK_CONVERSION``. Explicit ``True``/``False``
                takes precedence over the setting.
            out: If provided, the result will be stored in this tensor.

        Returns:
            The renormalized Hermite polynomial of given shape preserving the batch dimensions.
        """

        def check_out_shape(batch_shape):
            if out is not None and any(
                d_out < d for d_out, d in zip(out.shape, batch_shape + shape)
            ):
                raise ValueError(
                    f"batch+shape {batch_shape + shape} is too large for out.shape={out.shape}",
                )

        A = self.astensor(A, dtype=self.complex128)
        b = self.astensor(b, dtype=self.complex128)
        c = self.astensor(c, dtype=self.complex128)

        stable = settings.STABLE_FOCK_CONVERSION if stable is None else stable

        if A.ndim > 2 and b.ndim > 1 and c.ndim > 0:
            batch_shape = A.shape[:-2]
            check_out_shape(batch_shape)
            if b.shape[:-1] != batch_shape:
                raise ValueError(f"b.shape={b.shape} must match batch_shape={batch_shape}")
            if c.shape[: len(batch_shape)] != batch_shape:
                raise ValueError(f"c.shape={c.shape} must match batch_shape={batch_shape}")
            D = int(np.prod(batch_shape))
            A = self.reshape(A, (D, *A.shape[-2:]))
            b = self.reshape(b, (D, *b.shape[-1:]))
            c = self.reshape(c, (D,))
            result = self._apply(
                "hermite_renormalized_batched",
                (A, b, c),
                {
                    "shape": tuple(shape),
                    "stable": stable,
                    "out": self.reshape(out, (D, *shape)) if out is not None else None,
                },
            )
            return self.reshape(result, batch_shape + tuple(shape))
        if A.ndim == 2 and b.ndim > 1:  # b-batched case
            batch_shape = b.shape[:-1]
            check_out_shape(batch_shape)
            D = int(np.prod(batch_shape))
            b = self.reshape(b, (D, *b.shape[-1:]))
            A_broadcast = self.broadcast_to(A, (D, *A.shape))
            c_broadcast = self.broadcast_to(c, (D,))
            result = self._apply(
                "hermite_renormalized_batched",
                (A_broadcast, b, c_broadcast),
                {
                    "shape": tuple(shape),
                    "stable": stable,
                    "out": self.reshape(out, (D, *shape)) if out is not None else None,
                },
            )
            return self.reshape(result, batch_shape + tuple(shape))
        # Unbatched case
        check_out_shape(())
        return self._apply(
            "hermite_renormalized",
            (A, b, c),
            {"shape": tuple(shape), "stable": stable, "out": out},
        )

    def hermite_renormalized_diagonal(
        self,
        A: ArrayLike,
        b: ArrayLike,
        c: ArrayLike,
        cutoffs: tuple[int, ...],
        reorderedAB: bool = True,
    ) -> ComplexTensor:
        r"""Renormalized multidimensional Hermite polynomial given by the "exponential" Taylor
        series of :math:`exp(C + Bx - Ax^2)` at zero, where the series has :math:`sqrt(n!)` at the
        denominator rather than :math:`n!`. Note the minus sign in front of ``A``.

        Calculates the diagonal of the Fock representation (i.e. the PNR detection probabilities of all modes)
        by applying the recursion relation in a selective manner.

        Note: This function supports batching of different B's.

        Args:
            A: The A matrix.
            b: The b vector.
            c: The c scalar.
            cutoffs: upper boundary of photon numbers in each mode
            reorderedAB: Whether to reorder A and B parameters match conventions in mrmustard.math.numba.compactFock~.

        Returns:
            The renormalized Hermite polynomial.
        """
        return self._apply("hermite_renormalized_diagonal", (A, b, c, cutoffs, reorderedAB))

    def hermite_renormalized_1leftoverMode(
        self,
        A: ArrayLike,
        b: ArrayLike,
        c: ArrayLike,
        output_cutoff: int,
        pnr_cutoffs: tuple[int, ...],
        reorderedAB: bool = True,
    ) -> ComplexTensor:
        r"""Compute the conditional density matrix of mode 0, with all the other modes
        detected with PNR detectors up to the given photon numbers.

        Args:
            A: The A matrix.
            b: The b vector.
            c: The c scalar.
            output_cutoff: Upper boundary of photon numbers in mode 0.
            pnr_cutoffs: Upper boundary of photon numbers in the other modes.
            reorderedAB: Whether to reorder A and B parameters match conventions in mrmustard.math.numba.compactFock~.

        Returns:
            The conditional density matrix of mode 0. The final shape is
            ``(output_cutoff + 1, output_cutoff + 1, *pnr_cutoffs + 1)``.
        """
        return self._apply(
            "hermite_renormalized_1leftoverMode",
            (A, b, c, output_cutoff, pnr_cutoffs, reorderedAB),
        )

    def hermite_renormalized_binomial(
        self,
        A: ArrayLike,
        B: ArrayLike,
        C: ArrayLike,
        shape: tuple[int, ...],
        max_l2: float | None,
        global_cutoff: int | None,
    ) -> ComplexTensor:
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
            max_l2 : The maximum squared L2 norm of the tensor.
            global_cutoff: The global cutoff.

        Returns:
            The renormalized Hermite polynomial of given shape.
        """
        return self._apply("hermite_renormalized_binomial", (A, B, C, shape, max_l2, global_cutoff))

    def imag(self, array: ArrayLike) -> RealTensor:
        r"""The imaginary part of array.

        Args:
            array: The scalar or array-like input to take the imaginary part of.

        Returns:
            The imaginary part of array
        """
        return self._apply("imag", (array,))

    def inv(self, tensor: ArrayLike) -> Tensor:
        r"""The inverse of tensor.

        Args:
            tensor: The tensor to take the inverse of

        Returns:
            The inverse of tensor
        """
        return self._apply("inv", (tensor,))

    def iscomplexobj(self, x: Any) -> bool:
        r"""Whether the given object is complex.

        Args:
            x: The object to check.

        Returns:
            Whether the given array is a complex object.
        """
        return self._apply("iscomplexobj", (x,))

    def isnan(self, array: ArrayLike) -> BoolScalar:
        r"""Whether the given array contains any NaN values.

        Args:
            array: The array to check for NaN values.

        Returns:
            Whether the given array contains any NaN values.
        """
        return self._apply("isnan", (array,))

    def issubdtype(self, arg1: DTypeLike, arg2: DTypeLike) -> bool:
        r"""Whether the ``arg1`` is a typecode lower/equal in type hierarchy to ``arg2``.

        Args:
            arg1: The object to be tested
            arg2: The object to be compared against

        Returns:
            Whether arg1 is a subdtype of arg2.
        """
        return self._apply("issubdtype", (arg1, arg2))

    def lgamma(self, x: ArrayLike) -> Tensor:
        r"""The natural logarithm of the gamma function of ``x``.

        Args:
            x: The array to take the natural logarithm of the gamma function of.

        Returns:
            The natural logarithm of the gamma function of ``x``.
        """
        return self._apply("lgamma", (x,))

    def log(self, x: ArrayLike) -> Tensor:
        r"""The natural logarithm of ``x``.

        Args:
            x: The array to take the natural logarithm of

        Returns:
            The natural logarithm of ``x``
        """
        return self._apply("log", (x,))

    def make_complex(self, real: ArrayLike, imag: ArrayLike) -> ComplexTensor:
        """Given two real tensors representing the real and imaginary part of a complex number,
        this operation returns a complex tensor. The input tensors must have the same shape.

        Args:
            real: The real part of the complex number.
            imag: The imaginary part of the complex number.

        Returns:
            The complex array ``real + 1j * imag``.
        """
        return self._apply("make_complex", (real, imag))

    def matmul(self, *matrices: ArrayLike) -> Tensor:
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

    def max(self, array: ArrayLike) -> ScalarValue:
        r"""The maximum value of an array.

        Args:
            array: The array to take the maximum value of.

        Returns:
            The maximum value of the array.
        """
        return self._apply("max", (array,))

    def maximum(self, a: ArrayLike, b: ArrayLike) -> Tensor:
        r"""The element-wise maximum of ``a`` and ``b``.

        Args:
            a: The first array to take the maximum of.
            b: The second array to take the maximum of.

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

    def minimum(self, a: ArrayLike, b: ArrayLike) -> Tensor:
        r"""The element-wise minimum of ``a`` and ``b``.

        Args:
            a: The first array to take the minimum of.
            b: The second array to take the minimum of.

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

    def mod(self, a: ArrayLike, b: ArrayLike) -> Tensor:
        r"""Returns the element-wise remainder of division.

        Args:
            a: The dividend array.
            b: The divisor array.

        Returns:
            The element-wise remainder of division.
        """
        return self._apply("mod", (a, b))

    @overload
    def moveaxis[Arr: np.ndarray](
        self, array: Arr, old: int | Sequence[int], new: int | Sequence[int]
    ) -> Arr: ...
    @overload
    def moveaxis(
        self, array: ArrayLike, old: int | Sequence[int], new: int | Sequence[int]
    ) -> Tensor: ...
    def moveaxis(
        self, array: ArrayLike, old: int | Sequence[int], new: int | Sequence[int]
    ) -> Tensor:
        r"""Moves the axes of an array to a new position.

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

    def mean(
        self, array: ArrayLike, axis: int | Sequence[int] | None = None
    ) -> ScalarValue | Tensor:
        r"""The mean of array along an axis.

        Args:
            array: The array to take the mean of
            axis: The axis/axes to compute the mean over. If ``None``, the mean is computed over all axes.

        Returns:
            The mean of array
        """
        return self._apply("mean", (array, axis))

    def norm(
        self, array: ArrayLike, axis: int | Sequence[int] | None = None, keepdims: bool = False
    ) -> Tensor:
        r"""The norm of array.

        Args:
            array: The array to take the norm of
            axis: The axis or axes to norm over. If ``None``, the norm is computed over all axes.
            keepdims: Whether to keep the dimensions of the array.

        Returns:
            The norm of ``array``.
        """
        return self._apply("norm", (array, axis, keepdims))

    @overload
    def ones(self, shape: Sequence[int], dtype: type[np.floating] | None) -> RealTensor: ...
    @overload
    def ones(self, shape: Sequence[int], dtype: type[np.complexfloating]) -> ComplexTensor: ...
    @overload
    def ones(self, shape: Sequence[int], dtype: type[np.signedinteger]) -> IntTensor: ...
    @overload
    def ones(self, shape: Sequence[int], dtype: DTypeLike | None = None) -> Tensor: ...
    def ones(self, shape: Sequence[int], dtype: DTypeLike | None = None) -> Tensor:
        r"""Returns an array of ones with the given ``shape`` and ``dtype``.

        Args:
            shape: The shape of the array
            dtype: The dtype of the array.

        Returns:
            The array of ones
        """
        return self._apply("ones", (shape, dtype))

    def full(
        self, shape: Sequence[int], fill_value: Scalar, dtype: DTypeLike | None = None
    ) -> Tensor:
        r"""Returns an array of given shape filled with ``fill_value``.

        Args:
            shape: The shape of the array.
            fill_value: The value to fill the array with.
            dtype: The dtype of the array. If ``None``, the returned array is
                of type inferred from ``fill_value``.

        Returns:
            The array filled with ``fill_value``.
        """
        return self._apply("full", (shape, fill_value, dtype))

    @overload
    def ones_like[Arr: np.ndarray](self, array: Arr) -> Arr: ...
    @overload
    def ones_like(self, array: ArrayLike) -> Tensor: ...
    def ones_like(self, array: ArrayLike) -> Tensor:
        r"""Returns an array of ones with the same shape and ``dtype`` as ``array``.

        Args:
            array: The array to take the shape and dtype of

        Returns:
            The array of ones
        """
        return self._apply("ones_like", (array,))

    def outer(self, array1: ArrayLike, array2: ArrayLike) -> Tensor:
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
        array: ArrayLike,
        paddings: Sequence[tuple[int, int]],
        mode="CONSTANT",
        constant_values=0,
    ) -> Tensor:
        r"""The padded array.

        Args:
            array: The array to pad.
            paddings: Paddings to apply.
            mode: Mode to apply the padding.
            constant_values: Constant values to use for padding.

        Returns:
            The padded array
        """
        return self._apply("pad", (array, tuple(paddings), mode, constant_values))

    def pinv(self, matrix: ArrayLike) -> Tensor:
        r"""The pseudo-inverse of matrix.

        Args:
            matrix: The matrix to take the pseudo-inverse of.

        Returns:
            The pseudo-inverse of matrix
        """
        return self._apply("pinv", (matrix,))

    def pow(self, x: ArrayLike, y: ArrayLike) -> Tensor:
        r"""Returns :math:`x^y`. Broadcasts ``x`` and ``y`` if necessary.

        Args:
            x: The base.
            y: The exponent.

        Returns:
            The :math:`x^y`.
        """
        return self._apply("pow", (x, y))

    def kron(self, tensor1: ArrayLike, tensor2: ArrayLike) -> Tensor:
        r"""The Kroenecker product of the given tensors.

        Args:
            tensor1: A tensor.
            tensor2: Another tensor.

        Returns:
            The Kroenecker product.
        """
        return self._apply("kron", (tensor1, tensor2))

    def prod(
        self, array: ArrayLike, axis: int | tuple[int, ...] | None = None
    ) -> ScalarValue | Tensor:
        r"""The product of all elements in ``array``.

        Args:
            array: The array of elements to calculate the product of.
            axis: The axis along which a product is performed. If ``None``, it calculates
                the product of all elements in ``array``.

        Returns:
            The product of the elements in ``array``.
        """
        return self._apply("prod", (array, axis))

    def real(self, array: ArrayLike) -> RealTensor:
        r"""The real part of ``array``.

        Args:
            array: The scalar or array-like input to take the real part of.

        Returns:
            The real part of ``array``
        """
        return self._apply("real", (array,))

    @overload
    def reshape[Arr: np.ndarray](self, array: Arr, shape: int | Sequence[int]) -> Arr: ...
    @overload
    def reshape(self, array: ArrayLike, shape: int | Sequence[int]) -> Tensor: ...
    def reshape(self, array: ArrayLike, shape: int | Sequence[int]) -> Tensor:
        r"""The reshaped array.

        Args:
            array: The array to reshape.
            shape: Shape to reshape the array to.

        Returns:
            The reshaped array.
        """
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        return self._apply("reshape", (array, shape))

    def shape(self, array: ArrayLike) -> tuple[int, ...]:
        r"""The shape of an array.

        Args:
            array: The array to take the shape of.

        Returns:
            The shape of the array.
        """
        return self._apply("shape", (array,))

    def sin(self, array: ArrayLike) -> Tensor:
        r"""The sine of ``array``.

        Args:
            array: The array to take the sine of.

        Returns:
            The sine of ``array``.
        """
        return self._apply("sin", (array,))

    def sinh(self, array: ArrayLike) -> Tensor:
        r"""The hyperbolic sine of ``array``.

        Args:
            array: The array to take the hyperbolic sine of.

        Returns:
            The hyperbolic sine of ``array``.
        """
        return self._apply("sinh", (array,))

    def solve(self, matrix: ArrayLike, rhs: ArrayLike) -> Tensor:
        r"""The solution of the linear system :math:`Ax = b`.

        Args:
            matrix: The matrix :math:`A`.
            rhs: The vector :math:`b`.

        Returns:
            The solution :math:`x`.
        """
        return self._apply("solve", (matrix, rhs))

    @overload
    def sort[Arr: np.ndarray](self, array: Arr, axis: int = -1) -> Arr: ...
    @overload
    def sort(self, array: ArrayLike, axis: int = -1) -> Tensor: ...
    def sort(self, array: ArrayLike, axis: int = -1) -> Tensor:
        r"""Sort the array along an axis.

        Args:
            array: The array to sort.
            axis: The axis to sort along.

        Returns:
            A sorted version of the array in ascending order.
        """
        return self._apply("sort", (array, axis))

    def sqrt(self, x: ArrayLike, dtype: DTypeLike | None = None) -> Tensor:
        r"""The square root of ``x``.

        Args:
            x: The scalar or array-like input to take the square root of.
            dtype: ``dtype`` of the output array.

        Returns:
            The square root of ``x``.
        """
        return self._apply("sqrt", (x, dtype))

    def sqrtm(self, tensor: ArrayLike, dtype: DTypeLike | None = None) -> Tensor:
        r"""The matrix square root.

        Args:
            tensor: The tensor to take the matrix square root of.
            dtype: The ``dtype`` of the output tensor. If ``None``, the output
                is of type ``math.complex128``.

        Returns:
            The square root of ``x``.
        """
        return self._apply("sqrtm", (tensor, dtype))

    def stack(self, arrays: Sequence[ArrayLike], axis: int = 0) -> Tensor:
        r"""Stack arrays in sequence along a new axis.

        Args:
            arrays: Sequence of tensors to stack.
            axis: The axis along which to stack the arrays.

        Returns:
            The stacked array.
        """
        return self._apply("stack", (arrays, axis))

    def sum(self, array: ArrayLike, axis: int | Sequence[int] | None = None):
        r"""The sum of array.

        Args:
            array: The array to take the sum of.
            axis: The axis/axes to sum over.

        Returns:
            The sum of array.
        """
        if axis is not None and not isinstance(axis, int):
            neg = [a for a in axis if a < 0]
            pos = [a for a in axis if a >= 0]
            axis = tuple(sorted(neg) + sorted(pos)[::-1])
        return self._apply("sum", (array, axis))

    @overload
    def swapaxes[Arr: np.ndarray](self, array: Arr, axis1: int, axis2: int) -> Arr: ...
    @overload
    def swapaxes(self, array: ArrayLike, axis1: int, axis2: int) -> Tensor: ...
    def swapaxes(self, array: ArrayLike, axis1: int, axis2: int) -> Tensor:
        r"""Swap two axes of an array.

        Args:
            array: The array to swap axes of.
            axis1: The first axis to swap.
            axis2: The second axis to swap.

        Returns:
            The array with the axes swapped.
        """
        return self._apply("swapaxes", (array, axis1, axis2))

    def tensordot(self, a: ArrayLike, b: ArrayLike, axes: Sequence[int]) -> Tensor:
        r"""The tensordot product of ``a`` and ``b``.

        Args:
            a: The first array to take the tensordot product of.
            b: The second array to take the tensordot product of.
            axes: The axes to take the tensordot product over.

        Returns:
            The tensordot product of ``a`` and ``b``.
        """
        return self._apply("tensordot", (a, b, tuple(axes)))

    def tile(self, array: ArrayLike, repeats: Sequence[int]) -> Tensor:
        r"""The tiled array.

        Args:
            array: The array to tile.
            repeats: Number of times to tile the array along each axis.

        Returns:
            The tiled array.
        """
        return self._apply("tile", (array, tuple(repeats)))

    def trace(self, array: ArrayLike, dtype: DTypeLike | None = None) -> Tensor:
        r"""The trace of array.

        Args:
            array: The array to take the trace of.
            dtype: ``dtype`` of the output array.

        Returns:
            The trace of array.
        """
        return self._apply("trace", (array, dtype))

    @overload
    def transpose[Arr: np.ndarray](self, a: Arr, perm: Sequence[int] | None = None) -> Arr: ...
    @overload
    def transpose(self, a: ArrayLike, perm: Sequence[int] | None = None) -> Tensor: ...
    def transpose(self, a: ArrayLike, perm: Sequence[int] | None = None) -> Tensor:
        r"""The transposed arrays.

        Args:
            a: The array to transpose.
            perm: Permutation to apply to the array.

        Returns:
            The transposed array.
        """
        perm = tuple(perm) if perm is not None else None
        return self._apply("transpose", (a, perm))

    def tan(self, array: ArrayLike) -> Tensor:
        r"""The tangent of ``array``.

        Args:
            array: The array to take the tangent of

        Returns:
            The tangent of ``array``
        """
        return self._apply("tan", (array,))

    def tanh(self, array: ArrayLike) -> Tensor:
        r"""The hyperbolic tangent of ``array``.

        Args:
            array: The array to take the hyperbolic tangent of

        Returns:
            The hyperbolic tangent of ``array``
        """
        return self._apply("tanh", (array,))

    @overload
    def update_add_tensor[Arr: np.ndarray](
        self, tensor: Arr, indices: ArrayLike, values: ArrayLike
    ) -> Arr: ...
    @overload
    def update_add_tensor(
        self, tensor: ArrayLike, indices: ArrayLike, values: ArrayLike
    ) -> Tensor: ...
    def update_add_tensor(self, tensor: ArrayLike, indices: ArrayLike, values: ArrayLike) -> Tensor:
        r"""Updates a tensor in place by adding the given values.

        Args:
            tensor: The tensor to update.
            indices: The indices to update.
            values: The values to add.

        Returns:
            The updated tensor.
        """
        return self._apply("update_add_tensor", (tensor, indices, values))

    def value_and_gradients(
        self,
        cost_fn: Callable[..., Scalar],
        parameters: dict[str, list[Trainable]],
    ) -> tuple[Tensor, dict[str, list[Tensor]]]:
        r"""The loss and gradients of the given cost function.

        Args:
            cost_fn: Cost function to compute the loss and gradients of.
            parameters: Parameters to compute the loss and gradients of.

        Returns:
            The loss and gradients of the given cost function.
        """
        return self._apply("value_and_gradients", (cost_fn, parameters))

    def xlogy(self, x: ArrayLike, y: ArrayLike) -> Tensor:
        """Returns ``0`` if ``x == 0`` elementwise and ``x * log(y)`` otherwise.

        Args:
            x: The first array.
            y: The second array.

        Returns:
            The result of the xlogy operation.
        """
        return self._apply("xlogy", (x, y))

    @overload
    def zeros(self, shape: int | Sequence[int], dtype: type[np.floating] | None) -> RealTensor: ...
    @overload
    def zeros(
        self, shape: int | Sequence[int], dtype: type[np.complexfloating]
    ) -> ComplexTensor: ...
    @overload
    def zeros(self, shape: int | Sequence[int], dtype: type[np.signedinteger]) -> IntTensor: ...
    @overload
    def zeros(self, shape: int | Sequence[int], dtype: DTypeLike | None = None) -> Tensor: ...
    def zeros(self, shape: int | Sequence[int], dtype: DTypeLike | None = None) -> Tensor:
        r"""Returns an array of zeros with the given shape and ``dtype``.

        Args:
            shape: The shape of the array.
            dtype: The dtype of the array.

        Returns:
            The array of zeros.
        """
        return self._apply("zeros", (shape, dtype))

    def conditional(
        self, cond: ArrayLike, true_fn: Callable, false_fn: Callable, *args: Any
    ) -> Any:
        r"""Executes ``true_fn`` if ``cond`` is ``True``, otherwise ``false_fn``.

        Args:
            cond: The condition to check.
            true_fn: The function to execute if ``cond`` is ``True``.
            false_fn: The function to execute if ``cond`` is ``False``.
            *args: The arguments to pass to ``true_fn`` and ``false_fn``.

        Returns:
            The result of ``true_fn`` if ``cond`` is ``True``, otherwise ``false_fn``.
        """
        return self._apply("conditional", (cond, true_fn, false_fn, *args))

    def error_if(self, array: ArrayLike, condition: BoolScalar, msg: str) -> None:
        r"""Raises an error if ``condition`` is ``True``.

        Args:
            array: The array to check.
            condition: The condition to check; should only use array elements in the condition.
            msg: The message to raise if ``condition`` is ``True``.

        Raises:
            ValueError: If at least one element of ``condition`` is ``True``.
        """
        return self._apply("error_if", (array, condition, msg))

    def infinity_like(self, array: ArrayLike) -> Tensor:
        r"""Returns an array of infinities with the same shape as ``array``.

        Args:
            array: The array to take the shape of.

        Returns:
            An array of infinities with the same shape as ``array``.
        """
        return self._apply("infinity_like", (array,))

    @overload
    def zeros_like[Arr: np.ndarray](self, array: Arr) -> Arr: ...
    @overload
    def zeros_like(self, array: ArrayLike) -> Tensor: ...
    def zeros_like(self, array: ArrayLike) -> Tensor:
        r"""Returns an array of zeros with the same shape and ``dtype`` as ``array``.

        Args:
            array: The array to take the shape and ``dtype`` of.

        Returns:
            The array of zeros.
        """
        return self._apply("zeros_like", (array,))

    def map_fn(self, fn: Callable, elements: ArrayLike) -> Tensor:
        """Transforms elems by applying fn to each element unstacked on axis 0.

        Args:
            fn: The callable to be performed. It accepts one argument,
                which will have the same (possibly nested) structure as elems.
            elements: A tensor or (possibly nested) sequence of tensors,
                each of which will be unstacked along their first dimension.
                ``func`` will be applied to the nested sequence of the resulting slices.

        Returns:
            The result of applying ``fn`` on ``elements``.
        """
        return self._apply("map_fn", (fn, elements))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Fock lattice strategies
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def displacement(self, alpha: ComplexScalar, shape: tuple[int, int]) -> ComplexTensor:
        r"""Creates a single mode displacement matrix using a Fock lattice strategy.

        Args:
            alpha: The displacement.
            shape: The shape of the displacement matrix.

        Returns:
            The matrix representing the displacement gate.
        """
        return self._apply("displacement", (alpha, shape))

    def beamsplitter(
        self,
        theta: RealScalar,
        phi: RealScalar,
        shape: tuple[int, int, int, int],
        method: Literal["vanilla", "schwinger", "stable"],
    ) -> ComplexTensor:
        r"""Creates a beamsplitter matrix with given cutoffs using a Fock lattice strategy.

        Args:
            theta: Transmittivity angle of the beamsplitter.
            phi: Phase angle of the beamsplitter.
            shape: Output shape of the two modes.
            method: Method to compute the beamsplitter ("vanilla", "schwinger" or "stable").

        Returns:
            The matrix representing the beamsplitter gate.

        Raises:
            ValueError: If the method is not "vanilla", "schwinger" or "stable".
        """
        return self._apply("beamsplitter", (theta, phi), {"shape": shape, "method": method})

    def homodyne_projector(
        self,
        fock_dim: int,
        A: ComplexMatrix,
        b: ComplexVector,
        c: ComplexScalar,
        out: Tensor | None = None,
    ) -> ComplexTensor:
        r"""Creates a homodyne projector matrix.

        Args:
            fock_dim: The Fock dimension.
            A: The A matrix.
            b: The b vector.
            c: The c scalar.
            out: The output tensor.

        Returns:
            The homodyne projector matrix.
        """
        return self._apply("homodyne_projector", (fock_dim, A, b, c, out))

    def squeezed(self, r: RealScalar, phi: RealScalar, shape: tuple[int]) -> ComplexTensor:
        r"""Creates a single mode squeezed state matrix using a Fock lattice strategy.

        Args:
            r: Squeezing magnitude.
            phi: Squeezing angle.
            shape: Output shape of the mode.

        Returns:
            The matrix representing the squeezed state.
        """
        return self._apply("squeezed", (r, phi, shape))

    def squeezer(
        self, r: RealScalar, phi: RealScalar, shape: tuple[int, int]
    ) -> ComplexTensor:  # pragma: no cover
        r"""Creates a single mode squeezer matrix using a Fock lattice strategy.

        Args:
            r: Squeezing magnitude.
            phi: Squeezing angle.
            shape: Output shape of the two modes.

        Returns:
            The matrix representing the squeezer.
        """
        return self._apply("squeezer", (r, phi, shape))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Methods that build on the basic ops and don't need to be overridden in the backend implementation
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @overload
    def dagger[Arr: np.ndarray](self, array: Arr) -> Arr: ...
    @overload
    def dagger(self, array: ArrayLike) -> Tensor: ...
    def dagger(self, array: ArrayLike) -> Tensor:
        """The adjoint of ``array``. This operation swaps the first
        and second half of the indexes and then conjugates the matrix.

        Args:
            array: The array to take the adjoint of

        Returns:
            The adjoint of ``array``
        """
        N = len(self.shape(array)) // 2
        perm = list(range(N, 2 * N)) + list(range(N))
        return self.conj(self.transpose(array, perm=perm))

    def unitary_to_orthogonal(self, U: ArrayLike) -> Tensor:
        r"""Maps a unitary matrix (or batch of unitary matrices) into an orthogonal matrix (or batch)
        of twice the size.

        Args:
            U: A unitary matrix of shape (..., N, N) where ... represents optional batch dimensions.

        Returns:
            An orthogonal matrix of shape (..., 2N, 2N).
        """
        X = self.real(U)
        Y = self.imag(U)
        return self.block([[X, -Y], [Y, X]])

    def random_symplectic(
        self,
        num_modes: int,
        max_r: float = 1.0,
        seed: int | None = None,
        batch_shape: tuple[int, ...] = (),
    ) -> Tensor:
        r"""A random symplectic matrix in ``Sp(2*num_modes)``.

        Squeezing is sampled uniformly from 0.0 to ``max_r`` (1.0 by default).

        Args:
            num_modes: The number of modes.
            max_r: The maximum squeezing value.
            seed: The random seed. If ``None``, the global seed is used.
            batch_shape: The batch shape for generating multiple random matrices.
        """
        rng = settings.get_rng(seed)
        total_size = int(np.prod(batch_shape))

        W = unitary_group.rvs(dim=num_modes, size=total_size, random_state=rng)
        V = unitary_group.rvs(dim=num_modes, size=total_size, random_state=rng)
        r = rng.uniform(low=0.0, high=max_r, size=(*batch_shape, num_modes))
        OW = self.unitary_to_orthogonal(W).reshape(*batch_shape, 2 * num_modes, 2 * num_modes)
        OV = self.unitary_to_orthogonal(V).reshape(*batch_shape, 2 * num_modes, 2 * num_modes)
        diag_elements = self.concat([self.exp(-r), self.exp(r)], axis=-1)
        return self.einsum("...ij,...j,...jl->...il", OW, diag_elements, OV)

    def random_orthogonal(
        self, N: int, seed: int | None = None, batch_shape: tuple[int, ...] = ()
    ) -> Tensor:
        r"""A random orthogonal matrix in :math:`O(N)`.

        Args:
            N: The dimension of the matrix.
            seed: The random seed. If ``None``, the global seed is used.
            batch_shape: The batch shape for generating multiple random matrices.
        """
        rng = settings.get_rng(seed)
        matrices = ortho_group.rvs(dim=N, size=int(np.prod(batch_shape)), random_state=rng)
        return matrices.reshape(*batch_shape, N, N)

    def random_unitary(
        self, N: int, seed: int | None = None, batch_shape: tuple[int, ...] = ()
    ) -> Tensor:
        r"""A random unitary matrix in :math:`U(N)`.

        Args:
            N: The dimension of the matrix.
            seed: The random seed. If ``None``, the global seed is used.
            batch_shape: The batch shape for generating multiple random matrices.
        """
        rng = settings.get_rng(seed)
        matrices = unitary_group.rvs(dim=N, size=int(np.prod(batch_shape)), random_state=rng)
        return matrices.reshape(*batch_shape, N, N)

    def random_siegel(
        self,
        n: int,
        max_r: float = 0.9,
        seed: int | None = None,
        batch_shape: tuple[int, ...] = (),
    ) -> Tensor:
        r"""A random complex symmetric matrix in the open Siegel disk
        :math:`\mathcal{D}_n = \{Z \in \mathbb{C}^{n\times n} : Z = Z^T,\ I - Z^* Z \succ 0\}`.

        The matrix is sampled as :math:`Z = U \mathrm{diag}(r) U^T` with ``U`` Haar-random
        unitary and :math:`r_i \sim \mathrm{Uniform}(0, \mathtt{max\_r})`. The Takagi
        (singular) values of :math:`Z` coincide with the :math:`r_i`, so
        :math:`\|Z\|_\mathrm{op} < \mathtt{max\_r} < 1` and all eigenvalues of :math:`Z`
        lie in the open unit disk.

        Args:
            n: The dimension of the matrix.
            max_r: The maximum Takagi value. Must satisfy ``0 <= max_r < 1`` to remain
                strictly inside the open Siegel disk.
            seed: The random seed. If ``None``, the global seed is used.
            batch_shape: The batch shape for generating multiple random matrices.

        Returns:
            A complex symmetric matrix of shape ``(*batch_shape, n, n)``.

        Raises:
            ValueError: If ``max_r`` is not in ``[0, 1)``.
        """
        if not 0.0 <= max_r < 1.0:
            raise ValueError(
                f"max_r must be in [0, 1) for the open Siegel disk, got {max_r}.",
            )
        rng = settings.get_rng(seed)
        U = unitary_group.rvs(dim=n, size=int(np.prod(batch_shape)), random_state=rng).reshape(
            *batch_shape, n, n
        )
        r = rng.uniform(low=0.0, high=max_r, size=(*batch_shape, n)).astype(U.dtype)
        return self.einsum("...ij,...j,...kj->...ik", U, r, U)

    @staticmethod
    @lru_cache
    def Xmat(num_modes: int) -> RealTensor:
        r"""The matrix :math:`X_n = \begin{bmatrix}0 & I_n\\ I_n & 0\end{bmatrix}.`.

        Args:
            num_modes: A positive integer representing the number of modes.

        Returns:
            The :math:`2N\times 2N` array.
        """
        I = np.identity(num_modes)
        O = np.zeros((num_modes, num_modes))
        return np.block([[O, I], [I, O]])

    @staticmethod
    @lru_cache
    def Zmat(num_modes: int) -> RealTensor:
        r"""The matrix :math:`Z_n = \begin{bmatrix}I_n & 0\\ 0 & -I_n\end{bmatrix}.`.

        Args:
            num_modes: A positive integer representing the number of modes.

        Returns:
            The :math:`2N\times 2N` array.
        """
        I = np.identity(num_modes)
        O = np.zeros((num_modes, num_modes))
        return np.block([[I, O], [O, -I]])

    @staticmethod
    @lru_cache
    def rotmat(num_modes: int) -> ComplexTensor:
        r"""Rotation matrix from quadratures to complex amplitudes.

        Args:
            num_modes: A positive integer representing the number of modes.

        Returns:
            The rotation matrix.
        """
        I = np.identity(num_modes)
        return np.sqrt(0.5) * np.block([[I, 1j * I], [I, -1j * I]])

    @staticmethod
    @lru_cache
    def J(num_modes: int) -> RealTensor:
        r"""Symplectic form.

        Args:
            num_modes: A positive integer representing the number of modes.

        Returns:
            The symplectic form.
        """
        I = np.identity(num_modes)
        O = np.zeros_like(I)
        return np.block([[O, I], [-I, O]])

    @overload
    def all_diagonals(self, rho: ArrayLike, real: Literal[True]) -> RealTensor: ...
    @overload
    def all_diagonals(self, rho: ArrayLike, real: Literal[False]) -> Tensor: ...
    def all_diagonals(self, rho: ArrayLike, real: bool) -> Tensor:
        r"""Returns all the diagonals of a density matrix.

        Args:
            rho: The density matrix.
            real: Whether to return the real part of the diagonals.

        Returns:
            The diagonals of the density matrix.
        """
        rho = self.astensor(rho)
        cutoffs = rho.shape[: rho.ndim // 2]
        rho = self.reshape(rho, (int(np.prod(cutoffs)), int(np.prod(cutoffs))))
        diag = self.diag_part(rho)
        if real:
            return self.real(self.reshape(diag, cutoffs))

        return self.reshape(diag, cutoffs)

    def euclidean_to_symplectic(self, S: ArrayLike, dS_euclidean: ArrayLike) -> Tensor:
        r"""Convert the Euclidean gradient to a Riemannian gradient on the
        tangent bundle of the symplectic manifold.

        Implemented from:
            Wang J, Sun H, Fiori S. A Riemannian‐steepest‐descent approach
            for optimization on the real symplectic group.
            Mathematical Methods in the Applied Sciences. 2018 Jul 30;41(11):4273-86.

        Args:
            S: Symplectic matrix.
            dS_euclidean: Euclidean gradient tensor.

        Returns:
            The symplectic gradient tensor.
        """
        Jmat = self.J(self.shape(S)[-1] // 2)
        Z = self.matmul(self.swapaxes(S, -1, -2), dS_euclidean)
        return 0.5 * (Z + self.matmul(self.matmul(Jmat, self.swapaxes(Z, -1, -2)), Jmat))

    def euclidean_to_unitary(self, U: ArrayLike, dU_euclidean: ArrayLike) -> Tensor:
        r"""Convert the Euclidean gradient to a Riemannian gradient on the
        tangent bundle of the unitary manifold.

        Implemented from:
            Y Yao, F Miatto, N Quesada - arXiv preprint arXiv:2209.06069, 2022.

        Args:
            U: Unitary matrix.
            dU_euclidean: Euclidean gradient tensor.

        Returns:
            The unitary gradient tensor.
        """
        Z = self.matmul(self.conj(self.swapaxes(U, -1, -2)), dU_euclidean)
        return 0.5 * (Z - self.conj(self.swapaxes(Z, -1, -2)))

    def euclidean_to_siegel(self, Z: ArrayLike, dZ_euclidean: ArrayLike) -> Tensor:
        r"""Convert the Euclidean gradient to a Riemannian gradient on the Siegel disk
        :math:`\mathcal{D}_g = \{Z \in \mathbb{C}^{g\times g} : Z = Z^T,\ I - Z^* Z
        \succ 0\}` with the Bergman metric.

        .. math::
            \mathrm{grad}\,f(Z) = \mathrm{sym}\!\left[(I - Z Z^*)\,\nabla_E f\,
            (I - Z^* Z)\right],

        where :math:`\mathrm{sym}(M) = (M + M^T)/2` projects onto the tangent space
        :math:`T_Z \mathcal{D}_g \simeq \mathrm{Sym}(g,\mathbb{C})`.

        Args:
            Z: A point in :math:`\mathcal{D}_g` (complex symmetric).
            dZ_euclidean: Euclidean gradient tensor.

        Returns:
            The Siegel (Bergman) gradient tensor.
        """
        Z = self.astensor(Z)
        Z_H = self.conj(self.swapaxes(Z, -1, -2))
        eye = self.eye(Z.shape[-1], dtype=Z.dtype)
        left = eye - self.matmul(Z, Z_H)
        right = eye - self.matmul(Z_H, Z)
        grad = self.matmul(self.matmul(left, dZ_euclidean), right)
        return 0.5 * (grad + self.swapaxes(grad, -1, -2))
