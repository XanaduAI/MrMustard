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
from typing import Any

import numpy as np
from jax.errors import TracerArrayConversionError
from scipy.stats import ortho_group, unitary_group

from ..utils.settings import settings
from ..utils.typing import Batch, Matrix, Tensor, Trainable, Vector
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

# lazy import for jax
module_name_jax = "mrmustard.math.backend_jax"
module_jax, loader_jax = lazy_import(module_name_jax)

all_modules = {
    "numpy": {"module": module_np, "loader": loader_np, "object": "BackendNumpy"},
    "tensorflow": {
        "module": module_tf,
        "loader": loader_tf,
        "object": "BackendTensorflow",
    },
    "jax": {
        "module": module_jax,
        "loader": loader_jax,
        "object": "BackendJax",
    },
}


class BackendManager:
    r"""
    A class to manage the different backends supported by Mr Mustard.
    """

    # the backend in use, which is numpy by default
    _backend = BackendNumpy()

    # the configured Euclidean optimizer.
    _euclidean_opt: type | None = None

    def __init__(self) -> None:
        # binding types and decorators of numpy backend
        self._bind()

    def _apply(
        self,
        fn: str,
        args: Sequence[Any] | None = (),
        kwargs: dict | None = None,
        backend_name: str | None = None,
    ) -> Any:
        r"""
        Applies a function ``fn`` from the backend in use to the given ``args`` and ``kwargs``.

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
        r"""
        Binds the types and decorators of this backend manager to those of the given ``self._backend``.
        """
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
        r"""
        The backend that is being used.
        """
        return self._backend

    @property
    def backend_name(self) -> str:
        r"""
        The name of the backend in use.
        """
        return self._backend.name

    @property
    def euclidean_opt(self):
        r"""The configured Euclidean optimizer."""
        if not self._euclidean_opt:
            self._euclidean_opt = self.DefaultEuclideanOptimizer()
        return self._euclidean_opt

    @property
    def BackendError(self):
        r"""
        The error class for backend specific errors.

        Note that currently this only applies to the case where
        ``auto_shape`` is jitted  via the ``jax`` backend.
        """
        return TracerArrayConversionError

    def change_backend(self, name: str) -> None:
        r"""
        Changes the backend to a different one.

        Args:
            name: The name of the new backend.
        """
        if self.backend_name != name:
            # switch backend
            self._backend = self.get_backend(name)
            # bind
            self._bind()

    def get_backend(self, name: str | None = None) -> BackendBase:
        r"""
        Returns the backend with the given name.

        Args:
            name: The name of the backend.

        Returns:
            The backend with the given name.

        Raises:
            ValueError: If the backend name is not a supported one.
        """
        if name not in ["numpy", "tensorflow", "jax"]:
            raise ValueError("Backend must be either ``numpy`` or ``tensorflow`` or ``jax``.")

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

    def abs(self, array: Tensor) -> Tensor:
        r"""The absolute value of array.

        Args:
            array: The array to take the absolute value of.

        Returns:
            The absolute value of the given ``array``.
        """
        return self._apply("abs", (array,))

    def all(self, array: Tensor) -> bool:
        r"""
        Returns ``True`` if all elements of array are ``True``, ``False`` otherwise.

        Args:
            array: The array to check.

        Returns:
            ``True`` if all elements of array are ``True``, ``False`` otherwise.
        """
        array = self.astensor(array)
        return self._apply("all", (array,))

    def allclose(self, array1: Tensor, array2: Tensor, atol=1e-9, rtol=1e-5) -> bool:
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
        array1 = self.astensor(array1)
        array2 = self.astensor(array2)
        return self._apply("allclose", (array1, array2, atol, rtol))

    def angle(self, array: Tensor) -> Tensor:
        r"""
        The complex phase of ``array``.

        Args:
            array: The array to take the complex phase of.

        Returns:
            The complex phase of ``array``.
        """
        return self._apply("angle", (array,))

    def any(self, array: Tensor) -> bool:
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
    ) -> Tensor:
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

    def atleast_nd(self, array: Tensor, n: int, dtype=None) -> Tensor:
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

    def block(self, blocks: list[list[Tensor]], axes=(-2, -1)) -> Tensor:
        r"""
        Returns a matrix made from the given blocks.

        Args:
            blocks: A list of lists of compatible blocks.
            axes: The axes to stack the blocks along.

        Returns:
            The matrix made of blocks.
        """
        rows = [self.concat(row, axis=axes[1]) for row in blocks]
        return self.concat(rows, axis=axes[0])

    def broadcast_arrays(self, *arrays: list[Tensor]) -> list[Tensor]:
        r"""
        Broadcast arrays to a common shape.

        Args:
            *arrays: The arrays to broadcast.

        Returns:
            A list of broadcasted arrays.
        """
        return self._apply("broadcast_arrays", arrays)

    def broadcast_to(self, array: Tensor, shape: tuple[int, ...], dtype=None) -> Tensor:
        r"""Broadcasts an array to a new shape.

        Args:
            array: The array to broadcast.
            shape: The shape to broadcast to.
            dtype: The dtype to broadcast to.
        Returns:
            The broadcasted array.
        """
        array = self.astensor(array, dtype=dtype)
        return self._apply("broadcast_to", (array, shape))

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

    def einsum(
        self,
        string: str,
        *tensors,
        optimize: bool | str = "greedy",
        backend: str | None = None,
    ) -> Tensor:
        r"""The result of the Einstein summation convention on the tensors.

        Args:
            string: The string of the Einstein summation convention.
            tensors: The tensors to perform the Einstein summation on.
            optimize: Optional flag whether to optimize the contraction order.
                Allowed values are True, False, "greedy", "optimal" or "auto".
                Note the TF backend does not support False and converts it to "greedy".
                If None, ``settings.EINSUM_OPTIMIZE`` is used.
            backend: The name of the backend to use. If ``None``, the set backend is used.

        Returns:
            The result of the Einstein summation convention.
        """
        optimize = optimize or settings.EINSUM_OPTIMIZE
        return self._apply(
            "einsum",
            (string, *tensors),
            {"optimize": optimize},
            backend_name=backend,
        )

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

    def gather(self, array: Tensor, indices: Batch[int], axis: int | None = None) -> Tensor:
        r"""The values of the array at the given indices.

        Args:
            array: The array to gather values from.
            indices: The indices to gather values from.
            axis: The axis to gather values from.

        Returns:
            The values of the array at the given indices.
        """
        array = self.astensor(array)
        indices = self.astensor(indices, dtype=self.int64)
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
        A: Tensor,
        b: Tensor,
        c: Tensor,
        shape: tuple[int],
        stable: bool = False,
        out: Tensor | None = None,
    ) -> Tensor:
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

        stable = stable or settings.STABLE_FOCK_CONVERSION
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
            "hermite_renormalized_unbatched",
            (A, b, c),
            {"shape": tuple(shape), "stable": stable, "out": out},
        )

    def hermite_renormalized_diagonal(
        self,
        A: Tensor,
        b: Tensor,
        c: Tensor,
        cutoffs: tuple[int],
    ) -> Tensor:
        r"""Renormalized multidimensional Hermite polynomial for calculating the diagonal of the Fock representation.

        Args:
            A: The A matrix.
            b: The b vector.
            c: The c scalar.
            cutoffs: Upper boundary of photon numbers in each mode.

        Returns:
            The diagonal elements of the Fock representation (i.e., PNR detection probabilities).
        """
        return self._apply("hermite_renormalized_diagonal", (A, b, c, cutoffs))

    def hermite_renormalized_diagonal_batch(
        self,
        A: Tensor,
        B: Tensor,
        C: Tensor,
        cutoffs: tuple[int],
    ) -> Tensor:
        r"""First, reorder A and B parameters of Bargmann representation to match conventions in mrmustard.math.compactFock~
        Then, calculates the required renormalized multidimensional Hermite polynomial.
        Same as hermite_renormalized_diagonal but works for a batch of different B's."""
        return self._apply("hermite_renormalized_diagonal_batch", (A, B, C, cutoffs))

    def hermite_renormalized_1leftoverMode(
        self,
        A: Tensor,
        b: Tensor,
        c: Tensor,
        output_cutoff: int,
        pnr_cutoffs: tuple[int, ...],
    ) -> Tensor:
        r"""Compute the conditional density matrix of mode 0, with all the other modes
        detected with PNR detectors up to the given photon numbers.

        Args:
            A: The A matrix.
            b: The b vector.
            c: The c scalar.
            output_cutoff: upper boundary of photon numbers in mode 0
            pnr_cutoffs: upper boundary of photon numbers in the other modes

        Returns:
            The conditional density matrix of mode 0. The final shape is
            ``(output_cutoff + 1, output_cutoff + 1, *pnr_cutoffs + 1)``.
        """
        return self._apply(
            "hermite_renormalized_1leftoverMode",
            (A, b, c, output_cutoff, pnr_cutoffs),
        )

    def hermite_renormalized_binomial(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        shape: tuple[int],
        max_l2: float | None,
        global_cutoff: int | None,
    ) -> np.ndarray:
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
        return self._apply("hermite_renormalized_binomial", (A, B, C, shape, max_l2, global_cutoff))

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

    def isnan(self, array: Tensor) -> Tensor:
        r"""Whether the given array contains any NaN values.

        Args:
            array: The array to check for NaN values.

        Returns:
            Whether the given array contains any NaN values.
        """
        return self._apply("isnan", (array,))

    def is_trainable(self, tensor: Tensor) -> bool:
        r"""Whether the given tensor is trainable.

        Args:
            tensor: The tensor to train.

        Returns:
            Whether the given tensor can be trained.
        """
        return self._apply("is_trainable", (tensor,))

    def lgamma(self, x: Tensor) -> Tensor:
        r"""
        The natural logarithm of the gamma function of ``x``.

        Args:
            x: The array to take the natural logarithm of the gamma function of.

        Returns:
            The natural logarithm of the gamma function of ``x``.
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

    def max(self, array: Tensor) -> Tensor:
        r"""The maximum value of an array.

        Args:
            array: The array to take the maximum value of.

        Returns:
            The maximum value of the array.
        """
        return self._apply("max", (array,))

    def maximum(self, a: Tensor, b: Tensor) -> Tensor:
        r"""
        The element-wise maximum of ``a`` and ``b``.

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

    def minimum(self, a: Tensor, b: Tensor) -> Tensor:
        r"""
        The element-wise minimum of ``a`` and ``b``.

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
        bounds: tuple[float | None, float | None],
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

        shape = shape if isinstance(shape, int) else tuple(shape)
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
        paddings: Sequence[tuple[int, int]],
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
        return self._apply("pad", (array, tuple(paddings), mode, constant_values))

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
        array = self.astensor(array)
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
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        return self._apply("reshape", (array, shape))

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

    def stack(self, arrays: Sequence[Tensor], axis: int = 0) -> Tensor:
        r"""Stack arrays in sequence along a new axis.

        Args:
            arrays: Sequence of tensors to stack
            axis: The axis along which to stack the arrays

        Returns:
            The stacked array
        """
        return self._apply("stack", (arrays, axis))

    def sum(self, array: Tensor, axis: int | Sequence[int] | None = None):
        r"""The sum of array.

        Args:
            array: The array to take the sum of
            axis (int | Sequence[int] | None): The axis/axes to sum over

        Returns:
            The sum of array
        """
        array = self.astensor(array)
        if axis is not None and not isinstance(axis, int):
            neg = [a for a in axis if a < 0]
            pos = [a for a in axis if a >= 0]
            axis = tuple(sorted(neg) + sorted(pos)[::-1])
        return self._apply("sum", (array, axis))

    def swapaxes(self, array: Tensor, axis1: int, axis2: int) -> Tensor:
        r"""
        Swap two axes of an array.

        Args:
            array: The array to swap axes of.
            axis1: The first axis to swap.
            axis2: The second axis to swap.

        Returns:
            The array with the axes swapped.
        """
        return self._apply("swapaxes", (array, axis1, axis2))

    def tensordot(self, a: Tensor, b: Tensor, axes: Sequence[int]) -> Tensor:
        r"""The tensordot product of ``a`` and ``b``.

        Args:
            a: The first array to take the tensordot product of
            b: The second array to take the tensordot product of
            axes: The axes to take the tensordot product over

        Returns:
            The tensordot product of ``a`` and ``b``
        """
        return self._apply("tensordot", (a, b, tuple(axes)))

    def tile(self, array: Tensor, repeats: Sequence[int]) -> Tensor:
        r"""The tiled array.

        Args:
            array: The array to tile
            repeats (tuple): number of times to tile the array along each axis

        Returns:
            The tiled array
        """
        return self._apply("tile", (array, tuple(repeats)))

    def trace(self, array: Tensor, dtype=None) -> Tensor:
        r"""The trace of array.

        Args:
            array: The array to take the trace of
            dtype (type): ``dtype`` of the output array

        Returns:
            The trace of array
        """
        return self._apply("trace", (array, dtype))

    def transpose(self, a: Tensor, perm: Sequence[int] | None = None):
        r"""The transposed arrays.

        Args:
            a: The array to transpose
            perm (tuple): permutation to apply to the array

        Returns:
            The transposed array
        """
        perm = tuple(perm) if perm is not None else None
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
        self,
        cost_fn: Callable,
        parameters: dict[str, list[Trainable]],
    ) -> tuple[Tensor, dict[str, list[Tensor]]]:
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

    def conditional(self, cond: Tensor, true_fn: Callable, false_fn: Callable, *args) -> Tensor:
        r"""Executes ``true_fn`` if ``cond`` is ``True``, otherwise ``false_fn``.

        Args:
            cond: The condition to check
            true_fn: The function to execute if ``cond`` is ``True``
            false_fn: The function to execute if ``cond`` is ``False``
            *args: The arguments to pass to ``true_fn`` and ``false_fn``
        Returns:
            The result of ``true_fn`` if ``cond`` is ``True``, otherwise ``false_fn``.
        """
        return self._apply("conditional", (cond, true_fn, false_fn, *args))

    def error_if(self, array: Tensor, condition: Tensor, msg: str):
        r"""Raises an error if ``condition`` is ``True``.

        Args:
            array: The array to check
            condition: The condition to check; should only use array elements in the condition
                And must be boolean.
            msg: The message to raise if ``condition`` is ``True``

        Returns:
            None
            Raises an error if at least one element of ``cond`` is True.
        """
        return self._apply("error_if", (array, condition, msg))

    def infinity_like(self, array: Tensor) -> Tensor:
        r"""Returns an array of infinities with the same shape as ``array``.

        Args:
            array: The array to take the shape of.

        Returns:
            An array of infinities with the same shape as ``array``.
        """
        return self._apply("infinity_like", (array,))

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

    def DefaultEuclideanOptimizer(self):
        r"""Default optimizer for the Euclidean parameters."""
        return self._apply("DefaultEuclideanOptimizer")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Fock lattice strategies
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def displacement(self, x: float, y: float, shape: tuple[int, int], tol: float = 1e-15):
        r"""
        Creates a single mode displacement matrix using a numba-based fock lattice strategy.

        Args:
            x: The displacement magnitude.
            y: The displacement angle.
            shape: The shape of the displacement matrix.
            tol: The tolerance to determine if the displacement is small enough to be approximated by the identity.

        Returns:
            The matrix representing the displacement gate.
        """
        return self._apply("displacement", (x, y), {"shape": shape, "tol": tol})

    def beamsplitter(self, theta: float, phi: float, shape: tuple[int, int, int, int], method: str):
        r"""
        Creates a beamsplitter matrix with given cutoffs using a numba-based fock lattice strategy.

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

    def squeezed(self, r: float, phi: float, shape: tuple[int, int]):
        r"""
        Creates a single mode squeezed state matrix using a numba-based fock lattice strategy.

        Args:
            r: Squeezing magnitude.
            phi: Squeezing angle.
            shape: Output shape of the two modes.

        Returns:
            The matrix representing the squeezed state.
        """
        return self._apply("squeezed", (r, phi, shape))

    def squeezer(self, r: float, phi: float, shape: tuple[int, int]):  # pragma: no cover
        r"""
        Creates a single mode squeezer matrix using a numba-based fock lattice strategy.

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

    def dagger(self, array: Tensor) -> Tensor:
        """The adjoint of ``array``. This operation swaps the first
        and second half of the indexes and then conjugates the matrix.

        Args:
            array: The array to take the adjoint of

        Returns:
            The adjoint of ``array``
        """
        N = len(array.shape) // 2
        perm = list(range(N, 2 * N)) + list(range(N))
        return self.conj(self.transpose(array, perm=perm))

    def unitary_to_orthogonal(self, U):
        r"""
        Unitary to orthogonal mapping.

        Args:
            U: The unitary matrix in ``U(n)``

        Returns:
            The orthogonal matrix in :math:`O(2n)`
        """
        X = self.real(U)
        Y = self.imag(U)
        return self.block([[X, -Y], [Y, X]])

    def random_symplectic(self, num_modes: int, max_r: float = 1.0) -> Tensor:
        r"""
        A random symplectic matrix in ``Sp(2*num_modes)``.

        Squeezing is sampled uniformly from 0.0 to ``max_r`` (1.0 by default).
        """
        if num_modes == 1:
            W = self.exp(1j * 2 * np.pi * settings.rng.uniform(size=(1, 1)))
            V = self.exp(1j * 2 * np.pi * settings.rng.uniform(size=(1, 1)))
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
        r"""
        A random orthogonal matrix in :math:`O(N)`.
        """
        if N == 1:
            return np.array([[1.0]])
        return ortho_group.rvs(dim=N, random_state=settings.rng)

    def random_unitary(self, N: int) -> Tensor:
        r"""
        A random unitary matrix in :math:`U(N)`.
        """
        if N == 1:
            return self.exp(1j * settings.rng.uniform(size=(1, 1)))
        return unitary_group.rvs(dim=N, random_state=settings.rng)

    @staticmethod
    @lru_cache
    def Xmat(num_modes: int):
        r"""
        The matrix :math:`X_n = \begin{bmatrix}0 & I_n\\ I_n & 0\end{bmatrix}.`

        Args:
            num_modes (int): positive integer

        Returns:
            The :math:`2N\times 2N` array
        """
        I = np.identity(num_modes)
        O = np.zeros((num_modes, num_modes))
        return np.block([[O, I], [I, O]])

    @staticmethod
    @lru_cache
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
    @lru_cache
    def rotmat(num_modes: int):
        "Rotation matrix from quadratures to complex amplitudes."
        I = np.identity(num_modes)
        return np.sqrt(0.5) * np.block([[I, 1j * I], [I, -1j * I]])

    @staticmethod
    @lru_cache
    def J(num_modes: int):
        """Symplectic form."""
        I = np.identity(num_modes)
        O = np.zeros_like(I)
        return np.block([[O, I], [-I, O]])

    def all_diagonals(self, rho: Tensor, real: bool) -> Tensor:
        """Returns all the diagonals of a density matrix."""
        cutoffs = rho.shape[: rho.ndim // 2]
        rho = self.reshape(rho, (int(np.prod(cutoffs)), int(np.prod(cutoffs))))
        diag = self.diag_part(rho)
        if real:
            return self.real(self.reshape(diag, cutoffs))

        return self.reshape(diag, cutoffs)

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
