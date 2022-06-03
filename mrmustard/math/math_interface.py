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

"""This module contains the :class:`Math` interface that every backend has to implement."""

from abc import ABC, abstractmethod
from functools import lru_cache
from itertools import product
import numpy as np
from scipy.special import binom
from numba import njit
from scipy.stats import unitary_group, ortho_group

from mrmustard.types import (
    List,
    Tensor,
    Matrix,
    Scalar,
    Vector,
    Sequence,
    Tuple,
    Optional,
    Dict,
    Trainable,
    Callable,
    Any,
)

# pylint: disable=too-many-public-methods
class MathInterface(ABC):
    r"""The interface that all backends must implement."""
    _euclidean_opt: type = None  # NOTE this is an object that

    # backend is a singleton
    __instance = None

    # pylint: disable=unused-argument
    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    @abstractmethod
    def __getattr__(self, name):
        ...  # pass the call to the actual backend

    # ~~~~~~~~~
    # Basic ops
    # ~~~~~~~~~

    @abstractmethod
    def abs(self, array: Tensor) -> Tensor:
        r"""Returns the absolute value of array.

        Args:
            array (array): array to take the absolute value of

        Returns:
            array: absolute value of array
        """

    @abstractmethod
    def any(self, array: Tensor) -> bool:
        r"""Returns ``True`` if any element of array is ``True``.

        Args:
            array (array): array to check

        Returns:
            bool: True if any element of array is True
        """

    @abstractmethod
    def arange(self, start: int, limit: int = None, delta: int = 1, dtype: Any = None) -> Tensor:
        r"""Returns an array of evenly spaced values within a given interval.

        Args:
            start (int): start of the interval
            limit (int): end of the interval
            delta (int): step size
            dtype (type): dtype of the returned array

        Returns:
            array: array of evenly spaced values
        """
        # NOTE: is float64 by default

    @abstractmethod
    def asnumpy(self, tensor: Tensor) -> Tensor:
        r"""Converts a tensor to a NumPy array.

        Args:
            tensor (array): tensor to convert

        Returns:
            array: NumPy array
        """

    @abstractmethod
    def assign(self, tensor: Tensor, value: Tensor) -> Tensor:
        r"""Assigns value to tensor.

        Args:
            tensor (array): tensor to assign to
            value (array): value to assign

        Returns:
            array: tensor with value assigned
        """

    @abstractmethod
    def astensor(self, array: Tensor, dtype: str) -> Tensor:
        r"""Converts a numpy array to a tensor.

        Args:
            array (array): numpy array to convert
            dtype (str): dtype of the tensor

        Returns:
            array: tensor with dtype
        """

    @abstractmethod
    def atleast_1d(self, array: Tensor, dtype=None) -> Tensor:
        r"""Returns an array with at least one dimension.

        Args:
            array (array): array to convert
            dtype (dtype): data type of the array

        Returns:
            array: array with at least one dimension
        """

    @abstractmethod
    def cast(self, array: Tensor, dtype) -> Tensor:
        r"""Casts ``array`` to ``dtype``.

        Args:
            array (array): array to cast
            dtype (dtype): data type to cast to

        Returns:
            array: array cast to dtype
        """

    @abstractmethod
    def clip(self, array: Tensor, a_min: float, a_max: float) -> Tensor:
        r"""Clips array to the interval ``[a_min, a_max]``.

        Args:
            array (array): array to clip
            a_min (float): minimum value
            a_max (float): maximum value

        Returns:
            array: clipped array
        """

    @abstractmethod
    def concat(self, values: Sequence[Tensor], axis: int) -> Tensor:
        r"""Concatenates values along the given axis.

        Args:
            values (array): values to concatenate
            axis (int): axis along which to concatenate

        Returns:
            array: concatenated values
        """

    @abstractmethod
    def conj(self, array: Tensor) -> Tensor:
        r"""Returns the complex conjugate of array.

        Args:
            array (array): array to take the complex conjugate of

        Returns:
            array: complex conjugate of array
        """

    @abstractmethod
    def constraint_func(
        self, bounds: Tuple[Optional[float], Optional[float]]
    ) -> Optional[Callable]:
        r"""Returns a constraint function for the given bounds.

        A constraint function will clip the value to the interval given by the bounds.

        .. note::

            The upper and/or lower bounds can be ``None``, in which case the constraint
            function will not clip the value.

        Args:
            bounds (tuple): bounds of the constraint

        Returns:
            function: constraint function
        """

    @abstractmethod
    def convolution(
        self,
        array: Tensor,
        filters: Tensor,
        padding="VALID",
        data_format="NWC",
    ) -> Tensor:  # TODO: remove strides and data_format?
        r"""Performs a convolution on array with filters.

        Args:
            array (array): array to convolve
            filters (array): filters to convolve with
            padding (str): padding mode
            data_format (str): data format of the array

        Returns:
            array: convolved array
        """

    @abstractmethod
    def cos(self, array: Tensor) -> Tensor:
        r"""Returns the cosine of array.

        Args:
            array (array): array to take the cosine of

        Returns:
            array: cosine of array
        """

    @abstractmethod
    def cosh(self, array: Tensor) -> Tensor:
        r"""Returns the hyperbolic cosine of array.

        Args:
            array (array): array to take the hyperbolic cosine of

        Returns:
            array: hyperbolic cosine of array
        """

    @abstractmethod
    def det(self, matrix: Tensor) -> Tensor:
        r"""Returns the determinant of matrix.

        Args:
            matrix (matrix): matrix to take the determinant of

        Returns:
            determinant of matrix
        """

    @abstractmethod
    def diag(self, array: Tensor, k: int) -> Tensor:
        r"""Returns the array made by inserting the given array along the :math:`k`-th diagonal.

        Args:
            array (array): array to insert
            k (int): kth diagonal to insert array into

        Returns:
            array: array with array inserted into the kth diagonal
        """

    @abstractmethod
    def diag_part(self, array: Tensor) -> Tensor:
        r"""Returns the array of the main diagonal of array.

        Args:
            array (array): array to extract the main diagonal of

        Returns:
            array: array of the main diagonal of array
        """

    @abstractmethod
    def eigvals(self, tensor: Tensor) -> Tensor:
        r"""Returns the eigenvalues of a matrix."""

    @abstractmethod
    def einsum(self, string: str, *tensors) -> Tensor:
        r"""Returns the result of the Einstein summation convention on the tensors.

        Args:
            string (str): string of the Einstein summation convention
            tensors (array): tensors to perform the Einstein summation on

        Returns:
            array: result of the Einstein summation convention
        """

    @abstractmethod
    def exp(self, array: Tensor) -> Tensor:
        r"""Returns the exponential of array element-wise.

        Args:
            array (array): array to take the exponential of

        Returns:
            array: exponential of array
        """

    @abstractmethod
    def expand_dims(self, array: Tensor, axis: int) -> Tensor:
        r"""Returns the array with an additional dimension inserted at the given axis.

        Args:
            array (array): array to expand
            axis (int): axis to insert the new dimension

        Returns:
            array: array with an additional dimension inserted at the given axis
        """

    @abstractmethod
    def expm(self, matrix: Tensor) -> Tensor:
        r"""Returns the matrix exponential of matrix.

        Args:
            matrix (matrix): matrix to take the exponential of

        Returns:
            matrix: exponential of matrix
        """

    @abstractmethod
    def eye(self, size: int, dtype) -> Tensor:
        r"""Returns the identity matrix of size.

        Args:
            size (int): size of the identity matrix
            dtype (dtype): data type of the identity matrix

        Returns:
            matrix: identity matrix
        """

    @abstractmethod
    def from_backend(self, value: Any) -> bool:
        r"""Returns whether the given tensor is a tensor of the concrete backend."""

    @abstractmethod
    def gather(self, array: Tensor, indices: Tensor, axis: int) -> Tensor:
        r"""Returns the values of the array at the given indices.

        Args:
            array (array): array to gather values from
            indices (array): indices to gather values from
            axis (int): axis to gather values from

        Returns:
            array: values of the array at the given indices
        """

    @abstractmethod
    def hash_tensor(self, tensor: Tensor) -> int:
        r"""Returns the hash of the given tensor.

        Args:
            tensor (array): tensor to hash

        Returns:
            int: hash of the given tensor
        """

    @abstractmethod
    def hermite_renormalized(self, A: Matrix, B: Vector, C: Scalar, shape: Sequence[int]) -> Tensor:
        r"""Returns the array of hermite renormalized polynomials of the given coefficients.

        Args:
            A (array): Matrix coefficient of the hermite polynomial
            B (array): Vector coefficient of the hermite polynomial
            C (array): Scalar coefficient of the hermite polynomial
            shape (tuple): shape of the hermite polynomial

        Returns:
            array: renormalized hermite polynomials
        """

    @abstractmethod
    def imag(self, array: Tensor) -> Tensor:
        r"""Returns the imaginary part of array.

        Args:
            array (array): array to take the imaginary part of

        Returns:
            array: imaginary part of array
        """

    @abstractmethod
    def inv(self, tensor: Tensor) -> Tensor:
        r"""Returns the inverse of tensor.

        Args:
            tensor (array): tensor to take the inverse of

        Returns:
            array: inverse of tensor
        """

    @abstractmethod
    def is_trainable(self, tensor: Tensor) -> bool:
        r"""Returns whether the given tensor is trainable."""

    @abstractmethod
    def lgamma(self, x: Tensor) -> Tensor:
        r"""Returns the natural logarithm of the gamma function of ``x``.

        Args:
            x (array): array to take the natural logarithm of the gamma function of

        Returns:
            array: natural logarithm of the gamma function of ``x``
        """

    @abstractmethod
    def log(self, x: Tensor) -> Tensor:
        r"""Returns the natural logarithm of ``x``.

        Args:
            x (array): array to take the natural logarithm of

        Returns:
            array: natural logarithm of ``x``
        """

    @abstractmethod
    def matmul(
        self,
        a: Tensor,
        b: Tensor,
        transpose_a=False,
        transpose_b=False,
        adjoint_a=False,
        adjoint_b=False,
    ) -> Tensor:
        r"""Returns the matrix product of ``a`` and ``b``.

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

    @abstractmethod
    def matvec(self, a: Matrix, b: Vector, transpose_a=False, adjoint_a=False) -> Tensor:
        r"""Returns the matrix vector product of ``a`` (matrix) and ``b`` (vector).

        Args:
            a (array): matrix to multiply
            b (array): vector to multiply
            transpose_a (bool): whether to transpose ``a``
            adjoint_a (bool): whether to adjoint ``a``

        Returns:
            array: matrix vector product of ``a`` and ``b``
        """

    @abstractmethod
    def maximum(self, a: Tensor, b: Tensor) -> Tensor:
        r"""Returns the element-wise maximum of ``a`` and ``b``.

        Args:
            a (array): first array to take the maximum of
            b (array): second array to take the maximum of

        Returns:
            array: element-wise maximum of ``a`` and ``b``
        """

    @abstractmethod
    def minimum(self, a: Tensor, b: Tensor) -> Tensor:
        r"""Returns the element-wise minimum of ``a`` and ``b``.

        Args:
            a (array): first array to take the minimum of
            b (array): second array to take the minimum of

        Returns:
            array: element-wise minimum of ``a`` and ``b``
        """

    @abstractmethod
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

    @abstractmethod
    def new_constant(self, value: Tensor, name: str, dtype: Any) -> Tensor:
        r"""Returns a new constant with the given value.

        Args:
            value (array): value of the new constant
            name (str): name of the new constant
            dtype (type): dtype of the array

        Returns:
            array: new constant
        """

    @abstractmethod
    def norm(self, array: Tensor) -> Tensor:
        r"""Returns the norm of array.

        Args:
            array (array): array to take the norm of

        Returns:
            array: norm of array
        """

    @abstractmethod
    def ones(self, shape: Sequence[int], dtype) -> Tensor:
        r"""Returns an array of ones with the given ``shape`` and ``dtype``.

        Args:
            shape (tuple): shape of the array
            dtype (type): dtype of the array

        Returns:
            array: array of ones
        """
        # NOTE : should be float64 by default

    @abstractmethod
    def ones_like(self, array: Tensor) -> Tensor:
        r"""Returns an array of ones with the same shape and ``dtype`` as ``array``.

        Args:
            array (array): array to take the shape and dtype of

        Returns:
            array: array of ones
        """

    @abstractmethod
    def outer(self, array1: Tensor, array2: Tensor) -> Tensor:
        r"""Returns the outer product of ``array1`` and ``array2``.

        Args:
            array1 (array): first array to take the outer product of
            array2 (array): second array to take the outer product of

        Returns:
            array: outer product of array1 and array2
        """

    @abstractmethod
    def pad(
        self, array: Tensor, paddings: Sequence[Tuple[int, int]], mode="CONSTANT", constant_values=0
    ) -> Tensor:
        r"""Returns the padded array.

        Args:
            array (array): array to pad
            paddings (tuple): paddings to apply
            mode (str): mode to apply the padding
            constant_values (int): constant values to use for padding

        Returns:
            array: padded array
        """

    @abstractmethod
    def pinv(self, matrix: Tensor) -> Tensor:
        r"""Returns the pseudo-inverse of matrix.

        Args:
            matrix (array): matrix to take the pseudo-inverse of

        Returns:
            array: pseudo-inverse of matrix
        """

    @abstractmethod
    def pow(self, x: Tensor, y: Tensor) -> Tensor:
        r"""Returns :math:`x^y`. Broadcasts ``x`` and ``y`` if necessary.
        Args:
            x (array): base
            y (array): exponent

        Returns:
            array: :math:`x^y`
        """

    @abstractmethod
    def real(self, array: Tensor) -> Tensor:
        r"""Returns the real part of ``array``.

        Args:
            array (array): array to take the real part of

        Returns:
            array: real part of ``array``
        """

    @abstractmethod
    def reshape(self, array: Tensor, shape: Sequence[int]) -> Tensor:
        r"""Returns the reshaped array.

        Args:
            array (array): array to reshape
            shape (tuple): shape to reshape the array to

        Returns:
            array: reshaped array
        """

    @abstractmethod
    def sin(self, array: Tensor) -> Tensor:
        r"""Returns the sine of ``array``.

        Args:
            array (array): array to take the sine of

        Returns:
            array: sine of ``array``
        """

    @abstractmethod
    def sinh(self, array: Tensor) -> Tensor:
        r"""Returns the hyperbolic sine of ``array``.

        Args:
            array (array): array to take the hyperbolic sine of

        Returns:
            array: hyperbolic sine of ``array``
        """

    @abstractmethod
    def sqrt(self, x: Tensor, dtype=None) -> Tensor:
        r"""Returns the square root of ``x``.

        Args:
            x (array): array to take the square root of
            dtype (type): ``dtype`` of the output array

        Returns:
            array: square root of ``x``
        """

    @abstractmethod
    def sqrtm(self, tensor: Tensor) -> Tensor:
        r"""Returns the matrix square root."""

    @abstractmethod
    def sum(self, array: Tensor, axes: Sequence[int] = None):
        r"""Returns the sum of array.

        Args:
            array (array): array to take the sum of
            axes (tuple): axes to sum over

        Returns:
            array: sum of array
        """

    @abstractmethod
    def tensordot(self, a: Tensor, b: Tensor, axes: Sequence[int]) -> Tensor:
        r"""Returns the tensordot product of ``a`` and ``b``.

        Args:
            a (array): first array to take the tensordot product of
            b (array): second array to take the tensordot product of
            axes (tuple): axes to take the tensordot product over

        Returns:
            array: tensordot product of ``a`` and ``b``
        """

    @abstractmethod
    def tile(self, array: Tensor, repeats: Sequence[int]) -> Tensor:
        r"""Returns the tiled array.

        Args:
            array (array): array to tile
            repeats (tuple): number of times to tile the array along each axis

        Returns:
            array: tiled array
        """

    @abstractmethod
    def trace(self, array: Tensor, dtype: Any = None) -> Tensor:
        r"""Returns the trace of array.

        Args:
            array (array): array to take the trace of
            dtype (type): ``dtype`` of the output array

        Returns:
            array: trace of array
        """

    @abstractmethod
    def transpose(self, a: Tensor, perm: Sequence[int] = None):
        r"""Returns the transposed arrays.

        Args:
            a (array): array to transpose
            perm (tuple): permutation to apply to the array

        Returns:
            array: transposed array
        """

    @abstractmethod
    def unique_tensors(self, lst: List[Tensor]) -> List[Tensor]:
        r"""Returns the tensors in ``lst`` without duplicates and non-tensors.

        Args:
            lst (list): list of tensors to remove duplicates and non-tensors from.

        Returns:
            list: list of tensors without duplicates and non-tensors.
        """

    @abstractmethod
    def update_tensor(self, tensor: Tensor, indices: Tensor, values: Tensor) -> Tensor:
        r"""Updates a tensor in place with the given values.

        Args:
            tensor (array): tensor to update
            indices (array): indices to update
            values (array): values to update
        """

    @abstractmethod
    def update_add_tensor(self, tensor: Tensor, indices: Tensor, values: Tensor) -> Tensor:
        r"""Updates a tensor in place by adding the given values.

        Args:
            tensor (array): tensor to update
            indices (array): indices to update
            values (array): values to add
        """

    @abstractmethod
    def value_and_gradients(
        self, cost_fn: Callable, parameters: Dict[str, List[Trainable]]
    ) -> Tuple[Tensor, Dict[str, List[Tensor]]]:
        r"""Returns the loss and gradients of the given cost function.

        Args:
            cost_fn (callable): cost function to compute the loss and gradients of
            parameters (dict): parameters to compute the loss and gradients of

        Returns:
            tuple: loss and gradients (dict) of the given cost function
        """

    @abstractmethod
    def zeros(self, shape: Sequence[int], dtype) -> Tensor:
        r"""Returns an array of zeros with the given shape and ``dtype``.

        Args:
            shape (tuple): shape of the array
            dtype (type): dtype of the array

        Returns:
            array: array of zeros
        """

    @abstractmethod
    def zeros_like(self, array: Tensor) -> Tensor:
        r"""Returns an array of zeros with the same shape and ``dtype`` as ``array``.

        Args:
            array (array): array to take the shape and ``dtype`` of

        Returns:
            array: array of zeros
        """

    @property
    def euclidean_opt(self):
        r"""Returns the configured Euclidean optimizer."""
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
        r"""Returns the adjoint of ``array``.

        Args:
            array (array): array to take the adjoint of

        Returns:
            array: adjoint of ``array``
        """
        return self.conj(self.transpose(array))

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
            W = np.exp(1j * np.random.uniform(size=(1, 1)))
            V = np.exp(1j * np.random.uniform(size=(1, 1)))
        else:
            W = unitary_group.rvs(dim=num_modes)
            V = unitary_group.rvs(dim=num_modes)
        r = np.random.uniform(low=0.0, high=max_r, size=num_modes)
        OW = self.unitary_to_orthogonal(W)
        OV = self.unitary_to_orthogonal(V)
        dd = self.diag(self.concat([self.exp(-r), np.exp(r)], axis=0), k=0)
        return OW @ dd @ OV

    @staticmethod
    def random_orthogonal(N: int) -> Tensor:
        """A random orthogonal matrix in :math:`O(N)`."""
        if N == 1:
            return np.array([[1.0]])
        return ortho_group.rvs(dim=N)

    def random_unitary(self, N: int) -> Tensor:
        """a random unitary matrix in :math:`U(N)`"""
        if N == 1:
            return self.exp(1j * np.random.uniform(size=(1, 1)))
        return unitary_group.rvs(dim=N)

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
        r"""Returns the matrix :math:`X_n = \begin{bmatrix}0 & I_n\\ I_n & 0\end{bmatrix}.`

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



    @abstractmethod
    def sparse_matvec(self, matrix: Tensor, vector: Tensor, m_outmodes: List[int], m_inmodes: List[int], v_outmodes: List[int], like_0:bool) -> Tensor:
        r"""Mode-wise matrix-vector multiplication of a phase space matrix and a phase space vector.
        Assumes inputs are in xxpp ordering.

        Args:
            matrix (array): :math:`2M\times 2M` array
            vector (array): :math:`2N` vector
            m_outmodes (list(int)): list of ``M`` output modes of the matrix
            m_inmodes (list(int)): list of ``M`` input modes of the matrix
            v_outmodes (list(int)): list of ``N`` modes of the vector
            like_0 (bool): whether matrix is like_0 or not
        Returns:
            array: :math:`2N` new vector
        """
        ...


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


@njit
def sparse_matvec_data(matrix: Tensor, vector: Tensor, m_modes: Tuple[int], v_modes: Tuple[int], like_0:bool) -> tuple:
    r"""Computes the metadata for a sparse matrix-vector multiplication.
    """
    # batch dimensions
    B1 = matrix.shape[0]
    B2 = vector.shape[0]
    # matrix and vector dimensions
    M = matrix.shape[-1] // 2
    V = vector.shape[-1] // 2
    final_modes = [v for v in v_modes if v in m_modes] if like_0 else list(v_modes)
    F = len(final_modes)

    # at which index to read/write a given mode: (note that numba doesn't support dict comprehensions)
    findices = {}
    for i,m in enumerate(final_modes):
        findices[m] = i
    mindices = {}
    for i,m in enumerate(m_modes):
        mindices[m] = i
    vindices = {}
    for i,m in enumerate(v_modes):
        vindices[m] = i

    return final_modes, findices, mindices, vindices, B1, B2, F, M, V

@njit
def numba_sparse_matvec(matrix: Tensor, vector: Tensor, m_modes: Tuple[int], v_modes: Tuple[int], like_0:bool):
    r"""Numba implementation of the mode-wise matrix-vector multiplication of
    a batch of matrices and a batch of vectors in phase space. Assumes inputs are in xxpp ordering.
    Note that "sparse" is indended in the sense of modes, i.e. the matrix can contain
    fewer modes than the vector or the vector can contain fewer modes than the matrix.
    The operation will be performed only on the modes specified in the arguments.

    Args:
        matrix (array): :math:`B \times 2M\times 2M` batched array
        vector (array): :math:`B \times 2N` batched vector
        m_modes (list(int)): list of ``M`` modes of the matrix
        v_outmodes (list(int)): list of ``N`` modes of the vector
        like_0 (bool): whether the values outside the matrix are to be considered as 0s.
    Returns:
        array: :math: resulting vector (can have the same modes as the input vector or fewer)
    """
    final_modes, findices, mindices, vindices, B1, B2, F, M, V = sparse_matvec_data(matrix, vector, m_modes, v_modes, like_0)

    new_vec = np.zeros((B1*B2, 2*F), dtype=vector.dtype)
    for b1 in range(B1):
        for b2 in range(B2):
            b = b1*B2 + b2
            for f in final_modes: # filling the final vector entries
                if f in m_modes: # we need to multiply only if matrix acts on mode f
                    for n in final_modes:
                        if n in m_modes:
                            new_vec[b,findices[f]] += matrix[b1,mindices[f], mindices[n]] * vector[b2,vindices[n]] + matrix[b1,mindices[f], mindices[n]+M] * vector[b2,vindices[n]+V]
                            new_vec[b,findices[f]+F] += matrix[b1,mindices[f]+M, mindices[n]] * vector[b2,vindices[n]] + matrix[b1,mindices[f]+M, mindices[n]+M] * vector[b2,vindices[n]+V]
                elif not like_0: # if mode is not acted on we ignore it (like_0) or copy it (not like_0)
                    new_vec[b,findices[f]] = vector[b2,vindices[f]]
                    new_vec[b,findices[f]+F] = vector[b2,V+vindices[f]]
    return new_vec


@njit
def numba_sparse_matvec_vjp(self, dmatvec: Tensor, matrix: Tensor, vector: Tensor, m_modes: Tuple[int], v_modes: Tuple[int], like_0:bool):
    r"""Numba implementation of the vector-jacobian product of the mode-wise matrix-vector multiplication of
    a batch of matrices and a batch of vectors in phase space. Assumes inputs are in xxpp ordering.
    Note that "sparse" is indended in the sense of modes, i.e. the matrix can contain
    fewer mode than the vector or the vector can contain fewer modes than the matrix.
    The operation will be performed only on the modes specified in the arguments.

    Args:
        dmatvec (array): :math:`B1\times B2\times 2K` upstream gradient of the cost function with respect to a batch of matrix-vector products
        matrix (array): :math:`B1\times 2M\times 2M` array
        vector (array): :math:`B2\times 2N` vector
        m_modes (list(int)): list of ``M`` modes of the matrix (all elements of the batch are assumed to have the same modes)
        v_modes (list(int)): list of ``N`` modes of the vector (all elements of the batch are assumed to have the same modes)
        like_0 (bool): whether matrix is like_0 or not (all elements of the batch are assumed to have the same like_0)
    Returns:
        dv (array): :math:`B\times 2N` downstream gradient of the cost function with respect to `vector`
        dm (array): :math:`B\times 2M\times 2M` downstream gradient of the cost function with respect to `matrix`
    """
    final_modes, findices, mindices, vindices, B1, B2, F, M, V = sparse_matvec_data(matrix, vector, m_modes, v_modes, like_0)

    dm = np.zeros((B1, 2*M, 2*M), dtype=vector.dtype.name) # dL/dm_ik = sum_j dL/dmatvec_j * dmatvec_j/dm_ik = dL/dmatvec_i * v_k because dmatvec_j/dm_ik = delta_ij * v_k
    dv = np.zeros((B2, 2*V), dtype=vector.dtype.name)   # dL/dv_i = sum_j dL/dmatvec_j * dmatvec_j/dv_i = sum_j dL/dmatvec_j * m_ji
    for b1 in range(B1):
        for b2 in range(B2):
            b = b1*B2 + b2
            for f in final_modes:
                if f in m_modes:
                    for n in final_modes:
                        dv[b2, vindices[n]] += dmatvec[b, findices[f]] * matrix[b1, mindices[f], mindices[n]] + dmatvec[b, findices[f]+F] * matrix[b1, mindices[f]+M, mindices[n]]
                        dv[b2, vindices[n]+V] += dmatvec[b, findices[f]] * matrix[b1, mindices[f], mindices[n]+M] + dmatvec[b, findices[f]+F] * matrix[b1, mindices[f]+M, mindices[n]+M]
                        dm[b1, mindices[f], mindices[n]] += dmatvec[b, findices[f]] * vector[b2, vindices[n]]
                        dm[b1, mindices[f], mindices[n]+M] += dmatvec[b, findices[f]] * vector[b2, vindices[n]+V]
                        dm[b1, mindices[f]+M, mindices[n]] += dmatvec[b, findices[f]+F] * vector[b2, vindices[n]]
                        dm[b1, mindices[f]+M, mindices[n]+M] += dmatvec[b, findices[f]+F] * vector[b2, vindices[n]+V]
                elif not like_0:
                    dv[b2, vindices[f]] = dmatvec[b, findices[f]]
                    dv[b2, vindices[f]+V] = dmatvec[b, findices[f]+F]
    return dv, dm


@njit
def sparse_matmul_data(matrix1: Tensor, matrix2: Tensor, m1_modes: Tuple[int], m2_modes: Tuple[int], m1like_0: bool, m2like_0: bool) -> tuple:
    r"""Computes the data required for the mode-wise matrix multiplication of two matrices."""
    if m1like_0:  # final modes are a subset of m1_modes
        if m2like_0: # final modes are a subset of m2_modes
            final_modes = list(set(m1_modes).intersection(m2_modes))
        else:
            final_modes = list(m1_modes)
    else:
        if m2like_0: # final modes are a subset of m2_modes
            final_modes = list(m2_modes)
        else:
            final_modes = list(set(m1_modes).union(m2_modes))

        B1 = matrix1.shape[0]
        B2 = matrix2.shape[0]
        M = matrix1.shape[-1] // 2
        N = matrix2.shape[-1] // 2
        F = len(final_modes)

        # at which index to write a given mode:
        findices = {m:i for i,m in enumerate(final_modes)}
        m1indices = {m:i for i,m in enumerate(m1_modes)}
        m2indices = {m:i for i,m in enumerate(m2_modes)}

        return final_modes, findices, m1indices, m2indices, B, F, M, N

@njit
def numba_sparse_matmul(matrix1: Tensor, matrix2: Tensor, m1_modes: List[int], m2_modes: List[int], m1like_0:bool, m2like_0:bool):
    r"""Numba implementation of the mode-wise ("sparse") matrix-matrix multiplication `matrix1 @ matrix2`.
    Assumes inputs are in xxpp ordering.

    Args:
        matrix1 (array): :math:`B1\times 2M\times 2M` array
        matrix2 (array): :math:`B2\times 2N\times 2N` array
        m1_modes (list(int)): list of ``M`` modes of the first matrix
        m2_modes (list(int)): list of ``N`` modes of the second matrix
        m1like_0 (bool): whether first matrix is like_0 or not
        m2like_0 (bool): whether second matrix is like_0 or not
    Returns:
        new_matrix (array): :math:`B_1 B_2 \times 2F\times 2F` array where F is determined by the other arguments

    """
    final_modes, findices, m1indices, m2indices, B1, B2, F, M, N = sparse_matmul_data(matrix1, matrix2, m1_modes, m2_modes, m1like_0, m2like_0)
    
    new_matrix = np.zeros((B1*B2, 2*F, 2*F), dtype=matrix1.dtype)  # TODO: revisit dtype
    for b1 in range(B1):
        for b2 in range(B2):
            b = b1*B2 + b2
            for m in final_modes:
                for n in final_modes:
                    if m in m1_modes:
                        if n in m2_modes:  # if mode goes through both, add contribution
                            for p in final_modes:
                                new_matrix[b, findices[m], findices[n]] += matrix1[b1, m1indices[m], m1indices[p]] * matrix2[b2, m2indices[p], m2indices[n]] + matrix1[b1, m1indices[m], m1indices[p]+M] * matrix2[b2, m2indices[p]+N, m2indices[n]]
                                new_matrix[b, findices[m]+F, findices[n]] += matrix1[b1, m1indices[m]+M, m1indices[p]] * matrix2[b2, m2indices[p], m2indices[n]] + matrix1[b1, m1indices[m]+M, m1indices[p]+M] * matrix2[b2, m2indices[p]+N, m2indices[n]]
                                new_matrix[b, findices[m], findices[n]+F] += matrix1[b1, m1indices[m], m1indices[p]] * matrix2[b2, m2indices[p], m2indices[n]+N] + matrix1[b1, m1indices[m], m1indices[p]+M] * matrix2[b2, m2indices[p]+N, m2indices[n]+N]
                                new_matrix[b, findices[m]+F, findices[n]+F] += matrix1[b1, m1indices[m]+M, m1indices[p]] * matrix2[b2, m2indices[p], m2indices[n]+N] + matrix1[b1, m1indices[m]+M, m1indices[p]+M] * matrix2[b2, m2indices[p]+N, m2indices[n]+N]
                        elif not m2like_0: # if n is not in m2_modes it contributes only if m2 is not like_0, in which case it copies the mode from m1
                            new_matrix[b, findices[m], findices[n]] += matrix1[b1, m1indices[m], m1indices[n]]
                            new_matrix[b, findices[m]+F, findices[n]] += matrix1[b1, m1indices[m]+M, m1indices[n]]
                            new_matrix[b, findices[m], findices[n]+F] += matrix1[b1, m1indices[m], m1indices[n]+M]
                            new_matrix[b, findices[m]+F, findices[n]+F] += matrix1[b1, m1indices[m]+M, m1indices[n]+M]
                    elif not m1like_0:  # if m is not in m1_modes it matters only if m1 is not like_0, in which case it copies the mode from m2
                        new_matrix[b, findices[m], findices[n]] += matrix2[b2, m2indices[m], m2indices[n]]
                        new_matrix[b, findices[m]+F, findices[n]] += matrix2[b2, m2indices[m]+N, m2indices[n]]
                        new_matrix[b, findices[m], findices[n]+F] += matrix2[b2, m2indices[m], m2indices[n]+N]
                        new_matrix[b, findices[m]+F, findices[n]+F] += matrix2[b2, m2indices[m]+N, m2indices[n]+N]
    return new_matrix
            

def numba_sparse_matmul_vjp(dmatmul: Tensor, matrix1: Tensor, matrix2: Tensor, m1_modes: Tuple[int], m2_modes: Tuple[int], m1like_0:bool, m2like_0:bool):
    r"""Numba implementation of the mode-wise ("sparse") matrix-matrix multiplication `matrix1 @ matrix2`'s Jacobian.
    Assumes inputs are in xxpp ordering.

    Args:
        dmatmul (array): :math:`B_1B_2 \times 2M\times 2M` array the upstream gradient of the cost with respect to the matrix-matrix multiplication
        matrix1 (array): :math:`B_1\times 2M\times 2M` array
        matrix2 (array): :math:`B_2\times 2N\times 2N` array
        m1_modes (list(int)): list of ``M`` modes of the first matrix
        m2_modes (list(int)): list of ``N`` modes of the second matrix
        m1like_0 (bool): whether first matrix is like_0 or not
        m2like_0 (bool): whether second matrix is like_0 or not
    Returns:
        dmatrix1 (array): :math:`B_1\times 2M\times 2M` array the downstream gradient of the cost with respect to the first matrix
        dmatrix2 (array): :math:`B_2\times 2N\times 2N` array the downstream gradient of the cost with respect to the second matrix

    """
    final_modes, findices, m1indices, m2indices, B1, B2, F, M, N = sparse_matmul_data(matrix1, matrix2, m1_modes, m2_modes, m1like_0, m2like_0)
    dmatrix1 = np.zeros((B1, 2*M, 2*M), dtype=matrix1.dtype)
    dmatrix2 = np.zeros((B2, 2*N, 2*N), dtype=matrix2.dtype)
    for b1 in range(B1):
        for b2 in range(B2):
            b = b1*B2 + b2
            for m in final_modes:
                for n in final_modes:
                    if m in m1_modes:
                        if n in m2_modes:
                            for p in final_modes:
                                dmatrix1[b1, m1indices[m], m1indices[p]] += dmatmul[b, findices[m], findices[n]] * matrix2[b2, m2indices[p], m2indices[n]] + dmatmul[b, findices[m], findices[n]+F] * matrix2[b2, m2indices[p], m2indices[n]+N]
                                dmatrix1[b1, m1indices[m], m1indices[p]+M] += dmatmul[b, findices[m], findices[n]] * matrix2[b2, m2indices[p]+N, m2indices[n]] + dmatmul[b, findices[m], findices[n]+F] * matrix2[b2, m2indices[p]+N, m2indices[n]+N]
                                dmatrix1[b1, m1indices[m]+M, m1indices[p]] += dmatmul[b, findices[m]+F, findices[n]] * matrix2[b2, m2indices[p], m2indices[n]] + dmatmul[b, findices[m]+F, findices[n]+F] * matrix2[b2, m2indices[p], m2indices[n]+N]
                                dmatrix1[b1, m1indices[m]+M, m1indices[p]+M] += dmatmul[b, findices[m]+F, findices[n]] * matrix2[b2, m2indices[p]+N, m2indices[n]] + dmatmul[b, findices[m]+F, findices[n]+F] * matrix2[b2, m2indices[p]+N, m2indices[n]+N]
                                dmatrix2[b2, m2indices[p], m2indices[n]] += dmatmul[b, findices[m], findices[n]] * matrix1[b1, m1indices[m], m1indices[p]] + dmatmul[b, findices[m]+F, findices[n]] * matrix1[b1, m1indices[m]+M, m1indices[p]]
                                dmatrix2[b2, m2indices[p], m2indices[n]+N] += dmatmul[b, findices[m], findices[n]+F] * matrix1[b1, m1indices[m]+M, m1indices[p]] + dmatmul[b, findices[m]+F, findices[n]+F] * matrix1[b1, m1indices[m]+M, m1indices[p]+M]
                                dmatrix2[b2, m2indices[p]+N, m2indices[n]] += dmatmul[b, findices[m], findices[n]] * matrix1[b1, m1indices[m], m1indices[p]+M] + dmatmul[b, findices[m]+F, findices[n]] * matrix1[b1, m1indices[m]+M, m1indices[p]+M]
                                dmatrix2[b2, m2indices[p]+N, m2indices[n]+N] += dmatmul[b, findices[m], findices[n]+F] * matrix1[b1, m1indices[m], m1indices[p]+M] + dmatmul[b, findices[m]+F, findices[n]+F] * matrix1[b1, m1indices[m]+M, m1indices[p]+M]
                        elif not m2like_0:
                            dmatrix1[b1, m1indices[m], m1indices[n]] += dmatmul[b, findices[m], findices[n]]
                            dmatrix1[b1, m1indices[m]+M, m1indices[n]] += dmatmul[b, findices[m]+F, findices[n]]
                            dmatrix1[b1, m1indices[m], m1indices[n]+M] += dmatmul[b, findices[m], findices[n]+F]
                            dmatrix1[b1, m1indices[m]+M, m1indices[n]+M] += dmatmul[b, findices[m]+F, findices[n]+F]
                    elif not m1like_0:
                        dmatrix2[b2, m2indices[m], m2indices[n]] += dmatmul[b, findices[m], findices[n]]
                        dmatrix2[b2, m2indices[m]+N, m2indices[n]] += dmatmul[b, findices[m]+F, findices[n]]
                        dmatrix2[b2, m2indices[m], m2indices[n]+N] += dmatmul[b, findices[m], findices[n]+F]
                        dmatrix2[b2, m2indices[m]+N, m2indices[n]+N] += dmatmul[b, findices[m]+F, findices[n]+F]
    return dmatrix1, dmatrix2



def sparse_vec_add(vec1, vec2, modes1, modes2):
    B1 = len(vec1)
    B2 = len(vec2)
    fmodes = set(modes1).union(modes2)
    F = len(fmodes)

    vec = np.zeros((B1*B2, 2*F), dtype=vec1.dtype)
    for b1 in range(B1):
        for b2 in range(B2):
            b = b1*B2 + b2
            for f in fmodes:
                if f in modes1:
                    vec[b, fmodes[f]] += vec1[b1, modes1[f]]
                    vec[b, fmodes[f]+F] += vec1[b1, modes1[f]+F]
                if f in modes2:
                    vec[b, fmodes[f]] += vec2[b2, modes2[f]]
                    vec[b, fmodes[f]+F] += vec2[b2, modes2[f]+F]
    return vec


def sparse_mat_add(mat1, mat2, modes1, modes2, m1like_0, m2like_0):
    B1 = len(mat1)
    B2 = len(mat2)
    fm


#
#  def sparse_matadd_data(matrix1: Tensor, matrix2: Tensor, m1_modes: List[int], m2_modes: List[int], m1like_0:bool, m2like_0:bool):
#     r"""Computes the data required for the mode-wise addition of two matrices."""
#     assert m1like_0 or m2like_0  # TODO we can't add two like_1 matrices unless we say the result is like_1 and we store a 2 as overall coefficient
    
#     B1 = matrix1.shape[0]
#     B2 = matrix2.shape[0]
#     M = matrix1.shape[-1] // 2
#     N = matrix2.shape[-1] // 2

#     # if self contains other or other contains self (in the sense of the modes involved)
#     # then we can simply update the containing matrix. Otherwise we need to fill a new zero matrix.
#     if set(m1_modes).issubset(m2_modes):
#         final_modes = m2_modes
#     elif set(m2_modes).issubset(m1_modes):
#         final_modes = m1_modes
#     else:
#         final_modes = list(set(m1_modes).union(m2_modes))
    
#     F = len(final_modes)

#     # at which index to write a given mode:
#     findices = {m:i for i,m in enumerate(final_modes)}
#     m1indices = {m:i for i,m in enumerate(m1_modes)}
#     m2indices = {m:i for i,m in enumerate(m2_modes)}

#     return final_modes, findices, m1indices, m2indices, B, F, M, N




