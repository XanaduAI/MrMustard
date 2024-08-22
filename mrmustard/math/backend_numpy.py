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

"""This module contains the numpy backend."""

# pylint: disable = missing-function-docstring, missing-class-docstring, fixme

from __future__ import annotations

from math import lgamma as mlgamma
from typing import Sequence

import numpy as np
import scipy as sp
from scipy.linalg import expm as scipy_expm
from scipy.linalg import sqrtm as scipy_sqrtm
from scipy.special import xlogy as scipy_xlogy
from scipy.stats import multivariate_normal

from ..utils.settings import settings
from .autocast import Autocast
from .backend_base import BackendBase
from .lattice.strategies import binomial, vanilla, vanilla_average, vanilla_batch
from .lattice.strategies.compactFock.inputValidation import (
    hermite_multidimensional_1leftoverMode,
    hermite_multidimensional_diagonal,
    hermite_multidimensional_diagonal_batch,
)


# pylint: disable=too-many-public-methods
class BackendNumpy(BackendBase):  # pragma: no cover
    r"""
    A numpy backend.
    """

    int32 = np.int32
    float32 = np.float32
    float64 = np.float64
    complex64 = np.complex64
    complex128 = np.complex128

    def __init__(self):
        super().__init__(name="numpy")

    def __repr__(self) -> str:
        return "BackendNumpy()"

    def abs(self, array: np.ndarray) -> np.ndarray:
        return np.abs(array)

    def allclose(self, array1: np.array, array2: np.array, atol: float) -> bool:
        array1 = self.asnumpy(array1)
        array2 = self.asnumpy(array2)
        if array1.shape != array2.shape:
            raise ValueError("Cannot compare arrays of different shapes.")
        return np.allclose(array1, array2, atol=atol)

    def any(self, array: np.ndarray) -> np.ndarray:
        return np.any(array)

    def arange(
        self, start: int, limit: int | None = None, delta: int = 1, dtype=np.float64
    ) -> np.ndarray:
        return np.arange(start, limit, delta, dtype=dtype)

    def asnumpy(self, tensor: np.ndarray) -> np.ndarray:
        if isinstance(tensor, np.ndarray):
            return tensor
        return np.array(tensor)

    def assign(self, tensor: np.ndarray, value: np.ndarray) -> np.ndarray:
        tensor = value
        return tensor

    def astensor(self, array: np.ndarray, dtype=None) -> np.ndarray:
        array = np.array(array)
        return self.cast(array, dtype=dtype or array.dtype)

    def atleast_1d(self, array: np.ndarray, dtype=None) -> np.ndarray:
        return np.atleast_1d(self.astensor(array, dtype))

    def atleast_2d(self, array: np.ndarray, dtype=None) -> np.ndarray:
        return np.atleast_2d(self.astensor(array, dtype))

    def atleast_3d(self, array: np.ndarray, dtype=None) -> np.ndarray:
        array = self.atleast_2d(self.atleast_1d(array))
        if len(array.shape) == 2:
            array = array[None, ...]
        return array

    def block(self, blocks: list[list[np.ndarray]], axes=(-2, -1)) -> np.ndarray:
        rows = [self.concat(row, axis=axes[1]) for row in blocks]
        return self.concat(rows, axis=axes[0])

    def block_diag(self, *blocks: list[np.ndarray]) -> np.ndarray:
        return sp.linalg.block_diag(*blocks)

    def boolean_mask(self, tensor: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return np.array([t for i, t in enumerate(tensor) if mask[i]])

    def cast(self, array: np.ndarray, dtype=None) -> np.ndarray:
        if dtype is None:
            return array
        if dtype not in [self.complex64, self.complex128, "complex64", "complex128"]:
            array = self.real(array)
        return np.array(array, dtype=dtype)

    def clip(self, array, a_min, a_max) -> np.ndarray:
        return np.clip(array, a_min, a_max)

    def concat(self, values: list[np.ndarray], axis: int) -> np.ndarray:
        # tf.concat can concatenate lists of scalars, while np.concatenate errors
        try:
            return np.concatenate(values, axis)
        except ValueError:
            return np.array(values)

    def conj(self, array: np.ndarray) -> np.ndarray:
        return np.conj(array)

    def cos(self, array: np.ndarray) -> np.ndarray:
        return np.cos(array)

    def cosh(self, array: np.ndarray) -> np.ndarray:
        return np.cosh(array)

    def det(self, matrix: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            det = np.linalg.det(matrix)
        return det

    def diag(self, array: np.ndarray, k: int = 0) -> np.ndarray:
        if len(array.shape) == 1:
            return np.diag(array, k=k)
        elif len(array.shape) == 2:
            return np.array([np.diag(l, k=k).tolist() for l in array])
        else:
            # fallback into more complex algorithm
            original_sh = array.shape

            ravelled_sh = (np.prod(original_sh[:-1]), original_sh[-1])
            array = array.ravel().reshape(*ravelled_sh)

            ret = []
            for line in array:
                ret.append(np.diag(line, k))

            ret = np.array(ret)
            inner_shape = (
                original_sh[-1] + abs(k),
                original_sh[-1] + abs(k),
            )
            return ret.reshape(original_sh[:-1] + inner_shape)

    def diag_part(self, array: np.ndarray, k: int) -> np.ndarray:
        ret = np.diagonal(array, offset=k, axis1=-2, axis2=-1)
        ret.flags.writeable = True
        return ret

    def set_diag(self, array: np.ndarray, diag: np.ndarray, k: int) -> np.ndarray:
        i = np.arange(0, array.shape[-2] - abs(k))
        if k < 0:
            i -= array.shape[-2] - abs(k)

        j = np.arange(abs(k), array.shape[-1])
        if k < 0:
            j -= abs(k)

        array[..., i, j] = diag

        return array

    def einsum(self, string: str, *tensors) -> np.ndarray | None:
        if type(string) is str:
            return np.einsum(string, *tensors)
        return None  # provide same functionality as numpy.einsum or upgrade to opt_einsum

    def exp(self, array: np.ndarray) -> np.ndarray:
        return np.exp(array)

    def expand_dims(self, array: np.ndarray, axis: int) -> np.ndarray:
        return np.expand_dims(array, axis)

    def expm(self, matrix: np.ndarray) -> np.ndarray:
        return scipy_expm(matrix)

    def eye(self, size: int, dtype=np.float64) -> np.ndarray:
        return np.eye(size, dtype=dtype)

    def eye_like(self, array: np.ndarray) -> np.ndarray:
        return np.eye(array.shape[-1], dtype=array.dtype)

    def from_backend(self, value) -> bool:
        return isinstance(value, np.ndarray)

    def gather(self, array: np.ndarray, indices: np.ndarray, axis: int = 0) -> np.ndarray:
        return np.take(array, indices, axis=axis)

    def imag(self, array: np.ndarray) -> np.ndarray:
        return np.imag(array)

    def inv(self, tensor: np.ndarray) -> np.ndarray:
        return np.linalg.inv(tensor)

    def is_trainable(self, tensor: np.ndarray) -> bool:  # pylint: disable=unused-argument
        return False

    def lgamma(self, x: np.ndarray) -> np.ndarray:
        return np.array([mlgamma(v) for v in x])

    def log(self, x: np.ndarray) -> np.ndarray:
        return np.log(x)

    def make_complex(self, real: np.ndarray, imag: np.ndarray) -> np.ndarray:
        return real + 1j * imag

    @Autocast()
    def matmul(self, *matrices: np.ndarray) -> np.ndarray:
        mat = matrices[0]
        for matrix in matrices[1:]:
            mat = np.matmul(mat, matrix)
        return mat

    @Autocast()
    def matvec(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self.matmul(a, b[:, None])[:, 0]

    @Autocast()
    def maximum(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.maximum(a, b)

    @Autocast()
    def minimum(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.minimum(a, b)

    def moveaxis(
        self, array: np.ndarray, old: int | Sequence[int], new: int | Sequence[int]
    ) -> np.ndarray:
        return np.moveaxis(array, old, new)

    def new_variable(
        self,
        value,
        bounds: tuple[float | None, float | None] | None,
        name: str,
        dtype=np.float64,
    ):  # pylint: disable=unused-argument
        return np.array(value, dtype=dtype)

    def new_constant(self, value, name: str, dtype=np.float64):  # pylint: disable=unused-argument
        return np.array(value, dtype=dtype)

    def norm(self, array: np.ndarray) -> np.ndarray:
        return np.linalg.norm(array)

    def ones(self, shape: Sequence[int], dtype=np.float64) -> np.ndarray:
        return np.ones(shape, dtype=dtype)

    def ones_like(self, array: np.ndarray) -> np.ndarray:
        return np.ones(array.shape, dtype=array.dtype)

    @Autocast()
    def outer(self, array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
        return np.tensordot(array1, array2, [[], []])

    def pad(
        self,
        array: np.ndarray,
        paddings: Sequence[tuple[int, int]],
        mode="CONSTANT",
        constant_values=0,
    ) -> np.ndarray:
        if mode == "CONSTANT":
            mode = "constant"
        return np.pad(array, paddings, mode, constant_values=constant_values)

    @staticmethod
    def pinv(matrix: np.ndarray) -> np.ndarray:
        return np.linalg.pinv(matrix)

    @Autocast()
    def pow(self, x: np.ndarray, y: float) -> np.ndarray:
        return np.power(x, y)

    def kron(self, tensor1: np.ndarray, tensor2: np.ndarray):
        return np.kron(tensor1, tensor2)

    def prod(self, x: np.ndarray, axis: int | None):
        return np.prod(x, axis=axis)

    def real(self, array: np.ndarray) -> np.ndarray:
        return np.real(array)

    def reshape(self, array: np.ndarray, shape: Sequence[int]) -> np.ndarray:
        return np.reshape(array, shape)

    def round(self, array: np.ndarray, decimals: int = 0) -> np.ndarray:
        return np.round(array, decimals)

    def sin(self, array: np.ndarray) -> np.ndarray:
        return np.sin(array)

    def sinh(self, array: np.ndarray) -> np.ndarray:
        return np.sinh(array)

    def solve(self, matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        if len(rhs.shape) == len(matrix.shape) - 1:
            rhs = np.expand_dims(rhs, -1)
            return np.linalg.solve(matrix, rhs)[..., 0]
        return np.linalg.solve(matrix, rhs)

    def sort(self, array: np.ndarray, axis: int = -1) -> np.ndarray:
        return np.sort(array, axis)

    def sqrt(self, x: np.ndarray, dtype=None) -> np.ndarray:
        return np.sqrt(self.cast(x, dtype))

    def sum(self, array: np.ndarray, axes: Sequence[int] = None):
        if axes is None:
            return np.sum(array)

        ret = array
        for axis in axes:
            ret = np.sum(ret, axis=axis)
        return ret

    @Autocast()
    def tensordot(self, a: np.ndarray, b: np.ndarray, axes: list[int]) -> np.ndarray:
        return np.tensordot(a, b, axes)

    def tile(self, array: np.ndarray, repeats: Sequence[int]) -> np.ndarray:
        return np.tile(array, repeats)

    def trace(self, array: np.ndarray, dtype=None) -> np.ndarray:
        return self.cast(np.trace(array, axis1=-1, axis2=-2), dtype)

    def transpose(self, a: np.ndarray, perm: Sequence[int] = None) -> np.ndarray | None:
        if a is None:
            return None  # TODO: remove and address None inputs where tranpose is used
        return np.transpose(a, axes=perm)

    @Autocast()
    def update_tensor(
        self, tensor: np.ndarray, indices: np.ndarray, values: np.ndarray
    ) -> np.ndarray:
        indices = self.atleast_2d(indices)
        for i, v in zip(indices, values):
            tensor[tuple(i)] = v
        return tensor

    @Autocast()
    def update_add_tensor(
        self, tensor: np.ndarray, indices: np.ndarray, values: np.ndarray
    ) -> np.ndarray:
        indices = self.atleast_2d(indices)
        for i, v in zip(indices, values):
            tensor[tuple(i)] += v
        return tensor

    def zeros(self, shape: Sequence[int], dtype=np.float64) -> np.ndarray:
        return np.zeros(shape, dtype=dtype)

    def zeros_like(self, array: np.ndarray) -> np.ndarray:
        return np.zeros(np.array(array).shape, dtype=array.dtype)

    def map_fn(self, func, elements):
        # Is this done like this?
        return np.array([func(e) for e in elements])

    def squeeze(self, tensor, axis=None):
        return np.squeeze(tensor, axis=axis)

    def cholesky(self, input: np.ndarray):
        return np.linalg.cholesky(input)

    def Categorical(self, probs: np.ndarray, name: str):  # pylint: disable=unused-argument
        class Generator:
            def __init__(self, probs):
                self._probs = probs

            def sample(self):
                idx = [i for i, _ in enumerate(probs)]
                return np.random.choice(idx, p=probs / sum(probs))

        return Generator(probs)

    def MultivariateNormalTriL(self, loc: np.ndarray, scale_tril: np.ndarray):
        class Generator:
            def __init__(self, mean, cov):
                self._mean = mean
                self._cov = cov

            def sample(self, dtype=None):  # pylint: disable=unused-argument
                fn = np.random.default_rng().multivariate_normal
                ret = fn(self._mean, self._cov)
                return ret

            def prob(self, x):
                return multivariate_normal.pdf(x, mean=self._mean, cov=self._cov)

        scale_tril = scale_tril @ np.transpose(scale_tril)
        return Generator(loc, scale_tril)

    @staticmethod
    def eigvals(tensor: np.ndarray) -> np.ndarray:
        return np.linalg.eigvals(tensor)

    @staticmethod
    def xlogy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return scipy_xlogy(x, y)

    @staticmethod
    def eigh(tensor: np.ndarray) -> tuple:
        return np.linalg.eigh(tensor)

    def sqrtm(self, tensor: np.ndarray, dtype, rtol=1e-05, atol=1e-08) -> np.ndarray:
        if np.allclose(tensor, 0, rtol=rtol, atol=atol):
            ret = self.zeros_like(tensor)
        else:
            ret = scipy_sqrtm(tensor)

        if dtype is None:
            return self.cast(ret, self.complex128)
        return self.cast(ret, dtype)

    # ~~~~~~~~~~~~~~~~~
    # Special functions
    # ~~~~~~~~~~~~~~~~~

    @staticmethod
    def DefaultEuclideanOptimizer() -> None:
        return None

    def hermite_renormalized(
        self, A: np.ndarray, B: np.ndarray, C: np.ndarray, shape: tuple[int]
    ) -> np.ndarray:
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

        precision_bits = settings.PRECISION_BITS_HERMITE_POLY

        if precision_bits == 128:  # numba
            if settings.USE_VANILLA_AVERAGE:
                G = vanilla_average(tuple(shape), A, B, C)
            else:
                G = vanilla(tuple(shape), A, B, C)
        else:  # julia (with precision_bits = 512)
            # The following import must come after running "jl = Julia(compiled_modules=False)" in settings.py
            from juliacall import Main as jl  # pylint: disable=import-outside-toplevel

            A, B, C = (
                np.array(A).astype(np.complex128),
                np.array(B).astype(np.complex128),
                np.array(C).astype(np.complex128),
            )
            G = jl.Vanilla.vanilla(A, B, C.item(), np.array(shape, dtype=np.int64), precision_bits)

        return G

    def hermite_renormalized_batch(
        self, A: np.ndarray, B: np.ndarray, C: np.ndarray, shape: tuple[int]
    ) -> np.ndarray:
        if settings.USE_VANILLA_AVERAGE:
            G = vanilla_average(tuple(shape), A, B, C)
        else:
            G = vanilla_batch(tuple(shape), A, B, C)
        return G

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
        G, _ = binomial(
            tuple(shape),
            A,
            B,
            C,
            max_l2=max_l2 or settings.AUTOSHAPE_PROBABILITY,
            global_cutoff=global_cutoff or sum(shape) - len(shape) + 1,
        )

        return G

    def reorder_AB_bargmann(self, A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r"""In mrmustard.math.numba.compactFock~ dimensions of the Fock representation are ordered like [mode0,mode0,mode1,mode1,...]
        while in mrmustard.physics.bargmann the ordering is [mode0,mode1,...,mode0,mode1,...]. Here we reorder A and B.
        """
        ordering = np.arange(2 * A.shape[0] // 2).reshape(2, -1).T.flatten()
        A = self.gather(A, ordering, axis=1)
        A = self.gather(A, ordering)
        B = self.gather(B, ordering, axis=0)
        return A, B

    def hermite_renormalized_diagonal(
        self, A: np.ndarray, B: np.ndarray, C: np.ndarray, cutoffs: tuple[int]
    ) -> np.ndarray:
        r"""First, reorder A and B parameters of Bargmann representation to match conventions in mrmustard.math.numba.compactFock~
        Then, calculate the required renormalized multidimensional Hermite polynomial.
        """
        A, B = self.reorder_AB_bargmann(A, B)
        return self.hermite_renormalized_diagonal_reorderedAB(A, B, C, cutoffs=cutoffs)

    def hermite_renormalized_diagonal_reorderedAB(
        self, A: np.ndarray, B: np.ndarray, C: np.ndarray, cutoffs: tuple[int]
    ) -> np.ndarray:
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
        poly0, _, _, _, _ = hermite_multidimensional_diagonal(A, B, C, cutoffs)

        return poly0

    def hermite_renormalized_diagonal_batch(
        self, A: np.ndarray, B: np.ndarray, C: np.ndarray, cutoffs: tuple[int]
    ) -> np.ndarray:
        r"""Same as hermite_renormalized_diagonal but works for a batch of different B's."""
        A, B = self.reorder_AB_bargmann(A, B)
        return self.hermite_renormalized_diagonal_reorderedAB_batch(A, B, C, cutoffs=cutoffs)

    def hermite_renormalized_diagonal_reorderedAB_batch(
        self, A: np.ndarray, B: np.ndarray, C: np.ndarray, cutoffs: tuple[int]
    ) -> np.ndarray:
        r"""Same as hermite_renormalized_diagonal_reorderedAB but works for a batch of different B's.

        Args:
            A: The A matrix.
            B: The B vectors.
            C: The C scalar.
            cutoffs: upper boundary of photon numbers in each mode

        Returns:
            The renormalized Hermite polynomial from different B values.
        """
        poly0, _, _, _, _ = hermite_multidimensional_diagonal_batch(A, B, C, cutoffs)

        return poly0

    def hermite_renormalized_1leftoverMode(
        self, A: np.ndarray, B: np.ndarray, C: np.ndarray, cutoffs: tuple[int]
    ) -> np.ndarray:
        r"""First, reorder A and B parameters of Bargmann representation to match conventions in mrmustard.math.numba.compactFock~
        Then, calculate the required renormalized multidimensional Hermite polynomial.
        """
        A, B = self.reorder_AB_bargmann(A, B)
        return self.hermite_renormalized_1leftoverMode_reorderedAB(A, B, C, cutoffs=cutoffs)

    def hermite_renormalized_1leftoverMode_reorderedAB(
        self, A: np.ndarray, B: np.ndarray, C: np.ndarray, cutoffs: tuple[int]
    ) -> np.ndarray:
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
        poly0, _, _, _, _ = hermite_multidimensional_1leftoverMode(A, B, C, cutoffs)
        return poly0

    @staticmethod
    def getitem(tensor, *, key):
        value = np.array(tensor)[key]
        return value

    @staticmethod
    def setitem(tensor, value, *, key):
        _tensor = np.array(tensor)
        value = np.array(value)
        _tensor[key] = value

        return _tensor
