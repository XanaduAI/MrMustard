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

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy.linalg import expm as scipy_expm
from scipy.linalg import sqrtm as scipy_sqrtm
from scipy.special import loggamma as scipy_loggamma
from scipy.special import xlogy as scipy_xlogy

from ..utils.settings import settings
from .backend_base import BackendBase
from .lattice import strategies
from .lattice.strategies.compactFock.inputValidation import (
    hermite_multidimensional_diagonal,
)

np.set_printoptions(legacy="1.25")


class BackendNumpy(BackendBase):
    r"""
    A numpy backend.
    """

    int32 = np.int32
    int64 = np.int64
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

    def all(self, array: np.ndarray) -> bool:
        return np.all(array)

    def allclose(self, array1: np.array, array2: np.array, atol: float, rtol: float) -> bool:
        return np.allclose(array1, array2, atol=atol, rtol=rtol)

    def angle(self, array: np.ndarray) -> np.ndarray:
        return np.angle(array)

    def any(self, array: np.ndarray) -> np.ndarray:
        return np.any(array)

    def arange(
        self,
        start: int,
        limit: int | None = None,
        delta: int = 1,
        dtype=np.float64,
    ) -> np.ndarray:
        return np.arange(start, limit, delta, dtype=dtype)

    def asnumpy(self, tensor: np.ndarray) -> np.ndarray:
        return np.asarray(tensor)

    def astensor(self, array: np.ndarray, dtype=None) -> np.ndarray:
        return np.asarray(array, dtype=dtype)

    def atleast_nd(self, array: np.ndarray, n: int, dtype=None) -> np.ndarray:
        return np.array(array, ndmin=n, dtype=dtype)

    def broadcast_to(self, array: np.ndarray, shape: tuple[int]) -> np.ndarray:
        return np.broadcast_to(array, shape)

    def broadcast_arrays(self, *arrays: list[np.ndarray]) -> list[np.ndarray]:
        return np.broadcast_arrays(*arrays)

    def cast(self, array: np.ndarray, dtype=None) -> np.ndarray:
        if dtype is None:
            return array
        if dtype not in [self.complex64, self.complex128, "complex64", "complex128"]:
            array = self.real(array)
        return np.asarray(array, dtype=dtype)

    def clip(self, array, a_min, a_max) -> np.ndarray:
        return np.clip(array, a_min, a_max)

    def concat(self, values: list[np.ndarray], axis: int) -> np.ndarray:
        # tf.concat can concatenate lists of scalars, while np.concatenate errors
        try:
            return np.concatenate(values, axis)
        except ValueError:
            return np.asarray(values)

    def conj(self, array: np.ndarray) -> np.ndarray:
        return np.conj(array)

    def cos(self, array: np.ndarray) -> np.ndarray:
        return np.cos(array)

    def cosh(self, array: np.ndarray) -> np.ndarray:
        return np.cosh(array)

    def det(self, matrix: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            det = np.linalg.det(matrix)
        return det  # noqa: RET504

    def diag(self, array: np.ndarray, k: int = 0) -> np.ndarray:
        if array.ndim in (1, 2):
            return np.diag(array, k=k)
        # fallback into more complex algorithm
        original_sh = array.shape

        ravelled_sh = (np.prod(original_sh[:-1]), original_sh[-1])
        array = array.ravel().reshape(*ravelled_sh)

        ret = np.asarray([np.diag(line, k) for line in array])
        inner_shape = (
            original_sh[-1] + abs(k),
            original_sh[-1] + abs(k),
        )
        return ret.reshape(original_sh[:-1] + inner_shape)

    def diag_part(self, array: np.ndarray, k: int) -> np.ndarray:
        ret = np.diagonal(array, offset=k, axis1=-2, axis2=-1)
        ret.flags.writeable = True
        return ret

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

    def equal(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.equal(a, b)

    def gather(self, array: np.ndarray, indices: np.ndarray, axis: int = 0) -> np.ndarray:
        return np.take(array, indices, axis=axis)

    def imag(self, array: np.ndarray) -> np.ndarray:
        return np.imag(array)

    def inv(self, tensor: np.ndarray) -> np.ndarray:
        return np.linalg.inv(tensor)

    def iscomplexobj(self, x: Any) -> bool:
        return np.iscomplexobj(x)

    def isnan(self, array: np.ndarray) -> np.ndarray:
        return np.isnan(array)

    def issubdtype(self, arg1, arg2) -> bool:
        return np.issubdtype(arg1, arg2)

    def lgamma(self, x: np.ndarray) -> np.ndarray:
        return scipy_loggamma(x)

    def log(self, x: np.ndarray) -> np.ndarray:
        return np.log(x)

    def make_complex(self, real: np.ndarray, imag: np.ndarray) -> np.ndarray:
        return real + 1j * imag

    def matmul(self, *matrices: np.ndarray) -> np.ndarray:
        mat = matrices[0]
        for matrix in matrices[1:]:
            mat = np.matmul(mat, matrix)
        return mat

    def matvec(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return self.matmul(a, b[..., None])[..., 0]

    def max(self, array: np.ndarray) -> np.ndarray:
        return np.max(array)

    def maximum(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.maximum(a, b)

    def minimum(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.minimum(a, b)

    def mod(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.mod(a, b)

    def moveaxis(
        self,
        array: np.ndarray,
        old: int | Sequence[int],
        new: int | Sequence[int],
    ) -> np.ndarray:
        return np.moveaxis(array, old, new)

    def norm(self, array: np.ndarray) -> np.ndarray:
        return np.linalg.norm(array)

    def ones(self, shape: Sequence[int], dtype=np.float64) -> np.ndarray:
        return np.ones(shape, dtype=dtype)

    def ones_like(self, array: np.ndarray) -> np.ndarray:
        return np.ones(array.shape, dtype=array.dtype)

    def infinity_like(self, array: np.ndarray) -> np.ndarray:
        return np.full_like(array, np.inf)

    def conditional(
        self,
        cond: np.ndarray,
        true_fn: Callable,
        false_fn: Callable,
        *args,
    ) -> np.ndarray:
        if cond.all():
            return true_fn(*args)
        return false_fn(*args)

    def error_if(self, array: np.ndarray, condition: np.ndarray, msg: str):
        if np.any(condition):
            raise ValueError(msg)

    def outer(self, array1: np.ndarray, array2: np.ndarray) -> np.ndarray:
        return self.tensordot(array1, array2, [[], []])

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

    def stack(self, arrays: np.ndarray, axis: int = 0) -> np.ndarray:
        return np.stack(arrays, axis=axis)

    def sum(self, array: np.ndarray, axis: int | tuple[int] | None = None):
        return np.sum(array, axis=axis)

    def swapaxes(self, array: np.ndarray, axis1: int, axis2: int) -> np.ndarray:
        return np.swapaxes(array, axis1, axis2)

    def tensordot(self, a: np.ndarray, b: np.ndarray, axes: list[int]) -> np.ndarray:
        return np.tensordot(a, b, axes)

    def tile(self, array: np.ndarray, repeats: Sequence[int]) -> np.ndarray:
        return np.tile(array, repeats)

    def trace(self, array: np.ndarray, dtype=None) -> np.ndarray:
        return self.cast(np.trace(array, axis1=-1, axis2=-2), dtype)

    def transpose(self, a: np.ndarray, perm: Sequence[int] | None = None) -> np.ndarray | None:
        return np.transpose(a, axes=perm)

    def update_tensor(
        self,
        tensor: np.ndarray,
        indices: np.ndarray,
        values: np.ndarray,
    ) -> np.ndarray:
        tensor[indices] = values
        return tensor

    def update_add_tensor(
        self,
        tensor: np.ndarray,
        indices: np.ndarray,
        values: np.ndarray,
    ) -> np.ndarray:
        indices = self.atleast_nd(indices, 2)
        for i, v in zip(indices, values):
            tensor[tuple(i)] += v
        return tensor

    def zeros(self, shape: Sequence[int], dtype=np.float64) -> np.ndarray:
        return np.zeros(shape, dtype=dtype)

    def zeros_like(self, array: np.ndarray) -> np.ndarray:
        return np.zeros_like(array, dtype=array.dtype)

    def map_fn(self, func, elements):
        # Is this done like this?
        return np.asarray([func(e) for e in elements])

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

    def reorder_AB_bargmann(self, A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r"""In mrmustard.math.numba.compactFock~ dimensions of the Fock representation are ordered like [mode0,mode0,mode1,mode1,...]
        while in mrmustard.physics.bargmann_utils the ordering is [mode0,mode1,...,mode0,mode1,...]. Here we reorder A and B.
        """
        ordering = np.arange(2 * A.shape[0] // 2).reshape(2, -1).T.flatten()
        A = self.gather(A, ordering, axis=1)
        A = self.gather(A, ordering)
        B = self.gather(B, ordering, axis=0)
        return A, B

    # ~~~~~~~~~~~~~~~~~~~~
    # hermite_renormalized
    # ~~~~~~~~~~~~~~~~~~~~

    def hermite_renormalized(
        self,
        A: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        shape: tuple[int],
        stable: bool = False,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        if stable:
            return strategies.stable_numba(tuple(shape), A, b, c, out)
        return strategies.vanilla_numba(tuple(shape), A, b, c, out)

    def hermite_renormalized_batched(
        self,
        A: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        shape: tuple[int],
        stable: bool = False,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        return strategies.vanilla_batch_numba(tuple(shape), A, b, c, stable, out)

    def hermite_renormalized_binomial(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        shape: tuple[int],
        max_l2: float | None,
        global_cutoff: int | None,
    ) -> np.ndarray:
        return strategies.binomial(
            tuple(shape),
            A,
            B,
            C,
            max_l2=max_l2 or settings.AUTOSHAPE_PROBABILITY,
            global_cutoff=global_cutoff or sum(shape) - len(shape) + 1,
        )[0]

    def hermite_renormalized_diagonal(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        cutoffs: tuple[int],
        reorderedAB: bool,
    ) -> np.ndarray:
        A, B = self.reorder_AB_bargmann(A, B) if reorderedAB else (A, B)
        return hermite_multidimensional_diagonal(A, B, C, cutoffs)[0]

    def hermite_renormalized_1leftoverMode(
        self,
        A: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        output_cutoff: int,
        pnr_cutoffs: tuple[int, ...],
        stable: bool = False,
        reorderedAB: bool = True,
    ) -> np.ndarray:
        # why is this different from the jax version?
        return strategies.fast_diagonal(A, b, c, output_cutoff, pnr_cutoffs, stable).transpose(
            (-2, -1, *tuple(range(len(pnr_cutoffs)))),
        )

    # ~~~~~~~~~~~~~~~~~~~~~~~
    # Fock lattice strategies
    # ~~~~~~~~~~~~~~~~~~~~~~~

    def displacement(self, x: float, y: float, shape: tuple[int, int], tol: float):
        alpha = self.asnumpy(x) + 1j * self.asnumpy(y)
        if np.sqrt(x * x + y * y) > tol:
            gate = strategies.displacement(tuple(shape), alpha)
        else:
            gate = self.eye(max(shape), dtype="complex128")[: shape[0], : shape[1]]
        return self.astensor(gate, dtype=gate.dtype.name)

    def beamsplitter(self, theta: float, phi: float, shape: tuple[int, int, int, int], method: str):
        t, s = self.asnumpy(theta), self.asnumpy(phi)
        if method == "vanilla":
            bs_unitary = strategies.beamsplitter(shape, t, s)
        elif method == "schwinger":
            bs_unitary = strategies.beamsplitter_schwinger(shape, t, s)
        elif method == "stable":
            bs_unitary = strategies.stable_beamsplitter(shape, t, s)
        return self.astensor(bs_unitary, dtype=bs_unitary.dtype.name)

    def squeezed(self, r: float, phi: float, shape: tuple[int, int]):
        sq_ket = strategies.squeezed(shape, self.asnumpy(r), self.asnumpy(phi))
        return self.astensor(sq_ket, dtype=sq_ket.dtype.name)

    def squeezer(self, r: float, phi: float, shape: tuple[int, int]):
        sq_ket = strategies.squeezer(shape, self.asnumpy(r), self.asnumpy(phi))
        return self.astensor(sq_ket, dtype=sq_ket.dtype.name)
