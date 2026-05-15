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

from mrmustard import settings
from mrmustard.mathlib import cython_lattice
from mrmustard.mathlib.gaussian_integrals import (  # numba or guvectorized functions
    complex_gaussian_integral_1_guvectorized,
    complex_gaussian_integral_1_jitted,
    complex_gaussian_integral_2_guvectorized,
    complex_gaussian_integral_2_jitted,
)
from mrmustard.mathlib.lattice import strategies
from mrmustard.mathlib.lattice.strategies.compactFock.inputValidation import (
    hermite_multidimensional_diagonal,
)

from .backend_base import BackendBase

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

    def argmax(self, array: np.ndarray, axis: int | None = None) -> np.ndarray:
        return np.argmax(array, axis=axis)

    def argmin(self, array: np.ndarray, axis: int | None = None) -> np.ndarray:
        return np.argmin(array, axis=axis)

    def argsort(self, array: np.ndarray, axis: int | None = None) -> np.ndarray:
        return np.argsort(array, axis=axis)

    def asnumpy(self, tensor: np.ndarray) -> np.ndarray:
        return np.asarray(tensor)

    def astensor(self, array: np.ndarray, dtype=None) -> np.ndarray:
        return np.asarray(array, dtype=dtype)

    def atleast_nd(self, array: np.ndarray, n: int, dtype=None) -> np.ndarray:
        return np.array(array, ndmin=n, dtype=dtype)

    def BackendError(self):
        # no numpy backend specific errors
        raise NotImplementedError

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

    def complex_gaussian_integral_1_single(self, A, b, idx12, A_out, b_out, log_c_out):
        if A_out is None:  # assume all None
            m = A.shape[-2] - len(idx12)
            A_out = np.empty((m, m), dtype=A.dtype)
            b_out = np.empty((m,), dtype=A.dtype)
            log_c_out = np.empty((1,), dtype=A.dtype)

        complex_gaussian_integral_1_jitted(A, b, idx12, A_out, b_out, log_c_out)

        return A_out, b_out, log_c_out[..., 0]

    def complex_gaussian_integral_1_batched(self, A, b, idx12, A_out, b_out, log_c_out):
        batch_A = A.shape[:-2]
        batch_b = b.shape[:-1]
        target_batch = np.broadcast_shapes(batch_A, batch_b)

        if A_out is None:  # assume all None
            m = A.shape[-1] - len(idx12)
            A_out = np.empty((*target_batch, m, m), dtype=A.dtype)
            b_out = np.empty((*target_batch, m), dtype=A.dtype)
            log_c_out = np.empty((*target_batch, 1), dtype=A.dtype)
        complex_gaussian_integral_1_guvectorized(A, b, idx12, A_out, b_out, log_c_out)

        return A_out, b_out, log_c_out[..., 0]

    def complex_gaussian_integral_2_single(
        self, A1, b1, A2, b2, idx1, idx2, A_out, b_out, log_c_out
    ):
        if A_out is None:  # assume all None
            m = A1.shape[-2] + A2.shape[-2] - 2 * len(idx1)
            A_out = np.empty((m, m), dtype=A1.dtype)
            b_out = np.empty((m,), dtype=A1.dtype)
            log_c_out = np.empty((1,), dtype=A1.dtype)

        complex_gaussian_integral_2_jitted(A1, b1, A2, b2, idx1, idx2, A_out, b_out, log_c_out)

        return A_out, b_out, log_c_out[..., 0]

    def complex_gaussian_integral_2_batched(
        self, A1, b1, A2, b2, idx1, idx2, A_out, b_out, log_c_out
    ):
        batch_A1, batch_b1 = A1.shape[:-2], b1.shape[:-1]
        batch_A2, batch_b2 = A2.shape[:-2], b2.shape[:-1]
        target_batch = np.broadcast_shapes(
            np.broadcast_shapes(batch_A1, batch_b1), np.broadcast_shapes(batch_A2, batch_b2)
        )

        if A_out is None:  # assume all None
            output_size = A1.shape[-1] + A2.shape[-1] - 2 * len(idx1)
            A_out = np.empty((*target_batch, output_size, output_size), dtype=A1.dtype)
            b_out = np.empty((*target_batch, output_size), dtype=A1.dtype)
            log_c_out = np.empty((*target_batch, 1), dtype=A1.dtype)

        complex_gaussian_integral_2_guvectorized(
            A1, b1, A2, b2, idx1, idx2, A_out, b_out, log_c_out
        )
        return A_out, b_out, log_c_out[..., 0]

    def concat(self, values: list[np.ndarray], axis: int) -> np.ndarray:
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

    def diagonal(
        self, array: np.ndarray, offset: int | None, axis1: int | None, axis2: int | None
    ) -> np.ndarray:
        return np.diagonal(array, offset=offset, axis1=axis1, axis2=axis2)

    def diag(self, array: np.ndarray, k: int = 0) -> np.ndarray:
        return np.diag(array, k=k)

    def exp(self, array: np.ndarray) -> np.ndarray:
        return np.exp(array)

    def expand_dims(self, array: np.ndarray, axis: int) -> np.ndarray:
        return np.expand_dims(array, axis)

    def squeeze(self, array: np.ndarray, axis: int | tuple[int, ...] | None = None) -> np.ndarray:
        return np.squeeze(array, axis=axis)

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
        with np.errstate(divide="ignore"):
            return np.log(x)

    def make_complex(self, real: np.ndarray, imag: np.ndarray) -> np.ndarray:
        return real + 1j * imag

    def matmul(self, *matrices: np.ndarray) -> np.ndarray:
        use_matmul = self.any(matrix.ndim > 2 for matrix in matrices)
        if use_matmul:
            mat = matrices[0]
            for matrix in matrices[1:]:
                mat = np.matmul(mat, matrix)
        else:
            mat = np.linalg.multi_dot(matrices)
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

    def mean(self, array: np.ndarray, axis: int | tuple[int] | None = None) -> np.ndarray:
        return np.mean(array, axis=axis)

    def norm(
        self, array: np.ndarray, axis: int | tuple[int, int] | None = None, keepdims: bool = False
    ) -> np.ndarray:
        return np.linalg.norm(array, axis=axis, keepdims=keepdims)

    def ones(self, shape: Sequence[int], dtype=np.float64) -> np.ndarray:
        return np.ones(shape, dtype=dtype)

    def full(self, shape: Sequence[int], fill_value, dtype=None) -> np.ndarray:
        dtype = dtype or np.result_type(fill_value)
        return np.full(shape, fill_value, dtype=dtype)

    def ones_like(self, array: np.ndarray) -> np.ndarray:
        return np.ones(array.shape, dtype=array.dtype)

    def infinity_like(self, array: np.ndarray) -> np.ndarray:
        return np.full_like(array, np.inf)

    def conditional(
        self,
        cond: np.ndarray | bool,
        true_fn: Callable,
        false_fn: Callable,
        *args,
    ) -> np.ndarray:
        if self.asnumpy(cond).all():
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

    def shape(self, array: np.ndarray) -> tuple[int, ...]:
        return np.shape(array)

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

    def tan(self, array: np.ndarray) -> np.ndarray:
        return np.tan(array)

    def tanh(self, array: np.ndarray) -> np.ndarray:
        return np.tanh(array)

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
        return cython_lattice.vanilla(tuple(shape), A, b, c, stable, out)

    def hermite_renormalized_batched(
        self,
        A: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        shape: tuple[int],
        stable: bool = False,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        return cython_lattice.vanilla_batched(tuple(shape), A, b, c, stable, out)

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
        return strategies.fast_diagonal(A, b, c, output_cutoff, pnr_cutoffs, stable).transpose(
            (-2, -1, *tuple(range(len(pnr_cutoffs)))),
        )

    # ~~~~~~~~~~~~~~~~~~~~~~~
    # Fock lattice strategies
    # ~~~~~~~~~~~~~~~~~~~~~~~

    def displacement(self, alpha: complex | np.ndarray, shape: tuple[int, int]):
        alpha = self.astensor(alpha, dtype=self.complex128)
        batch_shape = alpha.shape
        if batch_shape == ():
            return cython_lattice.displacement(tuple(shape), alpha)
        alpha_flattened = alpha.reshape(-1)
        ret = cython_lattice.displacement_batched(tuple(shape), alpha_flattened)
        return ret.reshape((*batch_shape, *shape))

    def beamsplitter(
        self,
        theta: float | np.ndarray,
        phi: float | np.ndarray,
        shape: tuple[int, int, int, int],
        method: str,
    ):
        theta, phi = (
            self.astensor(theta, dtype=self.float64),
            self.astensor(phi, dtype=self.float64),
        )
        if method == "schwinger":
            return strategies.beamsplitter_schwinger(shape, theta, phi)
        stable = method == "stable"
        batch_shape = theta.shape
        if batch_shape == ():
            return cython_lattice.beamsplitter(shape, theta, phi, stable=stable)
        theta_flattened = theta.reshape(-1)
        phi_flattened = phi.reshape(-1)
        bs_unitary = cython_lattice.beamsplitter_batched(
            shape, theta_flattened, phi_flattened, stable=stable
        )
        return bs_unitary.reshape((*batch_shape, *shape))

    def homodyne_projector(
        self,
        fock_dim: int,
        A: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        out: np.ndarray | None,
    ):
        A, b, c = (
            self.astensor(A, dtype=self.complex128),
            self.astensor(b, dtype=self.complex128),
            self.astensor(c, dtype=self.complex128),
        )
        batch_shape = A.shape[:-2]
        if batch_shape == ():
            return cython_lattice.homodyne_projector(fock_dim, A, b, c, out=out)
        A_flattened = A.reshape(-1, *A.shape[-2:])
        b_flattened = b.reshape(-1, *b.shape[-1:])
        c_flattened = c.reshape(-1)
        out_flattened = out.reshape(-1, *out.shape[-3:]) if out is not None else None
        ret = cython_lattice.homodyne_projector_batched(
            fock_dim, A_flattened, b_flattened, c_flattened, out=out_flattened
        )
        return ret.reshape((*batch_shape, fock_dim, fock_dim, fock_dim))

    def squeezed(self, r: float, phi: float, shape: tuple[int]):
        r, phi = self.astensor(r, dtype=self.float64), self.astensor(phi, dtype=self.float64)
        batch_shape = r.shape
        if batch_shape == ():
            return cython_lattice.squeezed(int(shape[0]), r, phi)
        r_flattened = r.reshape(-1)
        phi_flattened = phi.reshape(-1)
        ret = cython_lattice.squeezed_batched(int(shape[0]), r_flattened, phi_flattened)
        return ret.reshape((*batch_shape, shape[0]))

    def squeezer(self, r: float, phi: float, shape: tuple[int, int]):
        r, phi = self.astensor(r, dtype=self.float64), self.astensor(phi, dtype=self.float64)
        batch_shape = r.shape
        if batch_shape == ():
            return cython_lattice.squeezer(shape, r, phi)
        r_flattened = r.reshape(-1)
        phi_flattened = phi.reshape(-1)
        ret = cython_lattice.squeezer_batched(shape, r_flattened, phi_flattened)
        return ret.reshape((*batch_shape, *shape))
