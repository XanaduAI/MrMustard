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

import importlib

import numpy as np
from scipy.linalg import expm as scipy_expm
from scipy.linalg import sqrtm as scipy_sqrtm
from scipy.special import loggamma as scipy_loggamma
from scipy.special import xlogy as scipy_xlogy

from mrmustard import settings
from mrmustard.mathlib import cython_lattice
from mrmustard.mathlib.lattice import strategies
from mrmustard.mathlib.lattice.strategies.compactFock.inputValidation import (
    hermite_multidimensional_1leftoverMode,
    hermite_multidimensional_diagonal,
)

from .backend_base import BackendBase

np.set_printoptions(legacy="1.25")


def _gaussian_integral_fn(name: str):
    r"""Return the named function from the gaussian_integrals module, loading it on first use."""
    module = importlib.import_module("mrmustard.mathlib.gaussian_integrals")
    return getattr(module, name)


class BackendNumpy(BackendBase):
    r"""A numpy backend."""

    int32 = np.int32
    int64 = np.int64
    float32 = np.float32
    float64 = np.float64
    complex64 = np.complex64
    complex128 = np.complex128

    def __init__(self):
        super().__init__(name="numpy")

    def __repr__(self):
        return "BackendNumpy()"

    def abs(self, array):
        return np.abs(array)

    def all(self, array):
        return np.all(array)

    def allclose(self, array1, array2, atol, rtol):
        return np.allclose(array1, array2, atol=atol, rtol=rtol)

    def angle(self, array):
        return np.angle(array)

    def any(self, array):
        return np.any(array)

    def arange(self, start, limit, delta, dtype=None):
        return np.arange(start, limit, delta, dtype=dtype)

    def argmax(self, array, axis):
        return np.argmax(array, axis=axis)

    def argmin(self, array, axis):
        return np.argmin(array, axis=axis)

    def argsort(self, array, axis):
        return np.argsort(array, axis=axis)

    def asnumpy(self, tensor):
        return np.asarray(tensor)

    def astensor(self, array, dtype=None):
        return np.asarray(array, dtype=dtype)

    def atleast_nd(self, array, n, dtype=None):
        return np.array(array, ndmin=n, dtype=dtype)

    def BackendError(self):
        # no numpy backend specific errors
        raise NotImplementedError

    def broadcast_to(self, array, shape):
        return np.broadcast_to(array, shape)

    def broadcast_arrays(self, *arrays):
        return np.broadcast_arrays(*arrays)

    def cast(self, array, dtype=None):
        if dtype is None:
            return array
        if dtype not in [self.complex64, self.complex128, "complex64", "complex128"]:
            array = self.real(array)
        return np.asarray(array, dtype=dtype)

    def clip(self, array, a_min, a_max):
        return np.clip(array, a_min, a_max)

    def complex_gaussian_integral_1_single(self, A, b, idx12, A_out, b_out, log_c_out):
        if A_out is None:  # assume all None
            m = A.shape[-2] - len(idx12)
            A_out = np.empty((m, m), dtype=A.dtype)
            b_out = np.empty((m,), dtype=A.dtype)
            log_c_out = np.empty((1,), dtype=A.dtype)

        _gaussian_integral_fn("complex_gaussian_integral_1_jitted")(
            A, b, idx12, A_out, b_out, log_c_out
        )

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
        _gaussian_integral_fn("complex_gaussian_integral_1_guvectorized")(
            A, b, idx12, A_out, b_out, log_c_out
        )

        return A_out, b_out, log_c_out[..., 0]

    def complex_gaussian_integral_2_single(
        self, A1, b1, A2, b2, idx1, idx2, A_out, b_out, log_c_out
    ):
        if A_out is None:  # assume all None
            m = A1.shape[-2] + A2.shape[-2] - 2 * len(idx1)
            A_out = np.empty((m, m), dtype=A1.dtype)
            b_out = np.empty((m,), dtype=A1.dtype)
            log_c_out = np.empty((1,), dtype=A1.dtype)

        _gaussian_integral_fn("complex_gaussian_integral_2_jitted")(
            A1, b1, A2, b2, idx1, idx2, A_out, b_out, log_c_out
        )

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

        _gaussian_integral_fn("complex_gaussian_integral_2_guvectorized")(
            A1, b1, A2, b2, idx1, idx2, A_out, b_out, log_c_out
        )
        return A_out, b_out, log_c_out[..., 0]

    def concat(self, values, axis):
        try:
            return np.concatenate(values, axis)
        except ValueError:
            return np.asarray(values)

    def conj(self, array):
        return np.conj(array)

    def cos(self, array):
        return np.cos(array)

    def cosh(self, array):
        return np.cosh(array)

    def det(self, matrix):
        with np.errstate(divide="ignore", invalid="ignore"):
            det = np.linalg.det(matrix)
        return det  # noqa: RET504

    def diagonal(self, array, offset, axis1, axis2):
        return np.diagonal(array, offset=offset, axis1=axis1, axis2=axis2)

    def diag(self, array, k=0):
        return np.diag(array, k=k)

    def exp(self, array):
        return np.exp(array)

    def expand_dims(self, array, axis):
        return np.expand_dims(array, axis)

    def squeeze(self, array, axis=None):
        return np.squeeze(array, axis=axis)

    def expm(self, matrix):
        return scipy_expm(matrix)

    def eye(self, size, dtype=None):
        return np.eye(size, dtype=dtype)

    def eye_like(self, array):
        return np.eye(array.shape[-1], dtype=array.dtype)

    def equal(self, a, b):
        return np.equal(a, b)

    def gather(self, array, indices, axis=0):
        return np.take(array, indices, axis=axis)

    def imag(self, array):
        return np.imag(array)

    def inv(self, tensor):
        return np.linalg.inv(tensor)

    def iscomplexobj(self, x):
        return np.iscomplexobj(x)

    def isnan(self, array):
        return np.isnan(array)

    def issubdtype(self, arg1, arg2):
        return np.issubdtype(arg1, arg2)

    def lgamma(self, x):
        return scipy_loggamma(x)

    def log(self, x):
        with np.errstate(divide="ignore"):
            return np.log(x)

    def make_complex(self, real, imag):
        return real + 1j * imag

    def matmul(self, *matrices):
        use_matmul = self.any(matrix.ndim > 2 for matrix in matrices)
        if use_matmul:
            mat = matrices[0]
            for matrix in matrices[1:]:
                mat = np.matmul(mat, matrix)
        else:
            mat = np.linalg.multi_dot(matrices)
        return mat

    def matvec(self, a, b):
        return self.matmul(a, b[..., None])[..., 0]

    def max(self, array):
        return np.max(array)

    def maximum(self, a, b):
        return np.maximum(a, b)

    def minimum(self, a, b):
        return np.minimum(a, b)

    def mod(self, a, b):
        return np.mod(a, b)

    def moveaxis(self, array, old, new):
        return np.moveaxis(array, old, new)

    def mean(self, array, axis=None):
        return np.mean(array, axis=axis)

    def norm(self, array, axis=None, keepdims=False):
        return np.linalg.norm(array, axis=axis, keepdims=keepdims)

    def ones(self, shape, dtype=None):
        return np.ones(shape, dtype=dtype)

    def full(self, shape, fill_value, dtype=None):
        dtype = dtype or np.result_type(fill_value)
        return np.full(shape, fill_value, dtype=dtype)

    def ones_like(self, array):
        return np.ones(array.shape, dtype=array.dtype)

    def infinity_like(self, array):
        return np.full_like(array, np.inf)

    def conditional(self, cond, true_fn, false_fn, *args):
        if self.asnumpy(cond).all():
            return true_fn(*args)
        return false_fn(*args)

    def error_if(self, array, condition, msg):
        if np.any(condition):
            raise ValueError(msg)

    def outer(self, array1, array2):
        return self.tensordot(array1, array2, [[], []])

    def pad(self, array, paddings, mode, constant_values):
        if mode == "CONSTANT":
            mode = "constant"
        return np.pad(array, paddings, mode, constant_values=constant_values)

    @staticmethod
    def pinv(matrix):
        return np.linalg.pinv(matrix)

    def pow(self, x, y):
        return np.power(x, y)

    def kron(self, tensor1, tensor2):
        return np.kron(tensor1, tensor2)

    def prod(self, x, axis):
        return np.prod(x, axis=axis)

    def real(self, array):
        return np.real(array)

    def reshape(self, array, shape):
        return np.reshape(array, shape)

    def shape(self, array):
        return np.shape(array)

    def sin(self, array):
        return np.sin(array)

    def sinh(self, array):
        return np.sinh(array)

    def solve(self, matrix, rhs):
        if len(rhs.shape) == len(matrix.shape) - 1:
            rhs = np.expand_dims(rhs, -1)
            return np.linalg.solve(matrix, rhs)[..., 0]
        return np.linalg.solve(matrix, rhs)

    def sort(self, array, axis=-1):
        return np.sort(array, axis)

    def sqrt(self, x, dtype=None):
        return np.sqrt(self.cast(np.asarray(x), dtype))

    def stack(self, arrays, axis=0):
        return np.stack(arrays, axis=axis)

    def sum(self, array, axis=None):
        return np.sum(array, axis=axis)

    def swapaxes(self, array, axis1, axis2):
        return np.swapaxes(array, axis1, axis2)

    def tensordot(self, a, b, axes):
        return np.tensordot(a, b, axes)

    def tile(self, array, repeats):
        return np.tile(array, repeats)

    def trace(self, array, dtype=None):
        return self.cast(np.trace(array, axis1=-1, axis2=-2), dtype)

    def transpose(self, a, perm=None):
        return np.transpose(a, axes=perm)

    def tan(self, array):
        return np.tan(array)

    def tanh(self, array):
        return np.tanh(array)

    def update_add_tensor(self, tensor, indices, values):
        indices = self.atleast_nd(indices, 2)
        for i, v in zip(indices, values):
            tensor[tuple(i)] += v
        return tensor

    def zeros(self, shape, dtype=None):
        return np.zeros(shape, dtype=dtype)

    def zeros_like(self, array):
        return np.zeros_like(array, dtype=array.dtype)

    def map_fn(self, func, elements):
        # Is this done like this?
        return np.asarray([func(e) for e in elements])

    @staticmethod
    def eigvals(tensor):
        return np.linalg.eigvals(tensor)

    @staticmethod
    def xlogy(x, y):
        return scipy_xlogy(x, y)

    @staticmethod
    def eigh(tensor):
        return np.linalg.eigh(tensor)

    def sqrtm(self, tensor, dtype=None, rtol=1e-05, atol=1e-08):
        if np.allclose(tensor, 0, rtol=rtol, atol=atol):
            ret = self.zeros_like(tensor)
        else:
            ret = scipy_sqrtm(tensor)

        if dtype is None:
            return self.cast(ret, self.complex128)
        return self.cast(ret, dtype)

    def reorder_AB_bargmann(self, A, B):
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

    def hermite_renormalized(self, A, b, c, shape, stable, out=None):
        return cython_lattice.vanilla(tuple(shape), A, b, c, stable, out)

    def hermite_renormalized_batched(self, A, b, c, shape, stable, out=None):
        return cython_lattice.vanilla_batched(tuple(shape), A, b, c, stable, out)

    def hermite_renormalized_binomial(self, A, B, C, shape, max_l2, global_cutoff):
        return strategies.binomial(
            tuple(shape),
            A,
            B,
            C,
            max_l2=max_l2 or settings.AUTOSHAPE_PROBABILITY,
            global_cutoff=global_cutoff or sum(shape) - len(shape) + 1,
        )[0]

    def hermite_renormalized_diagonal(self, A, B, C, cutoffs, reorderedAB):
        A, B = self.reorder_AB_bargmann(A, B) if reorderedAB else (A, B)
        return hermite_multidimensional_diagonal(A, B, C, cutoffs)[0]

    def hermite_renormalized_1leftoverMode(
        self, A, b, c, output_cutoff, pnr_cutoffs, reorderedAB=True
    ):
        A, b = self.reorder_AB_bargmann(A, b) if reorderedAB else (A, b)
        shape = (output_cutoff + 1, *tuple(p + 1 for p in pnr_cutoffs))
        return hermite_multidimensional_1leftoverMode(A, b, c, shape)[0]

    # ~~~~~~~~~~~~~~~~~~~~~~~
    # Fock lattice strategies
    # ~~~~~~~~~~~~~~~~~~~~~~~

    def displacement(self, alpha, shape):
        alpha = self.astensor(alpha, dtype=self.complex128)
        batch_shape = alpha.shape
        if batch_shape == ():
            return cython_lattice.displacement(tuple(shape), alpha)
        alpha_flattened = alpha.reshape(-1)
        ret = cython_lattice.displacement_batched(tuple(shape), alpha_flattened)
        return ret.reshape((*batch_shape, *shape))

    def beamsplitter(self, theta, phi, shape, method):
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

    def homodyne_projector(self, fock_dim, A, b, c, out):
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

    def squeezed(self, r, phi, shape):
        r, phi = self.astensor(r, dtype=self.float64), self.astensor(phi, dtype=self.float64)
        batch_shape = r.shape
        if batch_shape == ():
            return cython_lattice.squeezed(int(shape[0]), r, phi)
        r_flattened = r.reshape(-1)
        phi_flattened = phi.reshape(-1)
        ret = cython_lattice.squeezed_batched(int(shape[0]), r_flattened, phi_flattened)
        return ret.reshape((*batch_shape, shape[0]))

    def squeezer(self, r, phi, shape):
        r, phi = self.astensor(r, dtype=self.float64), self.astensor(phi, dtype=self.float64)
        batch_shape = r.shape
        if batch_shape == ():
            return cython_lattice.squeezer(shape, r, phi)
        r_flattened = r.reshape(-1)
        phi_flattened = phi.reshape(-1)
        ret = cython_lattice.squeezer_batched(shape, r_flattened, phi_flattened)
        return ret.reshape((*batch_shape, *shape))
