# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""This file contains the implementation of the complex Gaussian integral for one or two Abc.
Refer to the documentation of the complex_gaussian_integral_1 and complex_gaussian_integral_2
functions in the backend_manager.py file for more details.
"""

import numpy as np
from numba import guvectorize, njit

__all__ = [
    "complex_gaussian_integral_1_guvectorized",
    "complex_gaussian_integral_1_jitted",
    "complex_gaussian_integral_2_guvectorized",
    "complex_gaussian_integral_2_jitted",
]


@njit(cache=True, fastmath=True)
def _get_not_indices(total_size, indices_to_exclude):  # pragma: no cover
    r"""Creates a boolean mask and returns the indices where the mask is True."""
    mask = np.full(total_size, True)
    mask[indices_to_exclude] = False
    return np.where(mask)[0]


@njit(cache=True, fastmath=True)
def complex_gaussian_integral_1_jitted(A, b, idx12, A_out, b_out, log_c_out):  # pragma: no cover
    r"""Python implementation of the complex Gaussian integral for one Abc."""
    not_idx12 = _get_not_indices(A.shape[-1], idx12)  # because we want to njit this function

    B, C, D = A[idx12, :][:, idx12], A[not_idx12, :][:, idx12], A[not_idx12, :][:, not_idx12]
    g, h = b[idx12], b[not_idx12]

    n = B.shape[-1]  # this is 2 times m where m is number of conjugate pairs
    m = n // 2
    X = np.zeros((n, n), dtype=B.dtype)
    for i in range(m):
        X[i, m + i] = 1.0
        X[m + i, i] = 1.0

    M = B - X
    det = np.linalg.det(1j * M)

    if np.isclose(np.abs(det), 0, atol=1e-20):
        A_out[:, :] = np.inf
        b_out[:] = np.inf
        log_c_out[0] = np.inf
        return

    Minv = np.linalg.inv(M)

    A_out[:, :] = D - C @ Minv @ C.T
    b_out[:] = h - C @ Minv @ g
    log_c_out[0] = -0.5 * (g @ Minv @ g) - 0.5 * np.log(det)


@njit(cache=True, fastmath=True)
def complex_gaussian_integral_2_jitted(
    A1, b1, A2, b2, idx1, idx2, A_out, b_out, log_c_out
):  # pragma: no cover
    r"""Implementation of the complex Gaussian integral for two Abc."""
    not_idx1 = _get_not_indices(A1.shape[-1], idx1)
    not_idx2 = _get_not_indices(A2.shape[-1], idx2)

    A, C, B = A1[idx1, :][:, idx1], A1[not_idx1, :][:, idx1], A1[not_idx1, :][:, not_idx1]
    D, F, E = A2[idx2, :][:, idx2], A2[not_idx2, :][:, idx2], A2[not_idx2, :][:, not_idx2]
    g, h = b1[idx1], b1[not_idx1]
    i, j = b2[idx2], b2[not_idx2]

    m = A1.shape[-1] - len(idx1)
    invL = A @ D - np.eye(A.shape[0])
    detinv = np.linalg.det(-invL)

    if np.isclose(np.abs(detinv), 0, atol=1e-20):
        A_out[:, :] = np.inf
        b_out[:] = np.inf
        log_c_out[0] = np.inf
        return
    L = np.linalg.inv(invL)

    LCT = L @ C.T
    A_out[:m, :m] = B - C @ D @ LCT
    A_out[m:, :m] = -F @ LCT
    A_out[:m, m:] = A_out[m:, :m].T
    A_out[m:, m:] = E - F @ L @ A @ F.T

    Lg = L @ g
    LTi = L.T @ i
    b_out[:m] = h - C @ (D @ Lg) - C @ LTi
    b_out[m:] = j - F @ (A @ LTi) - F @ Lg

    log_c_out[0] = (-0.5 * (g @ D @ Lg + 2 * g @ LTi + i @ A @ LTi)) - 0.5 * np.log(detinv)


# NOTE: guvectorized functions cannot be called from njit function if target="parallel", but njit functions can be called from guvectorized ones.
# NOTE: guvectorized functions compile at decoration time, not at call time.


@guvectorize(
    ["(c16[:,:],c16[:],c16[:,:],c16[:],i8[:],i8[:],c16[:,:],c16[:],c16[:])"],
    "(n,n),(n),(m,m),(m),(k),(k),(j,j),(j),(c)",
    target="parallel",
    cache=True,
    fastmath=True,
)
def complex_gaussian_integral_2_guvectorized(
    A1, b1, A2, b2, idx1, idx2, A_out, b_out, log_c_out
):  # pragma: no cover
    r"""Guvectorized implementation of the complex Gaussian integral for two Abc."""
    return complex_gaussian_integral_2_jitted(A1, b1, A2, b2, idx1, idx2, A_out, b_out, log_c_out)


@guvectorize(
    ["(c16[:,:],c16[:],i8[:],c16[:,:],c16[:],c16[:])"],
    "(n,n),(n),(k),(j,j),(j),(c)",
    target="parallel",
    cache=True,
    fastmath=True,
)
def complex_gaussian_integral_1_guvectorized(
    A, b, idx12, A_out, b_out, log_c_out
):  # pragma: no cover
    r"""Guvectorized implementation of the complex Gaussian integral for one Abc."""
    return complex_gaussian_integral_1_jitted(A, b, idx12, A_out, b_out, log_c_out)
