# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains various numba functions needed by some methods in the :class:`Math` class."""

import numpy as np
from numba import njit
from mrmustard.types import *


@njit
def numba_sparse_matvec(matrix: Tensor, vector: Tensor, m_modes: Tuple[int], v_modes: Tuple[int], mlike_0:bool):
    r"""Numba implementation of the mode-wise matrix-vector multiplication of
    a batch of matrices and a batch of vectors in phase space. Assumes inputs are in xxpp ordering.
    Note that "sparse" is indended in the sense of modes, i.e. the matrix can contain
    fewer modes than the vector or the vector can contain fewer modes than the matrix.
    The operation will be performed only on the modes specified in the arguments.

    Args:
        matrix (array): :math:`B \times M\times M\times 2\times 2` batched array
        vector (array): :math:`B \times N\ times 2` batched vector
        m_modes (list(int)): list of ``M`` modes of the matrix
        v_modes (list(int)): list of ``N`` modes of the vector
        mlike_0 (bool): whether the matrix is considered to be zero on unspecified modes.
    Returns:
        array: :math: resulting vector (can have the same modes as the input vector or fewer)
    """
    # notes:
    # if mlike_0 is false, then we can always update vector in place because we are not chopping off any modes (and we don't care if m has more modes, as the vector is always like_0)
    # if mlike_0 is true, then we can update the vector in place only if v_modes is a subset of m_modes, otherwise we need to update a new (smaller) zero vector.

    sv = set(v_modes)
    sm = set(m_modes)
    f_modes = list(v_modes) if not mlike_0 else [v for v in v_modes if v in m_modes]
    B1 = matrix.shape[0]
    B2 = vector.shape[0]
    intersection = sv.intersection(sm)

    if mlike_0 and not sv.issubset(sm):
        output_vec = np.zeros((B1*B2, len(f_modes), 2), dtype=vector.dtype)
    else:
        if B2 > 1: # how about just outer with np.ones?
            vec_ = vector
            for _ in range(B2-1):
                vec_ = np.vstack((vec_, vector))
            output_vec = vec_
        else:
            output_vec = vector

    find = [f_modes.index(m) for m in intersection]
    mind = [m_modes.index(m) for m in intersection]
    vind = [v_modes.index(m) for m in intersection]

    for b1 in range(B1):
        for b2 in range(B2):
            b = b1*B2 + b2
            for i in range(len(intersection)):
                for j in range(len(intersection)):
                    output_vec[b, find[i]] = np.dot(matrix[b1, mind[i], mind[j]], vector[b2, vind[j]])
    return output_vec


def numba_sparse_matvec_vjp(dmatvec: Tensor, matrix: Tensor, vector: Tensor, m_modes: Tuple[int], v_modes: Tuple[int], like_0:bool):
    r"""Numba implementation of the mode-wise matrix-vector multiplication of
    a batch of matrices and a batch of vectors in phase space. Assumes inputs are in xxpp ordering.
    Note that "sparse" is indended in the sense of modes, i.e. the matrix can contain
    fewer modes than the vector or the vector can contain fewer modes than the matrix.
    The operation will be performed only on the modes specified in the arguments.

    Args:
        dmatvec (array): :math:`B \times M\times M\times 2\times 2` batched array
        matrix (array): :math:`B \times M\times M\times 2\times 2` batched array
        vector (array): :math:`B \times N\ times 2` batched vector
        m_modes (list(int)): list of ``M`` modes of the matrix
        v_modes (list(int)): list of ``N`` modes of the vector
        like_0 (bool): whether the matrix is considered to be zero on unspecified modes.
    Returns:
        array: :math: resulting vector (can have the same modes as the input vector or fewer)
    """
    sv = set(v_modes)
    sm = set(m_modes)
    f_modes = v_modes if not like_0 else [v for v in v_modes if v in m_modes]
    B1 = matrix.shape[0]
    B2 = vector.shape[0]

    dvector = np.zeros_like(vector)
    dmatrix = np.zeros_like(matrix)

    find = [f_modes.index(m) for m in sv.intersection(sm)]
    mind = [m_modes.index(m) for m in sv.intersection(sm)]
    vind = [v_modes.index(m) for m in sv.intersection(sm)]

    for b1 in range(B1):
        for b2 in range(B2):
            b = b1*B2 + b2
            for i in range(len(intersection)):
                for j in range(len(intersection)):
                    # TODO: review order of products and transposes
                    dvector[b2, vind[j]] += dmatvec[b, find[i]] @ matrix[b1, mind[i], mind[j]]
                    dmatrix[b1, mind[i], mind[j]] += np.outer(vector[b2, vind[i]], matvec[b, find[j]])

    return dvector, dmatrix


# @njit
# def numba_sparse_matmul_data(matrix1: Tensor, matrix2: Tensor, m1_modes: Tuple[int], m2_modes: Tuple[int], m1like_0: bool, m2like_0: bool) -> tuple:
#     r"""Computes the data required for the mode-wise matrix multiplication of two matrices."""
#     B1 = matrix1.shape[0]
#     B2 = matrix2.shape[0]
#     M = matrix1.shape[-1] // 2
#     N = matrix2.shape[-1] // 2

#     mode_union = list(set(m1_modes).union(set(m2_modes)))
#     mode_intersection = list(set(m1_modes).intersection(set(m2_modes)))

#     if m1like_0:  # final modes are a subset of m1_modes
#         if m2like_0: # final modes are a subset of m2_modes
#             final_modes = mode_intersection
#         else:
#             final_modes = list(m1_modes)
#     else:
#         if m2like_0: # final modes are a subset of m2_modes
#             final_modes = list(m2_modes)
#         else:
#             final_modes = mode_union

#     F = len(final_modes)

#     # at which index to write a given mode:
#     findices = {}
#     for i,m in enumerate(final_modes):
#         findices[m] = i
#     ind1 = {}
#     for i,m in enumerate(m1_modes):
#         ind1[m] = i
#     ind2 = {}
#     for i,m in enumerate(m2_modes):
#         ind2[m] = i

#     return mode_union, mode_intersection, final_modes, findices, ind1, ind2, B1, B2, F, M, N

# @njit
# def numba_sparse_matmul(matrix1: Tensor, matrix2: Tensor, m1_modes: List[int], m2_modes: List[int], m1like_0:bool, m2like_0:bool):
#     r"""Numba implementation of the mode-wise ("sparse") matrix-matrix multiplication `matrix1 @ matrix2`.
#     Assumes inputs are in xxpp ordering.

#     Args:
#         matrix1 (array): :math:`B1\times 2M\times 2M` array
#         matrix2 (array): :math:`B2\times 2N\times 2N` array
#         m1_modes (list(int)): list of ``M`` modes of the first matrix
#         m2_modes (list(int)): list of ``N`` modes of the second matrix
#         m1like_0 (bool): whether first matrix is like_0 or not
#         m2like_0 (bool): whether second matrix is like_0 or not
#     Returns:
#         new_matrix (array): :math:`B_1 B_2 \times 2F\times 2F` array where F is determined by the other arguments

#     """
#     union, intersection, final_modes, findices, ind1, ind2, B1, B2, F, M, N = numba_sparse_matmul_data(matrix1, matrix2, m1_modes, m2_modes, m1like_0, m2like_0)
    
#     new_matrix = np.zeros((B1*B2, 2*F, 2*F), dtype=matrix1.dtype)  # TODO: revisit dtype
#     for b1 in range(B1):
#         for b2 in range(B2):
#             b = b1*B2 + b2
#             for m in final_modes:
#                 for n in final_modes:
#                     if m in m1_modes:
#                         if n in m2_modes:  # if mode goes through both, add contribution
#                             for p in intersection:
#                                 new_matrix[b, findices[m], findices[n]] += matrix1[b1, ind1[m], ind1[p]] * matrix2[b2, ind2[p], ind2[n]] + matrix1[b1, ind1[m], ind1[p]+M] * matrix2[b2, ind2[p]+N, ind2[n]]
#                                 new_matrix[b, findices[m]+F, findices[n]] += matrix1[b1, ind1[m]+M, ind1[p]] * matrix2[b2, ind2[p], ind2[n]] + matrix1[b1, ind1[m]+M, ind1[p]+M] * matrix2[b2, ind2[p]+N, ind2[n]]
#                                 new_matrix[b, findices[m], findices[n]+F] += matrix1[b1, ind1[m], ind1[p]] * matrix2[b2, ind2[p], ind2[n]+N] + matrix1[b1, ind1[m], ind1[p]+M] * matrix2[b2, ind2[p]+N, ind2[n]+N]
#                                 new_matrix[b, findices[m]+F, findices[n]+F] += matrix1[b1, ind1[m]+M, ind1[p]] * matrix2[b2, ind2[p], ind2[n]+N] + matrix1[b1, ind1[m]+M, ind1[p]+M] * matrix2[b2, ind2[p]+N, ind2[n]+N]
#                         elif not m2like_0: # if n is not in m2_modes it contributes only if m2 is not like_0, in which case it copies the mode from m1
#                             new_matrix[b, findices[m], findices[n]] += matrix1[b1, ind1[m], ind1[n]]
#                             new_matrix[b, findices[m]+F, findices[n]] += matrix1[b1, ind1[m]+M, ind1[n]]
#                             new_matrix[b, findices[m], findices[n]+F] += matrix1[b1, ind1[m], ind1[n]+M]
#                             new_matrix[b, findices[m]+F, findices[n]+F] += matrix1[b1, ind1[m]+M, ind1[n]+M]
#                     elif not m1like_0:  # if m is not in m1_modes it matters only if m1 is not like_0, in which case it copies the mode from m2
#                         new_matrix[b, findices[m], findices[n]] += matrix2[b2, ind2[m], ind2[n]]
#                         new_matrix[b, findices[m]+F, findices[n]] += matrix2[b2, ind2[m]+N, ind2[n]]
#                         new_matrix[b, findices[m], findices[n]+F] += matrix2[b2, ind2[m], ind2[n]+N]
#                         new_matrix[b, findices[m]+F, findices[n]+F] += matrix2[b2, ind2[m]+N, ind2[n]+N]
#     return new_matrix

# @njit
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
        new_matrix (array): :math:`B_1 B_2 \times F\times F \times 2 \times 2` array where F is determined by the other arguments

    """
    B1 = matrix1.shape[0]
    B2 = matrix2.shape[0]
    s1 = set(m1_modes)
    s2 = set(m2_modes)
    union = s1.union(s2)
    intersection = s1.intersection(s2)
    I = len(intersection)
    U = len(union)
    # all cases fall in one of the following (we can always swap m1 and m2):
    # 1) 00
    # 2) 01 and 10
    # 3) 11

    if m1like_0 and m2like_0:
        # 1) 00
        if s1.issubset(s2):
            # 1.1) same as m1
            return numba_sparse_matmul_subset(matrix1, matrix2, m1_modes, m2_modes)
        elif s2.issubset(s1):
            # 1.2) same as m2
            return np.transpose(numba_sparse_matmul_subset(matrix2, matrix1, m2_modes, m1_modes), (0,2,1,4,3))
        elif len(intersection) == 0:
            # 1.3) no mode overlap
            return np.zeros((B1*B2, 0, 0, 2, 2), dtype=matrix1.dtype)
        else:
            # 1.4) some mode overlap
            output_mat = np.outer(np.ones((1,2,2)),np.identity(I)).transpose((0,3,4,1,2))
            output_mat = numba_sparse_matmul_subset(output_mat, matrix1, list(intersection), m1_modes)
            return numba_sparse_matmul_subset(output_mat, matrix2, list(intersection), m2_modes)
        
    if not m1like_0 and not m2like_0:
        # 3) 11
        if s1.issubset(s2):
            # 3.1) same as m2
            print(f'11 and {s1} subset of {s2}')
            return numba_sparse_matmul_subset(matrix1, matrix2, m1_modes, m2_modes)
        elif s2.issubset(s1):
            # 3.2) same as m1
            return np.transpose(numba_sparse_matmul_subset(matrix2, matrix1, m2_modes, m1_modes), (0,2,1,4,3))
        else:
            # 3.3) some or no mode overlap
            output_mat = np.outer(np.ones((1,2,2)),np.identity(U)).transpose((0,3,4,1,2))
            output_mat = numba_sparse_matmul_subset(output_mat, matrix1, list(union), m1_modes)
            return numba_sparse_matmul_subset(output_mat, matrix2, list(union), m2_modes)

    if m1like_0 != m2like_0:
        # 2) 01 and 10
        # 2.1) same as whichever is like_0
        if m1like_0:
            return numba_sparse_matmul_subset(matrix1, matrix2, m1_modes, m2_modes)
        else:
            return np.transpose(numba_sparse_matmul_subset(matrix2, matrix1, m2_modes, m1_modes), (0,2,1,4,3))
    

# @njit
def numba_sparse_matmul_subset(matrix1: Tensor, matrix2: Tensor, m1_modes: List[int], m2_modes: List[int]):
    r"""Numba implementation of the mode-wise ("sparse") matrix-matrix multiplication `matrix1 @ matrix2` where the
    result is guaranteed to have the same shape and modes as matrix1, i.e. m1_modes are a subset of m2_modes.

    Args:
        matrix1 (array): :math:`B1\times M\times M \times 2 \times 2` array
        matrix2 (array): :math:`B2\times N\times N \times 2 \times 2` array
        m1_modes (list(int)): list of ``M`` modes of the first matrix
        m2_modes (list(int)): list of ``N`` modes of the second matrix
    Returns:
        new_matrix (array): :math:`B_1 B_2 \times M\times M \times 2 \times 2` array
    """
    B1 = matrix1.shape[0]
    B2 = matrix2.shape[0]
    matrix1 = np.outer(np.ones((B2,)), matrix1).reshape((B1*B2,)+matrix1.shape[1:])
    output_mat = matrix1.copy()
    print(output_mat.shape)
    Z = np.zeros((2,2), dtype=matrix1.dtype)

    m2ind = [m2_modes.index(i) for i in m1_modes]

    for b1 in range(B1):
        for b2 in range(B2):
            b = b1*B2 + b2
            for i in range(len(m1_modes)):
                for k in range(len(m1_modes)):
                    output_mat[b, i, k] = Z
                    for j in range(len(m1_modes)):
                        output_mat[b, i, k] += matrix1[b1, i, j] @ matrix2[b2, m2ind[j], m2ind[k]]
    return output_mat

            

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
    union, intersection, final_modes, findices, ind1, ind2, B1, B2, F, M, N = numba_sparse_matmul_data(matrix1, matrix2, m1_modes, m2_modes, m1like_0, m2like_0)
    dmatrix1 = np.zeros((B1, 2*M, 2*M), dtype=matrix1.dtype)
    dmatrix2 = np.zeros((B2, 2*N, 2*N), dtype=matrix2.dtype)
    for b1 in range(B1):
        for b2 in range(B2):
            b = b1*B2 + b2
            for m in final_modes:
                for n in final_modes:
                    if m in m1_modes:
                        if n in m2_modes:
                            for p in intersection:
                                dmatrix1[b1, ind1[m], ind1[p]] += dmatmul[b, findices[m], findices[n]] * matrix2[b2, ind2[p], ind2[n]] + dmatmul[b, findices[m], findices[n]+F] * matrix2[b2, ind2[p], ind2[n]+N]
                                dmatrix1[b1, ind1[m], ind1[p]+M] += dmatmul[b, findices[m], findices[n]] * matrix2[b2, ind2[p]+N, ind2[n]] + dmatmul[b, findices[m], findices[n]+F] * matrix2[b2, ind2[p]+N, ind2[n]+N]
                                dmatrix1[b1, ind1[m]+M, ind1[p]] += dmatmul[b, findices[m]+F, findices[n]] * matrix2[b2, ind2[p], ind2[n]] + dmatmul[b, findices[m]+F, findices[n]+F] * matrix2[b2, ind2[p], ind2[n]+N]
                                dmatrix1[b1, ind1[m]+M, ind1[p]+M] += dmatmul[b, findices[m]+F, findices[n]] * matrix2[b2, ind2[p]+N, ind2[n]] + dmatmul[b, findices[m]+F, findices[n]+F] * matrix2[b2, ind2[p]+N, ind2[n]+N]
                                dmatrix2[b2, ind2[p], ind2[n]] += dmatmul[b, findices[m], findices[n]] * matrix1[b1, ind1[m], ind1[p]] + dmatmul[b, findices[m]+F, findices[n]] * matrix1[b1, ind1[m]+M, ind1[p]]
                                dmatrix2[b2, ind2[p], ind2[n]+N] += dmatmul[b, findices[m], findices[n]+F] * matrix1[b1, ind1[m]+M, ind1[p]] + dmatmul[b, findices[m]+F, findices[n]+F] * matrix1[b1, ind1[m]+M, ind1[p]+M]
                                dmatrix2[b2, ind2[p]+N, ind2[n]] += dmatmul[b, findices[m], findices[n]] * matrix1[b1, ind1[m], ind1[p]+M] + dmatmul[b, findices[m]+F, findices[n]] * matrix1[b1, ind1[m]+M, ind1[p]+M]
                                dmatrix2[b2, ind2[p]+N, ind2[n]+N] += dmatmul[b, findices[m], findices[n]+F] * matrix1[b1, ind1[m], ind1[p]+M] + dmatmul[b, findices[m]+F, findices[n]+F] * matrix1[b1, ind1[m]+M, ind1[p]+M]
                        elif not m2like_0:
                            dmatrix1[b1, ind1[m], ind1[n]] += dmatmul[b, findices[m], findices[n]]
                            dmatrix1[b1, ind1[m]+M, ind1[n]] += dmatmul[b, findices[m]+F, findices[n]]
                            dmatrix1[b1, ind1[m], ind1[n]+M] += dmatmul[b, findices[m], findices[n]+F]
                            dmatrix1[b1, ind1[m]+M, ind1[n]+M] += dmatmul[b, findices[m]+F, findices[n]+F]
                    elif not m1like_0:
                        dmatrix2[b2, ind2[m], ind2[n]] += dmatmul[b, findices[m], findices[n]]
                        dmatrix2[b2, ind2[m]+N, ind2[n]] += dmatmul[b, findices[m]+F, findices[n]]
                        dmatrix2[b2, ind2[m], ind2[n]+N] += dmatmul[b, findices[m], findices[n]+F]
                        dmatrix2[b2, ind2[m]+N, ind2[n]+N] += dmatmul[b, findices[m]+F, findices[n]+F]
    return dmatrix1, dmatrix2



