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



@njit
def numba_sparse_matvec_data(matrix: Tensor, vector: Tensor, m_modes: Tuple[int], v_modes: Tuple[int], like_0:bool) -> tuple:
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
def numba_sparse_matvec_vjp(dmatvec: Tensor, matrix: Tensor, vector: Tensor, m_modes: Tuple[int], v_modes: Tuple[int], like_0:bool):
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
def numba_sparse_matmul_data(matrix1: Tensor, matrix2: Tensor, m1_modes: Tuple[int], m2_modes: Tuple[int], m1like_0: bool, m2like_0: bool) -> tuple:
    r"""Computes the data required for the mode-wise matrix multiplication of two matrices."""
    B1 = matrix1.shape[0]
    B2 = matrix2.shape[0]
    M = matrix1.shape[-1] // 2
    N = matrix2.shape[-1] // 2

    mode_union = list(set(m1_modes).union(set(m2_modes)))
    mode_intersection = list(set(m1_modes).intersection(set(m2_modes)))

    if m1like_0:  # final modes are a subset of m1_modes
        if m2like_0: # final modes are a subset of m2_modes
            final_modes = mode_intersection
        else:
            final_modes = list(m1_modes)
    else:
        if m2like_0: # final modes are a subset of m2_modes
            final_modes = list(m2_modes)
        else:
            final_modes = mode_union

    F = len(final_modes)

    # at which index to write a given mode:
    findices = {}
    for i,m in enumerate(final_modes):
        findices[m] = i
    ind1 = {}
    for i,m in enumerate(m1_modes):
        ind1[m] = i
    ind2 = {}
    for i,m in enumerate(m2_modes):
        ind2[m] = i

    return mode_union, mode_intersection, final_modes, findices, ind1, ind2, B1, B2, F, M, N

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
    union, intersection, final_modes, findices, ind1, ind2, B1, B2, F, M, N = sparse_matmul_data(matrix1, matrix2, m1_modes, m2_modes, m1like_0, m2like_0)
    
    new_matrix = np.zeros((B1*B2, 2*F, 2*F), dtype=matrix1.dtype)  # TODO: revisit dtype
    for b1 in range(B1):
        for b2 in range(B2):
            b = b1*B2 + b2
            for m in final_modes:
                for n in final_modes:
                    if m in m1_modes:
                        if n in m2_modes:  # if mode goes through both, add contribution
                            for p in intersection:
                                new_matrix[b, findices[m], findices[n]] += matrix1[b1, ind1[m], ind1[p]] * matrix2[b2, ind2[p], ind2[n]] + matrix1[b1, ind1[m], ind1[p]+M] * matrix2[b2, ind2[p]+N, ind2[n]]
                                new_matrix[b, findices[m]+F, findices[n]] += matrix1[b1, ind1[m]+M, ind1[p]] * matrix2[b2, ind2[p], ind2[n]] + matrix1[b1, ind1[m]+M, ind1[p]+M] * matrix2[b2, ind2[p]+N, ind2[n]]
                                new_matrix[b, findices[m], findices[n]+F] += matrix1[b1, ind1[m], ind1[p]] * matrix2[b2, ind2[p], ind2[n]+N] + matrix1[b1, ind1[m], ind1[p]+M] * matrix2[b2, ind2[p]+N, ind2[n]+N]
                                new_matrix[b, findices[m]+F, findices[n]+F] += matrix1[b1, ind1[m]+M, ind1[p]] * matrix2[b2, ind2[p], ind2[n]+N] + matrix1[b1, ind1[m]+M, ind1[p]+M] * matrix2[b2, ind2[p]+N, ind2[n]+N]
                        elif not m2like_0: # if n is not in m2_modes it contributes only if m2 is not like_0, in which case it copies the mode from m1
                            new_matrix[b, findices[m], findices[n]] += matrix1[b1, ind1[m], ind1[n]]
                            new_matrix[b, findices[m]+F, findices[n]] += matrix1[b1, ind1[m]+M, ind1[n]]
                            new_matrix[b, findices[m], findices[n]+F] += matrix1[b1, ind1[m], ind1[n]+M]
                            new_matrix[b, findices[m]+F, findices[n]+F] += matrix1[b1, ind1[m]+M, ind1[n]+M]
                    elif not m1like_0:  # if m is not in m1_modes it matters only if m1 is not like_0, in which case it copies the mode from m2
                        new_matrix[b, findices[m], findices[n]] += matrix2[b2, ind2[m], ind2[n]]
                        new_matrix[b, findices[m]+F, findices[n]] += matrix2[b2, ind2[m]+N, ind2[n]]
                        new_matrix[b, findices[m], findices[n]+F] += matrix2[b2, ind2[m], ind2[n]+N]
                        new_matrix[b, findices[m]+F, findices[n]+F] += matrix2[b2, ind2[m]+N, ind2[n]+N]
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
    union, intersection, final_modes, findices, ind1, ind2, B1, B2, F, M, N = sparse_matmul_data(matrix1, matrix2, m1_modes, m2_modes, m1like_0, m2like_0)
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



def numba_sparse_vec_add(vec1, vec2, modes1, modes2):
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


def numba_sparse_mat_add(mat1, mat2, modes1, modes2, m1like_0, m2like_0):
    B1 = len(mat1)
    B2 = len(mat2)


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
#     ind1 = {m:i for i,m in enumerate(m1_modes)}
#     ind2 = {m:i for i,m in enumerate(m2_modes)}

#     return final_modes, findices, ind1, ind2, B, F, M, N




