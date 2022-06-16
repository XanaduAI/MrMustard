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
    final_modes, findices, mindices, vindices, B1, B2, F, M, V = numba_sparse_matvec_data(matrix, vector, m_modes, v_modes, like_0)

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
    final_modes, findices, mindices, vindices, B1, B2, F, M, V = numba_sparse_matvec_data(matrix, vector, m_modes, v_modes, like_0)

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
    union, intersection, final_modes, findices, ind1, ind2, B1, B2, F, M, N = numba_sparse_matmul_data(matrix1, matrix2, m1_modes, m2_modes, m1like_0, m2like_0)
    
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


@njit
def numba_sparse_vec_add_inplace(vec, subvec, modes, submodes):
    r"""Mode-wise sparse addition of vec_sub to vec, with outer product on the batch dimension.
    It assumes modes2 are a subset of modes1, hence we can just add to vec1 without creating a new array.
    
        Args:
            vec (array): :math:`B1\times 2M` batched large vector
            subvec (array): :math:`B2\times 2N` batched small vector
            modes (list): list of modes in vec
            submodes (list): list of modes in vec_sub, must be a subset of modes1
        Returns:
            array: :math:`(B1 B2)\times 2M` batched vec1 with added vec2
    """
    B1 = len(vec)
    B2 = len(subvec)
    if B2 > 1:
        vec_ = vec
        for _ in range(B2-1):
            vec_ = np.vstack((vec_, vec))
        vec = vec_
    findices = [modes.index(m) for m in submodes]
    for b1 in range(B1):
        for b2 in range(B2):
            b = b1*B2 + b2  # NOTE: isn't this giving the opposite vectorization order when vec1 and vec2 are swapped? isn't this a problem?
            for i,m in enumerate(submodes):
                vec[b, findices[m]] += subvec[b2, i]
                vec[b, findices[m]+len(modes)] += subvec[b2, i+len(submodes)]
    return vec

@njit
def numba_sparse_vec_add(vec1, vec2, modes1, modes2):
    r"""Mode-wise sparse addition of two vectors in phase space.
    Addition only takes place in the specified modes.
    """
    B1 = len(vec1)
    B2 = len(vec2)
    fmodes = set(modes1).union(set(modes2))
    F = len(fmodes)
    M = len(modes1)
    N = len(modes2)
    ind1 = [modes1.index(m) for m in modes1]
    ind2 = [modes2.index(m) for m in modes2]

    if set(modes2).issubset(set(modes1)):
        # add to vec1
        vec = numba_sparse_vec_add_inplace(vec1, vec2, modes1, modes2)
    elif set(modes1).issubset(set(modes2)):
        # add to vec2
        vec = numba_sparse_vec_add_inplace(vec2, vec1, modes2, modes1)
    else:
        # add both to a new vector
        vec = np.zeros((B1*B2, 2*F), dtype=vec1.dtype)
        for b1 in range(B1):
            for b2 in range(B2):
                b = b1*B2 + b2
                for k,f in enumerate(fmodes):
                    if f in modes1:
                        vec[b, k] += vec1[b1, modes1.index(f)]
                        vec[b, k+F] += vec1[b1, modes1.index(f)+M]
                    if f in modes2:
                        vec[b, k] += vec2[b2, modes2.index(f)]
                        vec[b, k+F] += vec2[b2, modes2.index(f)+N]
    return vec

@njit
def numba_sparse_vec_add_vjp(dLdsum, vec1, vec2, modes1, modes2):
    B1 = len(vec1)
    B2 = len(vec2)
    fmodes = set(modes1).union(set(modes2))
    F = len(fmodes)
    
    dvec1 = np.zeros((B1, 2*F), dtype=vec1.dtype)
    dvec2 = np.zeros((B2, 2*F), dtype=vec2.dtype)
    for b in range(B1*B2):
        b1 = b // B2
        b2 = b % B2
        for f in fmodes:
            if f in modes1:
                dvec1[b, fmodes[f]] += dLdsum[b, modes1[f]]
                dvec1[b, fmodes[f]+F] += dLdsum[b, modes1[f]+F]
            if f in modes2:
                dvec2[b, fmodes[f]] += dLdsum[b, modes2[f]]
                dvec2[b, fmodes[f]+F] += dLdsum[b, modes2[f]+F]
    return dvec1, dvec2

@njit
def numba_sparse_mat_add(mat1, mat2, modes1, modes2, m1like_0, m2like_0):
    fmodes, findices, ind1, ind2, B1, B2, F, M, N = sparse_mat_add_prep(modes1, modes2, m1like_0, m2like_0)

    matrix = np.zeros((B1*B2, 2*F, 2*F), dtype=mat1.dtype)
    # loop over vectorized batches
    for b1 in range(B1):
        for b2 in range(B2):
            b = b1*B2 + b2
            # loop over final modes of the final matrix
            for f1,f2 in fmodes:
                if f1 in modes1: # if f1 is in modes of the first matrix
                    matrix[b, findices[f1], findices[f2]] += mat1[b1, ind1[f1], ind1[f2]]
                    matrix[b, findices[f1]+F, findices[f2]] += mat1[b1, ind1[f1]+M, ind1[f2]]
                    matrix[b, findices[f1], findices[f2]+F] += mat1[b1, ind1[f1], ind1[f2]+M]
                    matrix[b, findices[f1]+F, findices[f2]+F] += mat1[b1, ind1[f1]+M, ind1[f2]+M]
                # if not, we add a 1 but only in the diagonal and only if m1 is not like 0
                elif f1==f2 and not m1like_0:
                    matrix[b, findices[f1], findices[f1]] += 1
                    matrix[b, findices[f1]+F, findices[f1]+F] += 1
                if f2 in modes2:
                    matrix[b, findices[f1], findices[f2]] += mat2[b2, ind2[f2], ind2[f1]]
                    matrix[b, findices[f1]+F, findices[f2]] += mat2[b2, ind2[f2]+N, ind2[f1]]
                    matrix[b, findices[f1], findices[f2]+F] += mat2[b2, ind2[f2], ind2[f1]+N]
                    matrix[b, findices[f1]+F, findices[f2]+F] += mat2[b2, ind2[f2]+N, ind2[f1]+N]
                elif f1==f2 and not m2like_0:
                    matrix[b, findices[f1], findices[f1]] += 1
                    matrix[b, findices[f1]+F, findices[f1]+F] += 1
    return matrix

@njit
def numba_sparse_mat_add_vjp(dLdsum, B1, B2, modes1, modes2):
    fmodes, findices, ind1, ind2, F, M, N = sparse_mat_add_prep(modes1, modes2)

    dmatrix1 = np.zeros((B1, 2*M, 2*M), dtype=mat1.dtype)
    dmatrix2 = np.zeros((B2, 2*N, 2*N), dtype=mat2.dtype)
    # loop over vectorized batches
    for b in range(B1*B2):
        b1 = b // B2
        b2 = b % B2
        # loop over final modes of the final matrix and skip addition of constants
        for f1 in fmodes:
            for f2 in fmodes:
                if f1 in modes1:
                    dmatrix1[b1, ind1[f1], ind1[f2]] += dLdsum[b, indf[f1], indf[f2]]
                    dmatrix1[b1, ind1[f1]+M, ind1[f2]] += dLdsum[b, indf[f1]+F, indf[f2]]
                    dmatrix1[b1, ind1[f1], ind1[f2]+M] += dLdsum[b, indf[f1], indf[f2]+F]
                    dmatrix1[b1, ind1[f1]+M, ind1[f2]+M] += dLdsum[b, indf[f1]+F, indf[f2]+F]
                if f2 in modes2:
                    dmatrix2[b2, ind2[f2], ind2[f1]] += dLdsum[b, indf[f1], indf[f2]]
                    dmatrix2[b2, ind2[f2]+N, ind2[f1]] += dLdsum[b, indf[f1]+F, indf[f2]]
                    dmatrix2[b2, ind2[f2], ind2[f1]+N] += dLdsum[b, indf[f1], indf[f2]+F]
                    dmatrix2[b2, ind2[f2]+N, ind2[f1]+N] += dLdsum[b, indf[f1]+F, indf[f2]+F]
    return dmatrix1, dmatrix2



@njit
def sparse_mat_add_prep(modes1, modes2, m1like_0, m2like_0):
    """
    Prepare the sparse matrix addition for numba.
    """
    # get the final modes as the union of the two input modes
    final_modes = set(modes1).union(set(modes2))
    # get the indices of the final modes
    indf = dict()
    for i,f in enumerate(final_modes):
        fmodes[f] = i
    # get the indices of the modes in the first matrix
    ind1 = dict()
    for i,m in enumerate(modes1):
        ind1[m] = i
    # get the indices of the modes in the second matrix
    ind2 = dict()
    for i,m in enumerate(modes2):
        ind2[m] = i

    return fmodes, findices, 

    # get the indices of the final modes in the final matrix




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




