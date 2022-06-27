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
def numba_vec_add(vec1, vec2, modes1, modes2):
    r"""Mode-wise sparse addition of vec1 and vec2, with outer product on the batch dimension.
    
        Args:
            vec1 (array): A vector of shape (B1, 2M) that lives on `modes1`
            vec2 (array): A vector of shape (B2, 2N) that lives on `modes2`
            modes1 (array): A vector of shape (M,) that contains the modes of `vec1`
            modes2 (array): A vector of shape (N,) that contains the modes of `vec2`
        
        Returns:
            array: A vector of shape (B1*B2, 2U) that lives on the union of `modes1` and `modes2`
    """
    s1 = set(modes1)
    s2 = set(modes2)
    union = s1.union(s2)

    if s2.issubset(s1):
        vec = numba_vec_add_inplace(vec1, vec2, modes1, modes2)
    elif s1.issubset(s2):
        vec = numba_vec_add_inplace(vec2, vec1, modes2, modes1)
    else:
        vec = np.zeros((1, 2*len(union)), dtype=vec1.dtype)  # TODO: revisit dtype
        vec = numba_vec_add_inplace(vec, vec1, [m for m in union], modes1)
        vec = numba_vec_add_inplace(vec, vec2, [m for m in union], modes2)

    return vec


@njit
def numba_vec_add_inplace(vec, subvec, modes, submodes):
    r"""Mode-wise sparse addition of vec_sub to vec, with outer product on the batch dimension.
    It assumes submodes are a subset of modes, hence we can just add to vec without creating a new array.
    Warning: it updates vec inplace.
    
        Args:
            vec (array): :math:`B1\times 2M` batched 'large' vector
            subvec (array): :math:`B2\times 2N` batched 'small' vector
            modes (list): list of modes in vec
            submodes (list): list of modes in subvec, must be a subset of modes
        Returns:
            array: :math:`(B1 B2)\times 2M` batched subvec with added vec
    """
    B1 = len(vec)
    B2 = len(subvec)
    if B2 > 1:
        vec_ = vec
        for _ in range(B2-1):
            vec_ = np.vstack((vec_, vec))
        vec = vec_

    modes_index = [modes.index(m) for m in submodes]

    for b1 in range(B1):
        for b2 in range(B2):
            b = b1*B2 + b2  # NOTE: is this giving the opposite vectorization order when vec1 and vec2 are swapped? isn't this a problem?
            for i,m in enumerate(submodes):
                vec[b, modes_index[i]] += subvec[b2, i]
                vec[b, modes_index[i]+len(modes)] += subvec[b2, i+len(submodes)]
    return vec


@njit
def numba_mat_add(mat1, mat2, modes1, modes2, m1like_0, m2like_0):
    r"""Mode-wise sparse addition of mat1 and mat2, with outer product on the batch dimension.
    Warning: it may update mat1 or mat2 inplace.

        Args:
            mat1 (array): A matrix of shape (B1, 2M, 2M) that lives on `modes1`
            mat2 (array): A matrix of shape (B2, 2N, 2N) that lives on `modes2`
            modes1 (array): A vector of shape (M,) that contains the modes of `mat1`
            modes2 (array): A vector of shape (N,) that contains the modes of `mat2`
            m1like_0 (bool): If True, mat1 is assumed to be like 0 on modes not in modes1
            m2like_0 (bool): If True, mat2 is assumed to be like 0 on modes not in modes2

        Returns:
            array: A matrix of shape (B1*B2, 2U, 2U) that lives on the union of `modes1` and `modes2`
    """

    s1 = set(modes1)
    s2 = set(modes2)
    union = s1.union(s2)
    union_list = list(union)

    if s2.issubset(s1):
        mat = numba_mat_add_inplace(mat1, mat2, modes1, modes2, m2like_0)
    elif s1.issubset(s2):
        mat = numba_mat_add_inplace(mat2, mat1, modes2, modes1, m1like_0)
    else:
        mat = np.zeros((len(mat1)*len(mat2), 2*len(union), 2*len(union)), dtype=mat1.dtype)
        mat = numba_mat_add_inplace(mat, mat1, union_list, modes1, m1like_0)
        mat = numba_mat_add_inplace(mat, mat2, union_list, modes2, m2like_0)
    return mat


@njit
def numba_mat_add_inplace(mat, submat, modes, submodes, submatlike_0):
    r"""Mode-wise sparse addition of mat_sub to mat, with outer product on the batch dimension.
    It assumes submodes are a subset of modes, hence we can just add to mat without creating a new array.
    Warning: it updates mat inplace.

        Args:
            mat (array): :math:`B1\times 2M\times 2M` batched matrix living in `modes`
            submat (array): :math:`B2\times 2N\times 2N` batched submatrix living in `submodes`
            modes (list): list of modes in mat
            submodes (list): list of modes in submat, must be a subset of modes
            matlike_0 (bool): whether mat is a like the zero matrix or the identity matrix on unspecified modes
            submatlike_0 (bool): whether submat is a like the zero matrix or the identity matrix on unspecified modes
        
        Returns:
            array: :math:`(B1 B2)\times 2M\times 2M` result respecting the effect on unspecified modes and the outer product on the batch dimension
    """
    B = len(mat)
    Bsub = len(submat)
    if Bsub > 1:
        # this loop is because numba doesn't support np.tile(mat, (Bsub, 1, 1)) 
        mat_ = mat
        for _ in range(Bsub-1):
            mat_ = np.vstack((mat_, mat))
        mat = mat_
    
    modes_index = [modes.index(m) for m in submodes]
    for b_ in range(B):
        for bsub in range(Bsub):
            b = b_*Bsub + bsub # vectorize
            for i,m in enumerate(submodes):
                for j,n in enumerate(submodes):
                    mat[b, modes_index[i], modes_index[j]] += submat[bsub, i, j]
                    mat[b, modes_index[i]+len(modes), modes_index[j]] += submat[bsub, i+len(submodes), j]
                    mat[b, modes_index[i], modes_index[j]+len(modes)] += submat[bsub, i, j+len(submodes)]
                    mat[b, modes_index[i]+len(modes), modes_index[j]+len(modes)] += submat[bsub, i+len(submodes), j+len(submodes)]
            if not submatlike_0:
                for i,m in enumerate(set(modes) - set(submodes)):
                    # +1 on the diagonal
                    mat[b, i, i] += 1
                    mat[b, i+len(modes), i+len(modes)] += 1
    return mat

########################
####### GRADIENTS ######
########################


@njit
def numba_vec_add_vjp(dL_dsum, vec1, vec2, modes1, modes2):
    r"""vjp of mode-wise sparse addition of vec1 to vec2.
    """
    s1 = set(modes1)
    s2 = set(modes2)
    union = s1.union(s2)
    
    if s2.issubset(s1):
        dL_dvec1, dL_dvec2 = numba_vec_add_inplace_vjp(dL_dsum, len(vec1), len(vec2), modes1, modes2)
    elif s1.issubset(s2):
        dL_dvec2, dL_dvec1 = numba_vec_add_inplace_vjp(dL_dsum, len(vec2), len(vec1), modes2, modes1)
    else:
        _, dL_dvec1 = numba_vec_add_inplace_vjp(dL_sum, len(dL_sum), len(vec1), [m for m in union], modes1)
        _, dL_dvec2 = numba_vec_add_inplace_vjp(dL_sum, len(dL_sum), len(vec2), [m for m in union], modes2)
        return dL_dvec1, dL_dvec2



@njit
def numba_vec_add_inplace_vjp(dL_sum, B, Bsub, modes, submodes):
    r"""vjp of mode-wise sparse addition of vec_sub to vec.
    
        Args:
            dL_dsum (array): :math:`(B1 B2)\times 2M` batched derivative of loss with respect to sum
            vec (array): :math:`B1\times 2M` batched large vector
            subvec (array): :math:`B2\times 2N` batched small vector
            modes (list): list of modes in vec
            submodes (list): list of modes in vec_sub, must be a subset of modes1
        Returns:
            dL_dvec (array): :math:`B1\times 2M` batched derivative of L wrt vec
            dL_dsubvec (array): :math:`B2\times 2N` batched derivative of L wrt subvec
    """
    dL_dvec = np.zeros((B, 2*len(modes)), dtype=dL_sum.dtype)
    dL_dsubvec = np.zeros((Bsub, 2*len(submodes)), dtype=dL_sum.dtype)
    modes_index = [modes.index(m) for m in submodes]
    for b1 in range(B):
        for b2 in range(Bsub):
            b = b1*B2 + b2
            for i,m in enumerate(submodes):
                dL_dvec[b1, modes_index[i]] += dL_sum[b, i]
                dL_dvec[b1, modes_index[i]+len(modes)] += dL_sum[b, i+len(submodes)]
                dL_dsubvec[b2, i] += dL_sum[b, modes_index[i]]
                dL_dsubvec[b2, i+len(submodes)] += dL_sum[b, modes_index[i]+len(modes)]
    return dL_dvec, dL_dsubvec







@njit
def numba_mat_add_vjp(dLdsum, B1, B2, modes1, modes2):
    pass
    # fmodes, findices, ind1, ind2, F, M, N = numba_sparse_mat_add_prep(modes1, modes2)

    # dmatrix1 = np.zeros((B1, 2*M, 2*M), dtype=mat1.dtype)
    # dmatrix2 = np.zeros((B2, 2*N, 2*N), dtype=mat2.dtype)
    # # loop over vectorized batches
    # for b in range(B1*B2):
    #     b1 = b // B2
    #     b2 = b % B2
    #     # loop over final modes of the final matrix and skip addition of constants
    #     for f1 in fmodes:
    #         for f2 in fmodes:
    #             if f1 in modes1:
    #                 dmatrix1[b1, ind1[f1], ind1[f2]] += dLdsum[b, indf[f1], indf[f2]]
    #                 dmatrix1[b1, ind1[f1]+M, ind1[f2]] += dLdsum[b, indf[f1]+F, indf[f2]]
    #                 dmatrix1[b1, ind1[f1], ind1[f2]+M] += dLdsum[b, indf[f1], indf[f2]+F]
    #                 dmatrix1[b1, ind1[f1]+M, ind1[f2]+M] += dLdsum[b, indf[f1]+F, indf[f2]+F]
    #             if f2 in modes2:
    #                 dmatrix2[b2, ind2[f2], ind2[f1]] += dLdsum[b, indf[f1], indf[f2]]
    #                 dmatrix2[b2, ind2[f2]+N, ind2[f1]] += dLdsum[b, indf[f1]+F, indf[f2]]
    #                 dmatrix2[b2, ind2[f2], ind2[f1]+N] += dLdsum[b, indf[f1], indf[f2]+F]
    #                 dmatrix2[b2, ind2[f2]+N, ind2[f1]+N] += dLdsum[b, indf[f1]+F, indf[f2]+F]
    # return dmatrix1, dmatrix2



@njit
def numba_mat_add_prep(modes1, modes2, m1like_0, m2like_0):
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
