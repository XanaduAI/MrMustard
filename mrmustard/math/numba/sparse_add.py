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
def vec_add(vec1, vec2, modes1, modes2):
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
        vec = vec_add_inplace(vec1, vec2, modes1, modes2)
    elif s1.issubset(s2):
        vec = vec_add_inplace(vec2, vec1, modes2, modes1)
    else:
        vec = np.zeros((1, 2*len(union)), dtype=vec1.dtype)  # TODO: revisit dtype
        vec = vec_add_inplace(vec, vec1, [m for m in union], modes1)
        vec = vec_add_inplace(vec, vec2, [m for m in union], modes2)

    return vec


@njit
def vec_add_inplace(vec, subvec, modes, submodes):
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

    for b1 in range(B1):
        for b2 in range(B2):
            b = b1*B2 + b2  # NOTE: is this giving the opposite vectorization order when vec1 and vec2 are swapped? isn't this a problem?
            for i,m in enumerate(submodes):
                vec[b, modes.index(m)] += subvec[b2, i]
                vec[b, modes.index(m)+len(modes)] += subvec[b2, i+len(submodes)]
    return vec

########################
####### GRADIENTS ######
########################


@njit
def vec_add_vjp(dL_dsum, vec1, vec2, modes1, modes2):
    r"""vjp of mode-wise sparse addition of vec1 to vec2.
    """
    s1 = set(modes1)
    s2 = set(modes2)
    union = s1.union(s2)
    
    if s2.issubset(s1):
        dL_dvec1, dL_dvec2 = vec_add_vjp_inplace(dL_dsum, len(vec1), len(vec2), modes1, modes2)
    elif s1.issubset(s2):
        dL_dvec2, dL_dvec1 = vec_add_vjp_inplace(dL_dsum, len(vec2), len(vec1), modes2, modes1)
    else:
        _, dL_dvec1 = vec_add_vjp_inplace(dL_sum, len(dL_sum), len(vec1), [m for m in union], modes1)
        _, dL_dvec2 = vec_add_vjp_inplace(dL_sum, len(dL_sum), len(vec2), [m for m in union], modes2)
        return dL_dvec1, dL_dvec2



@njit
def vec_add_inplace_vjp(dL_sum, B, Bsub, modes, submodes):
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
    for b1 in range(B):
        for b2 in range(Bsub):
            b = b1*B2 + b2
            for i,m in enumerate(submodes):
                dL_dvec[b1, modes.index(m)] += dL_sum[b, i]
                dL_dvec[b1, modes.index(m)+len(modes)] += dL_sum[b, i+len(submodes)]
                dL_dsubvec[b2, i] += dL_sum[b, modes.index(m)]
                dL_dsubvec[b2, i+len(submodes)] += dL_sum[b, modes.index(m)+len(modes)]
    return dL_dvec, dL_dsubvec



@njit
def mat_add_inplace(mat, submat, modes, submodes, matlike_0, submatlike_0):
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
    if B2 > 1:
        mat_ = mat
        for _ in range(Bsub-1):
            mat_ = np.vstack((mat_, mat))
        mat = mat_
    
    for b in range(B):
        for bsub in range(Bsub):
            b = b*Bsub + bsub # vectorize
            for i,m in enumerate(submodes):
                for j,n in enumerate(submodes):
                    mat[b, modes.index(m), modes.index(n)] += submat[bsub, i, j]
                    mat[b, modes.index(m)+len(modes), modes.index(n)+len(modes)] += submat[bsub, i, j]

            if submatlike_0:
                for i,m in enumerate(set(modes) - set(submodes)):
                    # +1 on the diagonal
                    mat[b, i, i] += 1



@njit
def mat_add(mat1, mat2, modes1, modes2, m1like_0, m2like_0):
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
def mat_add_vjp(dLdsum, B1, B2, modes1, modes2):
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
def mat_add_prep(modes1, modes2, m1like_0, m2like_0):
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
