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

"""This file contains batched, sparse and mode-wise vector and matrix addition functions
 needed by methods in the :class:`Math` class."""

import numpy as np
from numba import njit
from mrmustard.types import *


@njit
def numba_vec_add(vec1, vec2, modes1, modes2):
    r"""Mode-wise sparse addition of vec1 and vec2, with outer product on the batch dimension.
    
        Args:
            vec1 (array): A vector of shape (B1, M, 2) that lives on `modes1`
            vec2 (array): A vector of shape (B2, N, 2) that lives on `modes2`
            modes1 (array): A vector of shape (M,) that contains the modes of `vec1`
            modes2 (array): A vector of shape (N,) that contains the modes of `vec2`
        
        Returns:
            array: A "vector" of shape (B1*B2, U, 2) that lives on the union of `modes1` and `modes2`
    """
    s1 = set(modes1)
    s2 = set(modes2)
    union = s1.union(s2)

    if s2.issubset(s1):
        vec = numba_vec_add_inplace(vec1, vec2, modes1, modes2)
    elif s1.issubset(s2):
        vec = numba_vec_add_inplace(vec2, vec1, modes2, modes1)
    else:
        vec = np.zeros((1, len(union), 2), dtype=vec1.dtype)  # NOTE: add_inplace takes care of the batch dimension
        vec = numba_vec_add_inplace(vec, vec1, [m for m in union], modes1)
        vec = numba_vec_add_inplace(vec, vec2, [m for m in union], modes2)

    return vec


@njit
def numba_vec_add_inplace(vec, subvec, modes, submodes):
    r"""Mode-wise sparse addition of vec_sub to vec, with outer product on the batch dimension.
    It assumes submodes are a subset of modes, hence we can just add to vec without creating a new array.
    Warning: it updates vec inplace.
    
        Args:
            vec (array): :math:`B1\times M\times 2` batched 'large' vector
            subvec (array): :math:`B2\times N\times 2` batched 'small' vector
            modes (list): list of modes in vec
            submodes (list): list of modes in subvec, must be a subset of modes
        Returns:
            array: :math:`(B1 B2)\times M\times2` batched subvec with added vec
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
            b = b1*B2 + b2  # TODO: is this giving the opposite vectorization order when vec1 and vec2 are swapped? test this case.
            for i,m in enumerate(submodes):
                vec[b, modes_index[i]] += subvec[b2, i]
    return vec


@njit
def numba_mat_add(mat1, mat2, modes1, modes2, m1like_0, m2like_0):
    r"""Mode-wise sparse addition of `mat1` and `mat2`, with outer product on the batch dimension.
    Warning: it may update `mat1` or `mat2` inplace.

        Args:
            mat1 (array): A matrix of shape (B1, M, M, 2, 2) that lives on `modes1`
            mat2 (array): A matrix of shape (B2, N, N, 2, 2) that lives on `modes2`
            modes1 (array): A vector of shape (M,) that contains the modes of `mat1`
            modes2 (array): A vector of shape (N,) that contains the modes of `mat2`
            m1like_0 (bool): If True, mat1 is assumed to be like 0 on modes not in modes1
            m2like_0 (bool): If True, mat2 is assumed to be like 0 on modes not in modes2

        Returns:
            array: A matrix of shape (B1*B2, U, U, 2, 2) that lives on the union of `modes1` and `modes2`
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
        mat = np.zeros((1, len(union), len(union), 2, 2), dtype=mat1.dtype) # NOTE: add_inplace takes care of the batch dimension
        mat = numba_mat_add_inplace(mat, mat1, union_list, modes1, m1like_0)
        mat = numba_mat_add_inplace(mat, mat2, union_list, modes2, m2like_0)
    return mat


@njit
def numba_mat_add_inplace(mat, submat, modes, submodes, submatlike_0):
    r"""Mode-wise sparse addition of mat_sub to mat, with outer product on the batch dimension.
    It assumes submodes are a subset of modes, hence we can just add to mat without creating a new array.
    Warning: it updates `mat` inplace.

        Args:
            mat (array): :math:`B1\times M\times M \times 2 \times 2` batched matrix living in `modes`
            submat (array): :math:`B2\times N\times N \times 2 \times 2` batched submatrix living in `submodes`
            modes (list): list of modes in mat
            submodes (list): list of modes in submat, must be a subset of modes
            matlike_0 (bool): whether mat is a like the zero matrix or the identity matrix on unspecified modes
            submatlike_0 (bool): whether submat is a like the zero matrix or the identity matrix on unspecified modes
        
        Returns:
            array: :math:`(B1 B2)\times F\times F \times 2 \times 2` result respecting the effect on unspecified modes and the outer product on the batch dimension
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
    if not submatlike_0:
        other_modes = set(modes) - set(submodes)
        other_modes_index = [modes.index(m) for m in other_modes]
    for b_ in range(B):
        for bsub in range(Bsub):
            b = b_*Bsub + bsub # vectorize
            for i,m in enumerate(submodes):
                for j,n in enumerate(submodes):
                    mat[b, modes_index[i], modes_index[j]] += submat[bsub, i, j]
            if not submatlike_0:
                for i,m in enumerate(other_modes):
                    # +1 on the diagonal
                    mat[b, other_modes_index[i], other_modes_index[i], 0, 0] += 1
                    mat[b, other_modes_index[i], other_modes_index[i], 1, 1] += 1
    return mat


########################################################################
############################### GRADIENTS ##############################
########################################################################


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
            dL_dsum (array): :math:`(B1 B2)\times M\times 2` batched derivative of loss with respect to sum
            vec (array): :math:`B1\times M\times 2` batched large vector
            subvec (array): :math:`B2\times N\times 2` batched small vector
            modes (list): list of modes in vec
            submodes (list): list of modes in vec_sub, must be a subset of modes1
        Returns:
            dL_dvec (array): :math:`B1\times M\times 2` batched derivative of L wrt vec
            dL_dsubvec (array): :math:`B2\times N\times 2` batched derivative of L wrt subvec
    """
    dL_dvec = np.zeros((B, len(modes), 2), dtype=dL_sum.dtype)
    dL_dsubvec = np.zeros((Bsub, len(submodes), 2), dtype=dL_sum.dtype)
    modes_index = [modes.index(m) for m in submodes]
    for b1 in range(B):
        for b2 in range(Bsub):
            b = b1*B2 + b2
            for i,m in enumerate(submodes):
                dL_dvec[b1, modes_index[i]] += dL_sum[b, i]
                dL_dsubvec[b2, i] += dL_sum[b, modes_index[i]]
    return dL_dvec, dL_dsubvec



@njit
def numba_mat_add_vjp(dLdsum, mat1, mat2, modes1, modes2):
    r"""vjp of mode-wise sparse addition of mat1 to mat2.

        Args:
            dLdsum (array): :math:`(B1 B2)\times M\times M \times 2 \times 2` batched derivative of loss with respect to sum
            mat1 (array): :math:`B1\times M\times M \times 2 \times 2` batched first matrix
            mat2 (array): :math:`B2\times N\times N \times 2 \times 2` batched second matrix
            modes (list): list of modes in mat
            submodes (list): list of modes in submat, must be a subset of modes1
        Returns:
            dL_dmat (array): :math:`(B1 B2)\times M\times M \times 2 \times 2` batched derivative of L wrt mat
            dL_dsubmat (array): :math:`(B1 B2)\times N\times N \times 2 \times 2` batched derivative of L wrt submat
    """
    s1 = set(modes1)
    s2 = set(modes2)
    union = s1.union(s2)
    
    if s2.issubset(s1):
        dL_dmat1, dL_dmat2 = numba_mat_add_inplace_vjp(dLdsum, len(mat1), len(mat2), modes1, modes2)
    elif s1.issubset(s2):
        dL_dmat2, dL_dmat1 = numba_mat_add_inplace_vjp(dLdsum, len(mat2), len(mat1), modes2, modes1)
    else:
        _, dL_dmat1 = numba_mat_add_inplace_vjp(dLdsum, len(dLdsum), len(mat1), [m for m in union], modes1)
        _, dL_dmat2 = numba_mat_add_inplace_vjp(dLdsum, len(dLdsum), len(mat2), [m for m in union], modes2)
        return dL_dmat1, dL_dmat2


@njit
def numba_mat_add_inplace_vjp(dLdsum, B, Bsub, modes, submodes):
    r"""vjp of mode-wise sparse addition of mat_sub to mat.
    
        Args:
            dL_dsum (array): :math:`(B1 B2)\times M\times M\times 2\times 2` batched derivative of loss with respect to sum
            mat (array): :math:`B1\times M\times M \times 2\times 2` batched large matrix
            submat (array): :math:`B2\times N\times N\times 2\times 2` batched small matrix
            modes (list): list of modes in mat
            submodes (list): list of modes in mat_sub, must be a subset of modes1
        Returns:
            dL_dmat (array): :math:`B1\times M\times M\times 2\times 2` batched derivative of L wrt mat
            dL_dsubmat (array): :math:`B2\times N\times N\times 2\times 2` batched derivative of L wrt submat
    """
    dL_dmat = np.zeros((B, len(modes), len(modes), 2, 2), dtype=dLdsum.dtype)
    dL_dsubmat = np.zeros((Bsub, len(submodes), len(submodes), 2, 2), dtype=dLdsum.dtype)
    modes_index = [modes.index(m) for m in submodes]
    for b1 in range(B):
        for b2 in range(Bsub):
            b = b1*B2 + b2
            for i,m in enumerate(submodes):
                for j,n in enumerate(submodes):
                    dL_dmat[b1, modes_index[i], modes_index[j]] += dLdsum[b, i, j]
                    dL_dsubmat[b2, i, j] += dLdsum[b, modes_index[i], modes_index[j]]
    return dL_dmat, dL_dsubmat
