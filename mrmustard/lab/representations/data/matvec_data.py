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

from __future__ import annotations

import numpy as np

from typing import Set, Tuple

from mrmustard.lab.representations.data.data import Data
from mrmustard.math import Math
from mrmustard.physics.gaussian import reorder_matrix_from_qpqp_to_qqpp
from mrmustard.typing import Batch, Matrix, Scalar, Vector

math = Math()


class MatVecData(Data):  # Note: this class is abstract!
    r"""Contains matrix and vector -like data for certain Representation objects.

    Args:
        mat (Batch[Matrix]): the matrix-like data to be contained in the class
        vec (Batch[Vector]):    the vector-like data to be contained in the class
        coeffs (Batch[Scalar]): the coefficients
    """

    def __init__(self, mat: Batch[Matrix], vec: Batch[Vector], coeffs: Batch[Scalar]) -> None:
        # TODO: check that dimensions make sense, aka mat is NxDxD, vecs are NxD and coeffs are N
        if coeffs is None: #default cs should all be 1
            try:
                n = vec.shape[0] # number of elements if it's a numpy array
            except AttributeError:
                n = len(vec) #if it's a list
            coeffs = math.astensor(np.repeat(1.0, n))

        self.mat = mat
        self.vec = vec
        self.coeffs = coeffs

    def __neg__(self) -> MatVecData:
        new_coeffs = []
        for c in self.coeffs:
            new_coeffs.append(-c)
        return self.__class__(self.mat, self.vec, new_coeffs)

    def __eq__(self, other: MatVecData, ignore_scalars:bool=False) -> bool:
        try:
            answer = False
            if ignore_scalars: # objects have the same matrices and vectors
                if self._helper_vecs_or_mats_are_same(self.vec, other.vec):
                    if self._helper_vecs_or_mats_are_same(self.mat, other.mat):
                        answer = True
            else: # compare everything including scalars
                if self._helper_scalars_are_same(self.coeffs, other.coeffs):
                    if self._helper_vecs_or_mats_are_same(self.vec, other.vec):
                        if self._helper_vecs_or_mats_are_same(self.mat, other.mat):
                            answer = True
            return answer
        except AttributeError as e:
            raise TypeError(f"Cannot compare {self.__class__} and {other.__class__}.") from e

    def __add__(self, other: MatVecData) -> MatVecData:
        try:
            if self.__eq__(other, ignore_scalars=True):
                combined_coeffs = self.coeffs + other.coeffs
                return self.__class__(self.mat, self.vec, combined_coeffs)

            else:
                mat = math.concat([self.mat, other.mat], axis=0)
                vec = math.concat([self.vec, other.vec], axis=0)
                reorder_matrix = reorder_matrix_from_qpqp_to_qqpp(self.mat.shape[-1])
                mat = math.matmul(math.matmul(reorder_matrix, mat), math.transpose(reorder_matrix))
                vec = math.matvec(reorder_matrix, vec)
                combined_coeffs = math.concat([self.coeffs, other.coeffs], axis=0)
                return self.__class__(mat, vec, combined_coeffs)

        except AttributeError as e:
            raise TypeError(f"Cannot add/subtract {self.__class__} and {other.__class__}.") from e

    def __truediv__(self, x: Scalar) -> MatVecData:
        new_coeffs = self.coeffs / x
        return self.__class__(self.mat, self.vec, new_coeffs)

    def helper_check_is_real_symmetric(self, A:Batch[Matrix]) -> bool:
        r"""Checks that the matrix given is both real and symmetric."""
        return all([np.allclose(a, np.transpose(a)) for a in A])
    

    def _helper_vecs_or_mats_are_same(self, tensors_a, tensors_b, precision=3) -> bool:
        f = lambda x : np.linalg.norm(x)
        norms_a = [f(a) for a in tensors_a]
        norms_b = [f(b) for b in tensors_b]
        return self._helper_scalars_are_same(norms_a, norms_b, precision)

    def _helper_scalars_are_same(self, a, b, precision=3) -> bool:
        A, B = self._helper_to_sets(a, b, precision)
        return A.symmetric_difference(B) ==  set()

    @staticmethod
    def _helper_to_sets(a, b, precision) -> Tuple[Set,Set]:
        A = np.around(a, precision)
        B = np.around(b, precision)
        set_A = set(A)
        set_B = set(B)
        return (set_A, set_B)

    # def __and__(self, other: MatVecData) -> MatVecData:
    #     try: #TODO: ORDER OF ALL MATRICESA!
    #         mat = [math.block_diag([c1, c2]) for c1 in self.mat for c2 in other.mat]
    #         vec = [math.concat([v1, v2], axis= -1) for v1 in self.vec for v2 in other.vec]
    #         coeffs = [c1 * c2 for c1 in self.coeffs for c2 in other.coeffs]

    #         return self.__class__(math.astensor(mat), math.astensor(vec), math.astensor(coeffs))

    #     except AttributeError as e:
    #         raise TypeError(f"Cannot tensor {self.__class__} and {other.__class__}.") from e

    # # TODO: decide which simplify we want to keep
    # def simplify(self, rtol:float=1e-6, atol:float=1e-6) -> MatVecData:
    #     N = self.mat.shape[0]
    #     mask = np.ones(N, dtype=np.int8)

    #     for i in range(N):

    #         for j in range(i + 1, N):

    #             if mask[i] == 0 or i == j:  # evaluated previously
    #                 continue

    #             if np.allclose(
    #                 self.mat[i], self.mat[j], rtol=rtol, atol=atol, equal_nan=True
    #             ) and np.allclose(
    #                 self.vec[i], self.vec[j], rtol=rtol, atol=atol, equal_nan=True
    #             ):
    #                 self.coeffs[i] += self.coeffs[j]
    #                 mask[j] = 0

    #     return self.__class__(
    #         mat = self.mat[mask == 1],
    #         vec = self.vec[mask == 1],
    #         coeffs = self.coeffs[mask == 1]
    #         )

    # # TODO: decide which simplify we want to keep
    # def old_simplify(self) -> None:
    #     indices_to_check = set(range(self.batch_size))
    #     removed = set()

    #     while indices_to_check:
    #         i = indices_to_check.pop()

    #         for j in indices_to_check.copy():
    #             if np.allclose(self.mat[i], self.mat[j]) and np.allclose(
    #                 self.vec[i], self.vec[j]
    #             ):
    #                 self.coeffs[i] += self.coeffs[j]
    #                 indices_to_check.remove(j)
    #                 removed.add(j)

    #     to_keep = [i for i in range(self.batch_size) if i not in removed]
    #     self.mat = self.mat[to_keep]
    #     self.vec = self.vec[to_keep]
    #     self.coeffs = self.coeffs[to_keep]
