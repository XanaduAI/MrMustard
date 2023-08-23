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

from typing import List, Optional, Set, Tuple, Union

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
        if coeffs is None:  # default all 1s
            coeffs = math.ones(len(vec), dtype=math.float64)

        self.mat = math.atleast_3d(math.astensor(mat))
        self.vec = math.atleast_2d(math.astensor(vec))
        self.coeffs = math.atleast_1d(math.astensor(coeffs))

    def __neg__(self) -> MatVecData:
        return self.__class__(self.mat, self.vec, -self.coeffs)

    def __eq__(self, other: MatVecData, ignore_scalars: bool = False) -> bool:
        try:
            answer = False
            if ignore_scalars:  # objects have the same matrices and vectors
                if self._helper_vecs_are_same(self.vec, other.vec):
                    if self._helper_mats_are_same(self.mat, other.mat):
                        answer = True
            else:  # compare everything including scalars
                if self._helper_scalars_are_same(self.coeffs, other.coeffs):
                    if self._helper_vecs_are_same(self.vec, other.vec):
                        if self._helper_mats_are_same(self.mat, other.mat):
                            answer = True
            return answer
        except AttributeError as e:
            raise TypeError(f"Cannot compare {self.__class__} and {other.__class__}.") from e

    def __add__(self, other: MatVecData) -> MatVecData:
        try:
            if self.__eq__(other, ignore_scalars=True):
                # sorting and re-ordering necessary so the correct coeffs are paired up
                # (because equality doesn't guarantee anything in terms of order)
                new_ms, new_vs, new_cs = self._helper_make_new_object_params_for_add_sub(other)
                return self.__class__(new_ms, new_vs, new_cs)

            else:  # note that in subtract the coefficients were made negative beforehand so it's ok!
                combined_matrices = math.concat([self.mat, other.mat], axis=0)
                combined_vectors = math.concat([self.vec, other.vec], axis=0)
                combined_coeffs = math.concat([self.coeffs, other.coeffs], axis=0)
                return self.__class__(combined_matrices, combined_vectors, combined_coeffs)

        except AttributeError as e:
            raise TypeError(f"Cannot add/subtract {self.__class__} and {other.__class__}.") from e

    def __truediv__(self, x: Scalar) -> MatVecData:
        if not isinstance(x, (int, float, complex)):
            raise TypeError(f"Cannot divide {self.__class__} by {x.__class__}.")
        new_coeffs = self.coeffs / x
        return self.__class__(self.mat, self.vec, new_coeffs)

    def _helper_check_is_symmetric(self, M: Batch[Matrix]) -> bool:
        r"""Checks that the matrices in the given batch are symmetric.

        Args:
            M (Batch[Matrix]):  the batch of matrices to be examined

        Returns:
            (bool): True if all matrices in the batch are symmetric, False otherwise.

        """
        return np.allclose(M, np.transpose(M, (0, 2, 1)))

    def _helper_mats_are_same(
        self,
        mats_a: List[Matrix],
        mats_b: List[Matrix],
        precision: Optional[int] = 3,
    ) -> bool:
        r"""Determines whether the two sets of tensors given are the same up to precision.

        Given 2 lists of matrices or vectors, determines whether they are the same (based on
        norm) up to thye given precision. Order is irrelevant and permutations of a set of elements
        all evaluate to True.

        Note: there is one caveat to the way this equality is evaluated. Since the process computes
        the norms of all the tensors and then builds a set out of those elements, it means that any
        two norms with the same values up to precision will be stored only a single time. If both
        tensors have the same pair of tensors sharing a value this does not matter. However, if one
        of the tensors has more elements sharing the same norm than the other, this will be
        identified as both elements being the same (despite it not being the case). Our current bet
        is that this should happen seldom enough for it to not be problematic, but future
        developments should address this issue.
        Advice for next steps: checking the difference between the cardinal of the set and the
        length of the list (this gives how many items were mrophed into a single one in the passage
        to set). Compare this for both tensors, if they are the same it's more likely that they're
        the same, but this is still not a guarantee.

        Args:
            mats_a (Union[List[Matrix], List[Vector]])   : a list of either matrices or vectors
            mats_b (Union[List[Matrix], List[Vector]])   : a list of either matrices or vectors
            precision (Optional[int]):                      : the number of decimals to which to
                                                            round the resulting scalar, default
                                                            value is 3

        Returns:
            (bool) : True if both tensors have the same norms, up to precision, false otherwise.

        """
        norms_a = np.linalg.norm(mats_a, axis=(1, 2))
        norms_b = np.linalg.norm(mats_b, axis=(1, 2))
        return self._helper_scalars_are_same(norms_a, norms_b, precision)

    def _helper_vecs_are_same(
        self, vecs_a: List[Vector], vecs_b: List[Vector], precision: Optional[int] = 3
    ) -> bool:
        r"""Given 2 lists of matrices or vectors, determines whether they are the same (based on
        norm) up to thye given precision. Order is irrelevant and permutations of a set of elements
          all evaluate to True."""
        norms_a = np.linalg.norm(vecs_a, axis=1)
        norms_b = np.linalg.norm(vecs_b, axis=1)
        return self._helper_scalars_are_same(norms_a, norms_b, precision)

    def _helper_scalars_are_same(
        self, a: List[Scalar], b: List[Scalar], precision: Optional[int] = 3
    ) -> bool:
        r"""Given 2 lists of scalar, determines whether they are the same, up to precision. Order
        is irrelevant and permutations of a set of elements all evaluate to True."""
        A = np.around(a, precision)
        B = np.around(b, precision)
        return not set(A).isdisjoint(B)

    def _helper_make_new_object_params_for_add_sub(
        self, other: MatVecData
    ) -> Tuple[List[Matrix], List[Vector], List[Scalar]]:
        r"""Generates the new parameters for the object after addition/subtraction."""
        N = len(self.coeffs)
        sorted_tups_self = self._helper_make_sorted_list_of_tuples(self)
        sorted_tups_other = self._helper_make_sorted_list_of_tuples(other)

        new_mats = []
        new_vecs = []
        new_coeffs = []
        for i in range(N):
            new_mats.append(sorted_tups_self[i][0])
            new_vecs.append(sorted_tups_self[i][1])
            ith_coeff = sorted_tups_self[i][2] + sorted_tups_other[i][2]
            new_coeffs.append(ith_coeff)
        return (np.array(new_mats), np.array(new_vecs), np.array(new_coeffs))

    @staticmethod
    def _helper_make_sorted_list_of_tuples(obj: MatVecData) -> List[Tuple[Matrix, Vector, Scalar]]:
        r"""Given a MatVecData object, returns the list of tuples made by
        (mat[i], vec[i], coeffs[i])."""
        all_tuples = list(zip(obj.mat, obj.vec, obj.coeffs))
        all_tuples.sort(key=lambda x: np.linalg.norm(x[0]))
        return all_tuples

    # def __and__(self, other: MatVecData) -> MatVecData:
    #     try: #TODO: ORDER OF ALL MATRICES!
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
