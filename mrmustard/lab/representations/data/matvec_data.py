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

from typing import List, Optional, Set, Tuple, Union

import numpy as np

from mrmustard.lab.representations.data.data import Data
from mrmustard.math import Math
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
        assert (
            len(self.mat) == len(self.vec) == len(self.coeffs)
        ), "All inputs must have the same batch size."
        self.batch_dim = self.mat.shape[0]
        self.dim = self.mat.shape[-1]

    def __neg__(self) -> MatVecData:
        return self.__class__(self.mat, self.vec, -self.coeffs)

    def __eq__(self, other: MatVecData, include_scalars: bool = True) -> bool:
        A, B = sorted([self, other], key=lambda x: x.batch_dim)  # A smaller or equal batch than B
        try:
            if include_scalars and all(
                memoryview(c).tobytes() in memoryview(B.coeffs).tobytes() for c in A.coeffs
            ):
                if all(memoryview(v).tobytes() in memoryview(B.vec).tobytes() for v in A.vec):
                    if all(memoryview(m).tobytes() in memoryview(B.mat).tobytes() for m in A.mat):
                        return True
            return False
        except Exception as e:
            raise TypeError(f"Cannot compare {self.__class__} and {other.__class__}.") from e

    def __add__(self, other: MatVecData) -> MatVecData:
        try:
            combined_matrices = math.concat([self.mat, other.mat], axis=0)
            combined_vectors = math.concat([self.vec, other.vec], axis=0)
            combined_coeffs = math.concat([self.coeffs, other.coeffs], axis=0)
            return self.__class__(combined_matrices, combined_vectors, combined_coeffs)
        except Exception as e:
            raise TypeError(f"Cannot add/subtract {self.__class__} and {other.__class__}.") from e

    def __truediv__(self, x: Scalar) -> MatVecData:
        if not isinstance(x, (int, float, complex)):
            raise TypeError(f"Cannot divide {self.__class__} by {x.__class__}.")
        new_coeffs = self.coeffs / x
        return self.__class__(self.mat, self.vec, new_coeffs)

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
