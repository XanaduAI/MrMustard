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

from mrmustard import settings
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
        assert (
            self.mat.shape[-1] == self.mat.shape[-2] == self.vec.shape[-1]
        ), "A and b must have the same dimension and A must be symmetric"
        self.batch_dim = self.mat.shape[0]
        self.dim = self.mat.shape[-1]

    def __neg__(self) -> MatVecData:
        return self.__class__(self.mat, self.vec, -self.coeffs)

    def __eq__(self, other: MatVecData, exclude_scalars: bool = False) -> bool:
        A, B = sorted([self, other], key=lambda x: x.batch_dim)  # A smaller or equal batch than B
        # check scalars
        Ac = np.around(A.coeffs, settings.EQUALITY_PRECISION_DECIMALS)
        Bc = memoryview(np.around(B.coeffs, settings.EQUALITY_PRECISION_DECIMALS)).tobytes()
        if exclude_scalars or all(memoryview(c).tobytes() in Bc for c in Ac):
            # check vectors
            Av = np.around(A.vec, settings.EQUALITY_PRECISION_DECIMALS)
            Bv = memoryview(np.around(B.vec, settings.EQUALITY_PRECISION_DECIMALS)).tobytes()
            if all(memoryview(v).tobytes() in Bv for v in Av):
                # check matrices
                Am = np.around(A.mat, settings.EQUALITY_PRECISION_DECIMALS)
                Bm = memoryview(np.around(B.mat, settings.EQUALITY_PRECISION_DECIMALS)).tobytes()
                if all(memoryview(m).tobytes() in Bm for m in Am):
                    return True
        return False

    def __add__(self, other: MatVecData) -> MatVecData:
        if self.__eq__(other, exclude_scalars=True):
            new_coeffs = self.coeffs + other.coeffs
            return self.__class__(self.mat, self.vec, new_coeffs)
        combined_matrices = math.concat([self.mat, other.mat], axis=0)
        combined_vectors = math.concat([self.vec, other.vec], axis=0)
        combined_coeffs = math.concat([self.coeffs, other.coeffs], axis=0)
        return self.__class__(combined_matrices, combined_vectors, combined_coeffs)

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
