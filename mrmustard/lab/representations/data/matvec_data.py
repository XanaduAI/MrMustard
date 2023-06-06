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
from numba import njit
from typing import List
from mrmustard.lab import Data
from mrmustard.math import Math
from mrmustard.typing import Batch, Matrix, Scalar, Vector

math = Math()


class MatVecData(Data):  # Note : this class is abstract too!
    def __init__(
        self, mat: Batch[Matrix], vec: Batch[Vector], coeffs: Batch[Scalar]
    ) -> None:
        self.mat = math.atleast_3d(mat)
        self.vec = math.atleast_2d(vec)
        self.coeffs = math.atleast_1d(coeffs)

    @property
    def batch_size(self) -> int:
        return self.coeffs.shape[0]


    #@njit
    def __neg__(self) -> MatVecData:
        return self.__class__(mat=-self.mat, vec=-self.vec, coeffs=self.coeffs)


    def __eq__(self, other: MatVecData, rtol: float = 1e-6, atol: float = 1e-6) -> bool:

        try: 
            return super().same(
                X = [self.mat, self.vec, self.coeffs],
                Y = [other.mat, other.vec, other.coeffs],
                rtol = rtol,
                atol = atol
                ) 

        except AttributeError as e:
            raise TypeError(
                f"Cannot compare {self.__class__} and {other.__class__}.") from e


    #@njit(parallel=True)
    def __add__(
        self,
        other: MatVecData,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        check_for_equality: bool = False,
        sub:bool = False # TODO : find a more elegant way to subtract! cf below
        ) -> MatVecData:
        
        if check_for_equality:

            if super().same(X=[self.mat, self.vec], 
                            Y=[other.mat, other.vec],
                            rtol=rtol, 
                            atol=atol
                            ):
                if sub: # TODO : find a more elegant way to subtract! cf above
                    return self.__class__(self.mat, self.vec, self.coeff - other.coeff)
                else:
                    return self.__class__(self.mat, self.vec, self.coeff + other.coeff)


        else:
            try: # TODO : make sure subtract is handled correctly in this case, nothing different?
                return self.__class__(
                    math.concat([self.mat, other.mat], axis=0),
                    math.concat([self.vec, other.vec], axis=0),
                    math.concat([self.coeffs, other.coeffs], axis=0),
                )

            except AttributeError as e:
                raise TypeError(f"Cannot add {self.__class__} and {other.__class__}.") from e



    #@njit(parallel=True)
    def __sub__(self, other: MatVecData, rtol: float = 1e-6, atol: float = 1e-6) -> MatVecData:
        return self.__add__(other=other, atol=atol, rtol=rtol)
        


    #@njit(parallel=True)
    def __and__(self, other: MatVecData) -> MatVecData:
        r"Tensor product"

        try: # TODO : decide which code we keep, old commented or new?
            # mat = []
            # vec = []
            # coeffs = []

            # for c1 in self.mat:
            #     for c2 in other.mat:
            #         mat.append(math.block_diag([c1, c2]))

            # for m1 in self.vec:
            #     for m2 in other.vec:
            #         vec.append(math.concat([m1, m2], axis=-1))

            # for c1 in self.coeffs:
            #     for c2 in other.coeffs:
            #         coeffs.append(c1 * c2)

            mat = [math.block_diag([c1, c2]) for c1 in self.mat for c2 in other.mat]
            vec = [math.concat([v1, v2], axis=-1) for v1 in self.vec for v2 in other.vec]
            coeffs = [c1 * c2 for c1 in self.coeffs for c2 in other.coeffs]

            return self.__class__(
                mat=math.astensor(mat),
                vec=math.astensor(vec),
                coeffs=math.astensor(coeffs),
            )

        except AttributeError as e:
            raise TypeError(f"Cannot tensor {self.__class__} and {other.__class__}.") from e


    # TODO: decide which simplify we want to keep cf below
    def simplify(self) -> None:  # TODO make this functional and return a new object
        r"""Simplify the data by combining terms that are equal."""

        indices_to_check = set(range(self.batch_size))
        removed = set()

        while indices_to_check:
            i = indices_to_check.pop()

            for j in indices_to_check.copy():
                if np.allclose(self.mat[i], self.mat[j]) and np.allclose(
                    self.vec[i], self.vec[j]
                ):
                    self.coeffs[i] += self.coeffs[j]
                    indices_to_check.remove(j)
                    removed.add(j)

        to_keep = [i for i in range(self.batch_size) if i not in removed]
        self.mat = self.mat[to_keep]
        self.vec = self.vec[to_keep]
        self.coeff = self.coeff[to_keep]


    # TODO: decide which simplify we want to keep
    #@njit(parallel=True)
    def fast_simplify(self, rtol:float=1e-6, atol:float=1e-6) -> MatVecData:
        N = self.mat.shape[0]
        mask = np.ones(N, dtype=np.int8)

        for i in range(N):
            for j in range(i + 1, N):
                if mask[i] == 0 or i == j:  # evaluated previously
                    continue

                if np.allclose(
                    self.mat[i], self.mat[j], rtol=rtol, atol=atol, equal_nan=True
                ) and np.allclose(
                    self.vec[i], self.vec[j], rtol=rtol, atol=atol, equal_nan=True
                ):
                    self.coeff[i] += self.coeff[j]
                    mask[j] = 0

        return self.__class__(
            mat=self.mat[mask == 1],
            vec=self.vec[mask == 1],
            coeffs=self.coeffs[mask == 1],
            )
