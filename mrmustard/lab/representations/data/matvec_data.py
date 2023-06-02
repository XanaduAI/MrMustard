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

from mrmustard.representations.data import Data
from mrmustard.math import Math
from mrmustard.types import Batched, Matrix, Number, Scalar, Vector

from mrmustard.types

math = Math()

class MatVecData(Data):

    def __init__(self, mat: Batched[Matrix], vec: Batched[Vector], coeff: Batched[Scalar]):
        self.mat = math.atleast_3d(mat)
        self.vec = math.atleast_2d(vec)
        self.coeff = math.atleast_1d(coeff)

    
    @property
    def batch_size(self) -> int:
        return self.coeff.shape[0]


    def __eq__(self, other: MatVecData) -> bool:
        return (
            np.allclose(self.mat, other.mat)
            and np.allclose(self.vec, other.vec)
            and np.allclose(self.coeff, other.coeff)
        )


    def __add__(self, other: MatVecData) -> MatVecData:
        if self.__class__ != other.__class__:
            raise ValueError(f"Cannot add {self.__class__} and {other.__class__}.")
        
        elif np.allclose(self.mat, other.mat) and np.allclose(self.vec, other.vec):
            return MatVecData(self.mat, self.vec, self.coeff + other.coeff)
        
        else:
            return MatVecData(
                math.concat([self.mat, other.mat], axis=0),
                math.concat([self.vec, other.vec], axis=0),
                math.concat([self.coeff, other.coeff], axis=0),
            )


    @abstractmethod
    def __sub__():
        pass


    @abstractmethod
    def __neg__():
        pass


    @abstractmethod
    def __and__():
        pass


    def simplify(self) -> None:
        r"""Simplify the data by combining terms that are equal."""
        indices_to_check = set(range(self.batch_size)) # TODO switch to lists?
        removed = set()
        while indices_to_check:
            i = indices_to_check.pop()
            for j in indices_to_check.copy():
                if np.allclose(self.mat[i], self.mat[j]) and np.allclose(self.vec[i], self.vec[j]):
                    self.coeff[i] += self.coeff[j]
                    indices_to_check.remove(j)
                    removed.add(j)
        to_keep = [i for i in range(self.batch_size) if i not in removed] # TODO replace by filter or set substraction?
        self.mat = self.mat[to_keep]
        self.vec = self.vec[to_keep]
        self.coeff = self.coeff[to_keep]


    
    
