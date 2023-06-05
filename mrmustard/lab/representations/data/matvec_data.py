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

import numpy as np
from numba import njit
from mrmustard.representations.data import Data
from mrmustard.math import Math
from mrmustard.typing import Batched, Matrix, Scalar, Vector

math = Math()

__all__ = [MatVecData]

class MatVecData(Data): # Note : this class is abstract too!

    def __init__(self, mat: Batched[Matrix], vec: Batched[Vector], coeffs: Batched[Scalar]):
        self.mat = math.atleast_3d(mat)
        self.vec = math.atleast_2d(vec)
        self.coeffs = math.atleast_1d(coeffs)


    
    @property
    def batch_size(self) -> int:
        return self.coeffs.shape[0]
    


    @njit
    def __neg__(self) -> MatVecData:
        return self.__class__(mat = -self.mat, vec = -self.vec)



    def __eq__(self, other: MatVecData, rtol:float=1e-6, atol:float=1e-6) -> bool:
        # TODO : this could actually be implemented in parent class with a list of the things to compare
        
        try:
            return (
                np.allclose(self.mat, other.mat, rtol=rtol, atol=atol)
                and np.allclose(self.vec, other.vec, rtol=rtol, atol=atol)
                and np.allclose(self.coeffs, other.coeffs, rtol=rtol, atol=atol)
            )
    
        except AttributeError as e:
            raise TypeError(f"Cannot compare {self.__class__} and {other.__class__}.") from e



# TODO : find a smart way to do add/sub without repeating code and optimising for common case
    @njit(parallel=True)
    def __add__(self, other: MatVecData, rtol:float=1e-6, atol:float=1e-6) -> MatVecData:

        try:

            if (np.allclose(self.mat, other.mat, rtol=rtol, atol=atol) 
                and np.allclose(self.vec, other.vec, rtol=rtol, atol=atol)):

                return self.__class__(self.mat, self.vec, self.coeff + other.coeff)
            
            else:
                return self.__class__(
                    math.concat([self.mat, other.mat], axis=0),
                    math.concat([self.vec, other.vec], axis=0),
                    math.concat([self.coeffs, other.coeffs], axis=0)
                )
            
        except AttributeError as e:
            raise TypeError(f"Cannot add {self.__class__} and {other.__class__}.") from e



    @njit(parallel=True)
    def __sub__(self, other: MatVecData, rtol:float=1e-6, atol:float=1e-6) -> MatVecData:
       
       try:

            if (np.allclose(self.mat, other.mat, rtol=rtol, atol=atol) 
                and np.allclose(self.vec, other.vec, rtol=rtol, atol=atol)):

                return self.__class__(self.mat, self.vec, self.coeffs - other.coeffs)
            
            else:
                return self.__class__(
                    mat = math.concat([self.mat, other.mat], axis=0),
                    vec = math.concat([self.vec, other.vec], axis=0),
                    coeffs = math.concat([self.coeffs, other.coeffs], axis=0)
                )
            
       except AttributeError as e:
           raise TypeError(f"Cannot substract {self.__class__} and {other.__class__}.") from e



    @njit(parallel=True)
    def __and__(self, other: MatVecData) -> MatVecData:
        r"Tensor product"

        try:

            mat = []
            vec = []
            coeffs = []

            for c1 in self.mat:
                for c2 in other.mat:
                    mat.append(math.block_diag([c1, c2]))

            for m1 in self.vec:
                for m2 in other.vec:
                    vec.append(math.concat([m1, m2], axis=-1))

            for c1 in self.coeffs:
                for c2 in other.coeffs:
                    coeffs.append(c1 * c2)

            return self.__class__(mat = math.astensor(mat),
                                  vec = math.astensor(vec),
                                  coeffs = math.astensor(coeffs))
        
        except AttributeError as e:
            raise TypeError(f"Cannot tensor {self.__class__} and {other.__class__}.") from e



    # TODO: decide which simplify we want to keep cf below
    def simplify(self) -> None: # TODO make this functional and return a new object
        r"""Simplify the data by combining terms that are equal."""

        indices_to_check = set(range(self.batch_size)) # TODO switch to lists???
        removed = set()

        while indices_to_check:
            i = indices_to_check.pop()

            for j in indices_to_check.copy():

                if np.allclose(self.mat[i], self.mat[j]) and np.allclose(self.vec[i], self.vec[j]):
                    self.coeff[i] += self.coeff[j]
                    indices_to_check.remove(j)
                    removed.add(j)

        to_keep = [i for i in range(self.batch_size) if i not in removed]
        self.mat = self.mat[to_keep]
        self.vec = self.vec[to_keep]
        self.coeff = self.coeff[to_keep]



    # TODO: decide which simplify we want to keep
    @njit(parallel=True)
    def fast_simplify(self, rtol=1e-6, atol=1e-6) -> MatVecData:
        
        N = self.mat.shape[0]
        mask = np.ones(N, dtype=np.int8)

        for i in range(N):

            for j in range(i+1,N):

                if mask[i] == 0 or i == j: #evaluated previously
                    continue

                if ( np.allclose(self.mat[i], self.mat[j], rtol=rtol, atol=atol, equal_nan=True) 
                 and np.allclose(self.vec[i], self.vec[j], rtol=rtol, atol=atol, equal_nan=True)
                   ):
                    self.coeff[i] += self.coeff[j]
                    mask[j] = 0

        return self.__class__(mat=self.mat[mask==1], vec=self.vec[mask==1], coeff=self.coeff[mask==1])

        
        
        


    
    
