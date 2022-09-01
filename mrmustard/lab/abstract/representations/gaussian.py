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

"""Gaussian representation of quantum states, i.e. in terms of a covariance matrix and a mean vector."""

from .representation import Representation

class Characteristic(Representation):
    def from_wigner_gaussian(self, wigner):
        return Gaussian(math.inv(wigner.covariance), wigner.mean)
        
class Wigner(Representation):
    def from_characteristic_gaussian(self, characteristic):
        return Gaussian(math.inv(characteristic.covariance), characteristic.mean)

class Husimi(Representation):
    def from_characteristic_gaussian(self, characteristic):
        cov = characteristic.covariance
        return Gaussian(cov + math.identity(cov.shape[-1])/2, characteristic.mean)

class Bargmann(Representation):
    def from_characteristic_gaussian(self, characteristic):
        Q = Husimi.from_characteristic_gaussian(characteristic).covariance
        Q_inv =  math.inv(Q)
        return Gaussian(X @ (math.identity(Q.shape[-1]) - Q_inv), X @ Q_inv @ characteristic.mean)

    def from_husimi_gaussian(self, husimi):
        Q_inv =  math.inv(husimi.covariance)
        return Gaussian(X @ (math.identity(Q_inv.shape[-1]) - Q_inv), X @ Q_inv @ husimi.mean)

class Gaussian:

    def __init__(self, matrix, vector, scalar, coefficients = [1.0]):
        """Vector of Gaussian objects in any representation.

        Args:
            matrix (Array): the batched covariance matrix
            vector (Array): the batched mean vector
            scalar (float): the scalar prefactor
            coefficients (Array): the coefficients of the 'linear combination' of Gaussian objects
        """
        self.matrix = matrix
        self.vector = vector
        self.scalar = scalar
        self.coefficients = coefficients



    def __repr__(self):
        return f"{self.__class__.__qualname__} | matrix.shape = {self.matrix.shape} | vector.shape = {self.vector.shape}"