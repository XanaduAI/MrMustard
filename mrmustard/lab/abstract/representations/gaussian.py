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

from mrmustard.math import Math
math = Math()

from .representation import Representation



class Gaussian:

    def __init__(self, cov: Array, mean: Array, val=math.astensor([1.0])):
        self.cov = cov if cov.ndim == 3 else cov[None,...]
        self.mean = mean if mean.ndim == 2 else mean[None,...]
        self.val = math.atleast_1d(val)

    def __add__(self, other):
        if not isinstance(other, Gaussian):
            raise ValueError("Can only add a Gaussian to a Gaussian")
        if self.cov.shape != other.cov.shape:
            raise ValueError("Cannot add Gaussians of different shape")
        return Gaussian(cov=math.concat(self.cov, other.cov, axis=0),
                        means=math.concat(self.mean, other.mean, axis=0),
                        val=math.concat(self.val, other.val, axis=0))

    def __mul__(self, other):
        if not isinstance(other, Number):
            raise ValueError("Can only multiply a Gaussian by a number")
        return Gaussian(cov=self.cov, mean=self.mean, val=self.val * other)

    def __sub__(self, other):
        return self + (-1 * other)

    def __rmul__(self, other):
        return self * other

    def __repr__(self):
        return f"{self.__class__.__qualname__} | bosonic rank = {len(self.val)}"

    
class Characteristic(Representation, Gaussian, FunctionArray):

    def __init__(self, cov=None, mean=None, val=None, array=None):
        if cov is None and mean is None and val is None and array is not None:
            super().__init__(array)
        else:
            self.cov = self.data
            self.mean = mean
            self.val = val


    @staticmethod
    def from_characteristic_gaussian(characteristic):
        return Characteristic(characteristic.covariance, characteristic.mean)
    
    @staticmethod
    def from_wigner_gaussian(wigner):
        return Characteristic(math.inv(wigner.covariance), wigner.mean)
        
class Wigner(Representation):

    @staticmethod
    def from_wigner_gaussian(wigner):
        return Wigner(wigner.covariance, wigner.mean)

    @staticmethod
    def from_characteristic_gaussian(characteristic):
        return Wigner(math.inv(characteristic.covariance), characteristic.mean)

class Husimi(Representation):

    @staticmethod
    def from_husimi_gaussian(husimi):
        return Husimi(husimi.covariance, husimi.mean)

    @staticmethod
    def from_characteristic_gaussian(characteristic):
        cov = characteristic.covariance
        return Husimi(cov + math.identity(cov.shape[-1])/2, characteristic.mean)

class Bargmann(Representation):

    @staticmethod
    def from_bargmann_gaussian(bargmann):
        return Bargmann(bargmann.covariance, bargmann.mean, bargmann.scalar)

    @staticmethod
    def from_characteristic_gaussian(characteristic):
        Q = Husimi.from_characteristic_gaussian(characteristic).covariance
        Q_inv =  math.inv(Q)
        return Bargmann(X @ (math.identity(Q.shape[-1]) - Q_inv), X @ Q_inv @ characteristic.mean)

    @staticmethod
    def from_husimi_gaussian(husimi):
        Q_inv =  math.inv(husimi.covariance)
        return Bargmann(X @ (math.identity(Q_inv.shape[-1]) - Q_inv), X @ Q_inv @ husimi.mean)


class Position(Representation):

    @staticmethod
    def from_characteristic_gaussian(characteristic):
        return Position(characteristic.covariance, characteristic.mean)


class Gaussian:

    def __init__(self, covariance, mean, scalar=1):
        self.covariance = covariance
        self.mean = mean
        self.scalar = scalar