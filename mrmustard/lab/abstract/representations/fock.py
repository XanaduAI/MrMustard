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

"""Ket representation of quantum states."""

from .representation import Representation
from .densitymatrix import DensityMatrix
from .wavefunction import WaveFunction
from .bargmann import Bargmann
from .wigner import Wigner
from numbers import Number

from mrmustard.physics.fock import dm_to_ket

def Ket(Representation):
    """Array-based coherent representation of quantum states as a vector in a Fock space.
    """

    def __init__(self, data: Union[Representation, Array]):
        if isinstance(data, Array):
            self.array = data
        elif purity(data) < 1:
            raise ValueError("Cannot convert a mixed state to a ket")
        else:
            super().__init__(data)

    def __repr__(self):
        return f"{self.__class__.__qualname__} | shape = {self.data.shape}"

    def __add__(self, other):
        if not isinstance(other, Ket):
            raise ValueError("Can only add a ket to a ket")
        if self.array.shape != other.array.shape:
            raise ValueError("Cannot add kets of different shape")
        return Ket(self.array + other.array)

    def __mul__(self, other):
        if not isinstance(other, Number):
            raise ValueError("Can only multiply a ket by a number")
        return Ket(self.data * other)
    
