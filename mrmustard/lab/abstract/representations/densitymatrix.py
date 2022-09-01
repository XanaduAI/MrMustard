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

import numpy as np
from functools import cached_property
from mrmustard.math import Math; math = Math()
from mrmustard.physics.fock import ket_to_dm
from .representation import Representation
from .densitymatrix import DensityMatrix

class DensityMatrix(Representation):

    def __init__(self, data: Union[Representation, Array]):
        if isinstance(data, Array):
            self.array = data
        elif purity(data) < 1: # assume it's a Representation
            raise ValueError("Cannot convert a mixed state to a ket")
        else:
           super().__init__(data)

    @cached_property
    def purity(self):
        return math.sum(math.abs(self.data)**2)

    @cached_property
    def norm(self):
        return math.trace(self.data)

    def from_ket(self, ket):
        print(f'ket->{self.__class__.__qualname__}')
        self.data = ket_to_dm(ket.data)

    def from_dm(self, dm):
        print(f'dm->{self.__class__.__qualname__}')
        self.data = dm.data

    def from_bargmann(self, bargmann):
        A,b,c = bargmann.data
        self.data = math.hermite_renormalized(A,b,c)
    
    def from_wf(self, wavefunction):
        print(f'wf->{self.__class__.__qualname__}')

    def __repr__(self):
        return f"{self.__class__.__qualname__} | shape = {self.data.shape}"

    def __add__(self, other):
        if not isinstance(other, Ket):
            raise ValueError("Can only add a density matrix to a density matrix")
        if self.array.shape != other.array.shape:
            raise ValueError("Cannot add density matrices of different shape")
        return DensityMatrix(self.data + other.data)

    def __mul__(self, other):
        if not isinstance(other, Number):
            raise ValueError("Can only multiply by a scalar")
        return DensityMatrix(self.data * other)