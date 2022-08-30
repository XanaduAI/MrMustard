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

    def __init__(self, data: Union[Representation, Array]):
        if isinstance(data, Array):
            self.array = data
        elif purity(data) < 1: # assume it's a Representation
            raise ValueError("Cannot convert a mixed state to a ket")
        else:
            super().__init__(data)
    
    def purity(self):
        "assumes normalized purity"
        return 1.0

    @cached_property
    def norm(self):
        return math.sqrt(math.sum(math.abs(self.data)**2))

    def from_ket(self, ket):
        print('implementing ket->ket transform...')
        self.data = ket.data

    def from_dm(self, dm):
        print('implementing dm->ket transform...')
        self.data = dm_to_ket(dm.data)

    def from_wf(self, wavefunction):
        print('implementing wf->ket transform...')

    def __repr__(self):
        return f"{self.__class__.__qualname__} | shape = {self.data.shape}"

    def __add__(self, other):
        if not isinstance(other, Ket):
            raise ValueError("Can only add a ket to a ket")
        return Ket(self.data + other.data)

    def __mul__(self, other):
        if not isinstance(other, Number):
            raise ValueError("Can only multiply a ket by a number")
        k = Ket(self.data * other)
        k.purity = self.purity * math.abs(other)**2
        return k
    
