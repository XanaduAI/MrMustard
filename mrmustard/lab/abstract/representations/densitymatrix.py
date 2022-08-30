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
from .wavefunction import WaveFunction
from .bargmann import Bargmann
from .wigner import Wigner

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
        print('implementing ket->dm transform...')
        self.data = ket_to_dm(ket.data)

    def from_dm(self, dm):
        print('implementing dm->dm transform...')
        self.data = dm.data

    def from_bargmann(self, bargmann):
        print('implementing bargmann->dm transform...')
        A,b,c = bargmann.data
        self.data = math.hermite_renormalized(A,b,c)
    
    def from_wf(self, wavefunction):
        print('implementing wf->dm transform...')

    def __repr__(self):
        return f"{self.__class__.__qualname__} | shape = {self.data.shape}"