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
from mrmustard.math import Math
from mrmustard.lab.representations.fock import Fock
from mrmustard.typing import Tensor

math = Math()

class FockKet(Fock):
    r""" Fock representation of a state described by a ket.
    
    Args:
        data: the data used to represent the state to be encoded as Fock representation
    """

    def __init__(self, array:np.array):
        super().__init__(array=array)
        self.num_modes = len(self.data.array.shape)
        self.cutoffs = self.data.array.shape


    @property
    def purity(self) -> float:
        return 1.0


    @property
    def norm(self) -> float:
        return math.abs(math.norm(self.data.array))
    

    @property
    def probability(self) -> Tensor: 
        return math.abs(self.data.array) #TODO: cutoffs