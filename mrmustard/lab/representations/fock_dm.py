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

class FockDM(Fock):

    def __init__(self, array):
        super().__init__(array=array)
        self.num_modes = len(self.data.array.shape) // 2
        self.cutoffs = self.data.array.shape


    @property
    def purity(self) -> float:
        dm = self.data.array
        cutoffs = self.data.cutoffs
        d = int(np.prod(cutoffs))  # combined cutoffs in all modes
        dm = math.reshape(dm, (d, d))
        dm = dm / math.trace(dm)  # assumes all nonzero values are included in the density matrix
        return math.abs(math.sum(math.transpose(dm) * dm))  # tr(rho^2)


    @property
    def norm(self) -> float:
        r""" The norm. (:math:`|amp|^2` for ``dm``). """
        return math.sum(math.all_diagonals(self.data.array, real = True))


    @property
    def probability(self) -> Tensor:
        return math.all_diagonals(self.data.array, real = True) #TODO: cutoffs adjust