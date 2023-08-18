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
from mrmustard.lab.representations.data.array_data import ArrayData
from mrmustard.typing import Tensor


math = Math()


class FockDM(Fock):
    r"""
    The Fock ket representation is to describe the mixed state in the photon number basis or Fock basis :math:`\langle m|\rho|n\rangle`.

    Args:
        data: the Data class instance to store the fock tensor of the state.
    """

    def __init__(self, array):
        # Check it is a physical state: the norm is from 0 to 1
        if not math.norm(array) > 0 and math.norm(array) <= 1:
            raise ValueError("The array does not represent a physical state.")
        self.data = ArrayData(array=array)

    @property
    def purity(self) -> float:
        r"""The purity of the pure state is :math:`Tr(\rho^2)`."""
        dm = self.data.array
        cutoffs = dm.shape[: len(dm.shape) // 2]
        d = int(np.prod(cutoffs))  # combined cutoffs in all modes
        dm = math.reshape(dm, (d, d))
        dm = dm / math.trace(dm)  # assumes all nonzero values are included in the density matrix
        return math.abs(math.sum(math.transpose(dm) * dm))  # tr(rho^2)

    @property
    def norm(self) -> float:
        r"""The norm of the mixed state (:math:`|amp|^2`)."""
        return math.sum(math.all_diagonals(self.data.array, real=True))

    def probability(self) -> Tensor:
        return math.all_diagonals(self.data.array, real=True)
