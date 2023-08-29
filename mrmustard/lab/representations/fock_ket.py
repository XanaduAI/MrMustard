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
from mrmustard.typing import RealVector, Tensor

math = Math()


class FockKet(Fock):
    r"""
    The Fock ket representation is to describe the pure state in the photon number basis or Fock basis :math:`\langle n|\psi\rangle`.

    Args:
        data: the Data class instance to store the fock tensor of the state.

    Properties:
        purity: the purity of the state.
        norm: the norm of the state.

    Methods:
        probability: get the probability of the state in quadrature basis.
    """

    def __init__(self, array: np.array):
        # Check it is a physical state: the norm is from 0 to 1
        if not math.norm(array) > 0 and math.norm(array) <= 1:
            raise ValueError("The array does not represent a physical state.")
        self.data = ArrayData(array=array)

    @property
    def purity(self) -> float:
        r"""The purity of the pure state is 1.0."""
        return 1.0

    @property
    def norm(self) -> float:
        r"""The norm of the pure state (:math:`|amp|`)."""
        return math.abs(math.norm(self.data.array))

    def probability(self) -> Tensor:
        return math.abs(self.data.array)
