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

from abc import ABC, abstractproperty
from mrmustard.typing import RealMatrix, RealVector, Scalar, Tensor
from mrmustard.math import Math
from mrmustard.lab.representations.data.data import Data

math = Math()


class Representation(ABC):
    r"""Abstract base class for the different Representation of quantum states.
        
        Args:
            data (Data)

        Methods:
            purity
            norm
            number_means
            number_cov
            number_variances
            number_stdev
            probability
    """

    def __init__(self, data):
        self.data = Data(data)


    @abstractproperty
    def purity(self) -> Scalar:
        r"""Purity of the quantum state, defined as :math:`\mathrm{Tr} \rho^2`."""
        # Valid for all representation classes


    @abstractproperty
    def norm(self) -> float:
        r"""The norm of the state, defined as :math:`` for pure state and :math:`` for mixed state."""
        # Valid for Fock and WaveFunctionQ


    @abstractproperty
    def number_means(self) -> RealVector:
        r"""The photon number means vector."""
        # Valid for Fock and Wigner

    @abstractproperty
    def probability(self) -> Tensor:
        r"""Probability tensor, either extracted from a DM or from a Ket"""
        # Valid for Fock and WaveFucntionQ
