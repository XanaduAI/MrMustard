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
    Each representation class has a data attribute to store all the information related to this state in this representation.
    Each representation class has several methods to calculate the state properties.

    Attributes:
        data (Data): the state information in different Data class.

    Methods:
        purity: the purity of the state.
        norm: the norm of the state.
        number_means: :math: mean vector of the photon number.
        number_cov: covariance matrix of the photon number.
        number_variances: math:`<\bar{n}^2> - <\bar{n}>^2` variances of the photon number.
        probability: fock probability or wavefunction probability.
    """

    def __init__(self, data):
        r"""Representation class is initialized by the data related to the state.

        Args:
        data (Data): the information related to the representation.

        """
        self.data = Data(data)

    @abstractproperty
    def purity(self) -> Scalar:
        r"""Returns the purity of the quantum state, defined as :math:`\mathrm{Tr} \rho^2`."""
        # Valid for all representation classes

    @abstractproperty
    def norm(self) -> float:
        r"""Returns the norm of the state, defined as :math:`` for pure state and :math:`` for mixed state."""
        # Valid for Fock and WaveFunctionQ

    @abstractproperty
    def number_means(self) -> RealVector:
        r"""Returns the photon number means vector."""
        # Valid for Fock and Wigner

    @abstractproperty
    def number_cov(self) -> RealVector:
        r"""Returns the photon number covariance matrix."""
        # Valid for Wigner

    @abstractproperty
    def number_variances(self) -> RealVector:
        r"""Returns the variance of the number operator."""
        # Valid for Fock

    @abstractproperty
    def probability(self) -> Tensor:
        r"""Returns the probability tensor, either extracted from a ket state or density matrix."""
        # Valid for Fock and WaveFucntionQ
