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
from typing import Optional
from abc import ABC, abstractmethod
from mrmustard.typing import Matrix, Vector, Tensor
from mrmustard.math import Math

math = Math()


class Representation(ABC):
    r"""Abstract base class for the different Representation of quantum states.
    Each representation class has a data attribute to store all the information related to this state in this representation.
    Each representation class has several methods to calculate the state properties.

    Properties:
        purity: the purity of the state.
        norm: the norm of the state.
    Methods:
        number_means: :math: mean vector of the photon number.
        number_cov: covariance matrix of the photon number.
        number_variances: math:`<\bar{n}^2> - <\bar{n}>^2` variances of the photon number.
        probability: fock probability or wavefunction probability.
    """

    @property
    @abstractmethod
    def purity(self) -> Optional[float]:
        r"""Returns the purity of the quantum state."""
        # Valid for all representation classes

    @property
    @abstractmethod
    def norm(self) -> float:
        r"""Returns the norm of the state."""
        # Valid for Fock and WaveFunctionQ

    @abstractmethod
    def number_means(self) -> Optional[Vector]:
        r"""Returns the photon number means vector."""
        # Valid for Fock and Wigner

    @abstractmethod
    def number_cov(self) -> Optional[Matrix]:
        r"""Returns the photon number covariance matrix."""
        # Valid for Wigner

    @abstractmethod
    def number_variances(self) -> Vector:
        r"""Returns the variance of the photon number operator."""
        # Valid for Fock

    @abstractmethod
    def probability(self) -> Tensor:
        r"""Returns the probability tensor, either extracted from a ket state or density matrix."""
        # Valid for Fock and WaveFucntionQ
