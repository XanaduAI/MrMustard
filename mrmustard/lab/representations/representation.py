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

math = Math()


class Representation(ABC):
    r"""Abstract parent class for the different Representation of quantum states."""

    @abstractproperty
    def purity(self) -> Scalar:
        r"""Valid for all representation classes : the purity of the state."""
        raise NotImplementedError()

    @abstractproperty
    def norm(self) -> float:
        r"""Valid for Fock and WaveFunctionQ: the norm of the state."""
        raise NotImplementedError()

    @abstractproperty
    def number_means(self) -> RealVector:
        r"""Valid for Fock and Wigner: the photon number means vector."""
        raise NotImplementedError()

    @abstractproperty
    def number_cov(self) -> RealMatrix:
        r"""Valid for Wigner: the photon number covariance matrix."""
        raise NotImplementedError()

    @abstractproperty
    def number_variances(self) -> int:
        r"""Valid for Fock : variance of the number operator in each mode."""
        raise NotImplementedError()

    def number_stdev(self) -> int:
        r"""Valid for Fock: square root of the photon number variances (standard deviation)
        in each mode."""
        return math.sqrt(self.number_variances())

    @abstractproperty
    def probability(self) -> Tensor:  # TODO : add doc
        r"""Valid for Fock and WaveFucntionQ: Probability tensor, either extracted from a DM or from a Ket"""
        raise NotImplementedError()
