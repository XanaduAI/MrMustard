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
from mrmustard.typing import Scalar, Tensor
from mrmustard.math import Math

math = Math()
from mrmustard.lab.representations.wavefunction import WaveFunction


class WaveFunctionKet(WaveFunction):
    r"""Wavefunction representation of a ket state."""

    def __init__(self, qs: np.array, quadrature_angle: np.float, wavefunction: np.array):
        r"""The wavefunction representation is initialized through three parameters.

        Args:
            points: variable points along the basis.
            quadrature_angle: quadrature angle along different basis.
            array: the wavefunction values according to each points.
        """
        super().__init__(qs=qs, quadrature_angle=quadrature_angle, wavefunction=wavefunction)

    @property
    def purity(self) -> Scalar:
        return 1.0

    @property
    def norm(self) -> float:
        return math.abs(math.norm(self.data.array))

    @property
    def probability(self) -> Tensor:
        return math.abs(self.data.array, real=True)
