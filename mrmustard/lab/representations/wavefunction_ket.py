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

from mrmustard.typing import Tensor
from mrmustard.math import Math
from mrmustard.lab.representations.wavefunction import WaveFunction
from mrmustard.lab.representations.data.wavefunctionarray_data import WavefunctionArrayData

math = Math()


class WaveFunctionKet(WaveFunction):
    r"""The wavefunction ket representation is to describe the pure state in the quadrature basis.

    The wavefunction is defined with a discrete array of points in the quadrature basis.

    Args:
        points: variable points along the basis.
        quadrature_angle: quadrature angle along different basis.
        wavefunction: the wavefunction values according to each points.
    """

    def __init__(self, points: Tensor, quadrature_angle: float, wavefunction: Tensor):
        # Check it is a physical state: the norm is from 0 to 1
        if not math.norm(wavefunction) > 0 and math.norm(wavefunction) <= 1:
            raise ValueError("The array does not represent a physical state.")
        self.data = WavefunctionArrayData(qs=points, array=wavefunction)
        self.quadrature_angle = quadrature_angle

    @property
    def purity(self) -> float:
        return 1.0

    @property
    def norm(self) -> float:
        return math.abs(math.norm(self.data.array))

    def probability(self) -> Tensor:
        return math.abs(self.data.array, real=True)
