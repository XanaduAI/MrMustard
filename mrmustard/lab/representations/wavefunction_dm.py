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


class WaveFunctionDM(WaveFunction):
    r"""The wavefunction ket representation is to describe the pure state in the quadrature basis.


    Args:
        points: variable points along the basis.
        quadrature_angle: quadrature angle along different basis.
        wavefunction: the wavefunction values according to each points.

    Properties:
        purity: the purity of the state.
        norm: the norm of the state.

    Methods:
        probability: get the probability of the state in quadrature basis.
    """

    def __init__(self, points: Tensor, quadrature_angle: float, wavefunction: Tensor):
        # Check it is a physical state: the norm is from 0 to 1
        if not math.norm(wavefunction) > 0 and math.norm(wavefunction) <= 1:
            raise ValueError("The array does not represent a physical state.")
        self.data = WavefunctionArrayData(qs=points, array=wavefunction)
        self.quadrature_angle = quadrature_angle

    @property
    def purity(self) -> float:
        raise NotImplementedError(
            f"This property is not available in {self.__class__.__qualname__} representation"
        )

    @property
    def norm(self) -> float:
        r"""The norm. (:math:`|amp|^2` for ``dm``)."""
        return math.sum(math.all_diagonals(self.data.array, real=True))

    def probability(self) -> Tensor:
        return math.all_diagonals(self.data.array, real=True)
