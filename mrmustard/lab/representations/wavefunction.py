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
from mrmustard.typing import Matrix, Vector
from mrmustard.lab.representations.representation import Representation
from mrmustard.lab.representations.data.wavefunctionarray_data import WavefunctionArrayData


class WaveFunction(Representation):
    r"""Wavefunction representation of a state.

        The wavefunction representation is to describe the state in the quadrature basis.
        The quandrature angle is used to characterize the rotation of the basis inside the phase space.
        For example, if the angle is 0, we say it is the wavefunction in the position basis.
        If the angle is 90 degree, we say it is the wavefunction in the momentum basis.

    """

    def __init__(self, points: np.array, quadrature_angle: np.float, wavefunction: np.array) -> None:
        r"""The wavefunction representation is initialized through three parameters.

        Args:
            points: variable points along the basis.
            quadrature_angle: quadrature angle along different basis.
            array: the wavefunction values according to each points.
        """
        self.data = WavefunctionArrayData(qs=points, array=wavefunction)
        self.quadrature_angle = quadrature_angle

    @property
    def norm(self) -> float:
        raise NotImplementedError(
            f"This property is not available in {self.__class__.__qualname__} representation"
        )

    @property
    def number_means(self) -> Vector:
        raise NotImplementedError(
            f"This property is not available in {self.__class__.__qualname__} representation"
        )

    @property
    def number_cov(self) -> Matrix:
        raise NotImplementedError(
            f"This property is not available in {self.__class__.__qualname__} representation"
        )

    @property
    def number_variances(self) -> int:
        raise NotImplementedError(
            f"This property is not available in {self.__class__.__qualname__} representation"
        )

    @property
    def number_stdev(self) -> int:
        raise NotImplementedError(
            f"This property is not available in {self.__class__.__qualname__} representation"
        )
