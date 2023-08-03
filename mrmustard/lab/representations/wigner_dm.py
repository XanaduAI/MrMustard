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

from mrmustard.math import Math
from mrmustard.lab.representations.wigner import Wigner
from mrmustard.lab.representations.data.gaussian_data import GaussianData
from mrmustard.typing import Matrix, Vector, Scalar
from typing import List
from mrmustard import settings

math = Math()


class WignerDM(Wigner):
    r"""Wigner representation of a mixed state.

    Args:
        cov: covariance matrices (real symmetric)
        mean: means (real)
        coeffs: coefficients (complex)
    """

    def __init__(self, cov: Matrix, means: Vector, coeffs: Scalar = 1.0) -> None:
        self.data = GaussianData(cov=cov, means=means, coeffs=coeffs)

    @property
    def cov(self):
        return self.data.cov

    @property
    def means(self):
        return self.data.means

    @property
    def purity(self) -> List[float]:
        purity_list = []
        for i in range(self.data.cov[-1]):
            purity_list.append(1 / math.sqrt(math.det((2 / settings.HBAR) * self.data.cov[i, :])))
        return purity_list
