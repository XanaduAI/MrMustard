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
from mrmustard.lab.representations.bargmann import Bargmann
from mrmustard.lab.representations.data.qpoly_data import QPolyData
from mrmustard.typing import Matrix, Scalar, Vector
from mrmustard.math import Math

math = Math()


class BargmannDM(Bargmann):
    r"""BargmannDM representation is the Bargmann representation of a mixed state.
    It is defined as :math:`\langle \alpha||\rho||\beta\rangle = e^{-1/2|\alpha|^2}e^{-1/2|\beta|^2}\langle \alpha|\rho|\beta\rangle = c\exp\left(\frac12 x^T A x + b^T x\right)`.

    Args:
        A: complex symmetric matrix
        b: complex vector
        c: constants
    """

    def __init__(self, A: Matrix, b: Vector, c: Scalar):
        # Check the covariance matrices is real symmetric
        if not np.allclose(math.transpose(A), A):
            raise ValueError("The A matrix is symmetric!")
        self.data = QPolyData(A=A, b=b, c=c)
