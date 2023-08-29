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
import numpy as np
from mrmustard.lab.representations.bargmann import Bargmann
from mrmustard.lab.representations.data.abc_data import ABCData
from mrmustard.typing import Matrix, Scalar, Vector
from mrmustard.math import Math

math = Math()


class BargmannDM(Bargmann):
    r"""BargmannDM representation is the Bargmann representation of a mixed state.
    It is defined as :math:`\langle \alpha||\rho||\beta\rangle = e^{-1/2|\alpha|^2}e^{-1/2|\beta|^2}\langle \alpha|\rho|\beta\rangle = c\exp\left(\frac12 x^T A x + b^T x\right)`.

    Args:
        A (Optional[Matrix]): complex symmetric matrices and the first dimension is the batch dimension indicates the linear combination of different BargmannKet Classes.
        b (Optional[Vector]): complex vectors and the first dimension is the batch dimension indicates the linear combination of different BargmannKet Classes.
        c (Optional[Scalar]): constants and the first dimension is the batch dimension indicates the linear combination of different BargmannKet Classes.
    """

    def __init__(self, A: Optional[Matrix], b: Optional[Vector], c: Optional[Scalar]):
        # Check the covariance matrices is real symmetric
        if not np.allclose(math.transpose(A), A):
            raise ValueError("The A matrix is symmetric!")

        if A.shape == 2:
            A = math.expand_dims(A, axis=0)
            b = math.expand_dims(b, axis=0)

        self.data = ABCData(A=A, b=b, c=c)
