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

from mrmustard.lab.representations.bargmann import Bargmann
from mrmustard.typing import Matrix, Scalar, Vector
from mrmustard.math import Math

math = Math()


class BargmannDM(Bargmann):
    r"""Fock-Bargmann representation of a mixed state."""

    def __init__(self, A: Matrix, b: Vector, c: Scalar) -> None:
        r"""Fock-Bargmann representation of a mixed state.

        Args:
            A: complex symmetric matrix
            b: complex vector
            c: constants
        """
        # Check the covariance matrices is real symmetric
        if not math.transpose(A) == A:
            raise ValueError("The A matrix is symmetric!")
        super().__init__(A=A, b=b, c=c)

    @property
    def purity(self):
        raise NotImplementedError(
            f"This property is not available in {self.__class__.__qualname__} representation"
        )
