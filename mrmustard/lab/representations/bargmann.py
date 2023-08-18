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
from mrmustard.typing import Matrix, Vector, Tensor
from mrmustard.math import Math
from mrmustard.lab.representations.representation import Representation

math = Math()


class Bargmann(Representation):
    r"""Bargmann representation of a Gaussian state.

    The Bargmann representation is to characterize the Gaussian state in the coherent basis with an extra prefactor.

    The Bargmann representation of a Gaussian state is in the quadrature polynomials form:
        ::math::
            c \exp\left(
                        \frac12 x^T A x + b^T x
                    \right)

    So we use the triple :math:`(A,b,c)` to characterize a Gaussian state in the Bargmann representation.

    """

    @property
    def purity(self) -> Optional[float]:
        raise NotImplementedError(
            f"This property is not available in {self.__class__.__qualname__} representation"
        )

    @property
    def norm(self) -> float:
        raise NotImplementedError(
            f"This property is not available in {self.__class__.__qualname__} representation"
        )

    def number_means(self) -> Optional[Vector]:
        # NOTE: we can use the universal trace formula to do it!
        raise NotImplementedError(
            f"This property is not available in {self.__class__.__qualname__} representation"
        )

    def number_cov(self) -> Optional[Matrix]:
        # NOTE: we can use the universal trace formula to do it!
        raise NotImplementedError(
            f"This property is not available in {self.__class__.__qualname__} representation"
        )

    def number_variances(self) -> Vector:
        raise NotImplementedError(
            f"This property is not available in {self.__class__.__qualname__} representation"
        )

    def probability(self) -> Tensor:
        raise NotImplementedError(
            f"This property is not available in {self.__class__.__qualname__} representation"
        )
