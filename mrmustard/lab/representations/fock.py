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


from mrmustard.lab.representations.representation import Representation
from mrmustard.typing import Vector, Matrix
from mrmustard.math import Math

math = Math()


class Fock(Representation):
    r"""Fock representation of a state.

    The Fock representation is to describe the state in the photon number basis or Fock basis.

    """

    def number_means(self) -> Vector:
        probs = self.probability()
        modes = list(range(len(probs.shape)))
        marginals = [math.sum(probs, axes=modes[:k] + modes[k + 1 :]) for k in range(len(modes))]
        result = [math.sum(m * math.arange(len(m), dtype=m.dtype)) for m in marginals]
        return math.astensor(result)

    def number_variances(self) -> Vector:
        probs = self.probability()
        modes = list(range(len(probs.shape)))
        marginals = [math.sum(probs, axes=modes[:k] + modes[k + 1 :]) for k in range(len(modes))]
        t = marginals[0].dtype
        result = [
            (
                math.sum(m * math.arange(m.shape[0], dtype=t) ** 2)
                - math.sum(m * math.arange(m.shape[0], dtype=t)) ** 2
            )
            for m in marginals
        ]
        return math.astensor(result)

    def number_cov(self) -> Matrix:
        raise NotImplementedError(
            f"This property is not available in {self.__class__.__qualname__} representation"
        )
