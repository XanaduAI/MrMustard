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
from mrmustard.math import Math
from mrmustard.lab.representations.representation import Representation
from mrmustard.lab.representations.data.array_data import ArrayData
from mrmustard.typing import Tensor, RealMatrix

math = Math()


class Fock(Representation):
    r"""Fock representation of a state.

    The Fock representation is to describe the state in the photon number basis or Fock basis.

    """

    def __init__(self, array: np.array) -> None:
        r"""The Fock representation is initialized through one parameter.

        Args:
            array: the fock tensor.
        """
        self.data = ArrayData(array=array)

    @property
    def number_means(self) -> Tensor:
        r"""Returns the photon number means vector."""
        probs = self.probability()
        nb_modes = range(len(probs.shape))
        modes = list(nb_modes)  # NOTE : there is probably a more optimized way of doing this
        marginals = [math.sum(probs, axes=modes[:k] + modes[k + 1 :]) for k in nb_modes]
        result = [math.sum(m * math.arange(len(m), dtype=m.dtype)) for m in marginals]
        return math.astensor(result)

    @property
    def number_variances(self) -> Tensor:
        r"""Returns the variance of the number operator in each mode."""
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
