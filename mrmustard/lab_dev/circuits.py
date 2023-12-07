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

"""
A base class for the components of quantum circuits.
"""

from __future__ import annotations

from typing import Optional, Sequence
from ..math.parameter_set import ParameterSet
from ..math.parameters import Constant, Variable
from .circuit_components import CircuitComponent
from .wires import Wires


class TensorNetwork:
    r"""
    """
    def __init__(self) -> None:
        self._network = {}

    @property
    def network(self) -> dict[int, int]:
        r"""
        A dictionary from ``int`` to ``int``, where the first ``int`` represents
        the index labelling an output wire and the second ``int`` represents the index
        labelling an input wire.
        """
        return self._network



class Circuit:
    r"""
    A quantum circuits.

    Arguments:
        components: A list of 
    """

    def __init__(
        self,
        # components: Optional[Sequence[CircuitComponent]] = None,
    ) -> None:
        self._components = []
        self._network = TensorNetwork()

    @property
    def components(self) -> Sequence[CircuitComponent]:
        return self._components
    
    @property
    def network(self) -> TensorNetwork:
        r"""
        A dictionary
        """
        self._network

