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

from typing import Optional, Sequence, Union
from ..math.parameter_set import ParameterSet
from ..math.parameters import Constant, Variable
from .circuit_components import CircuitComponent
from .wires import Wires


class Circuit:
    r"""
    A quantum circuit.

    Arguments:
        components: A list of ``CircuitComponent``.
    """

    def __init__(
        self,
        components: Optional[Sequence[CircuitComponent]] = None,
    ) -> None:
        self._components = components or []

    @property
    def components(self) -> Sequence[CircuitComponent]:
        return self._components

    # def append(self, component):
    #     if not self._components:
    #         self._components = [component]
    #     else:
    #         self._components[-1] >> component
    #         self._components += [component]

    # def append(self, component: CircuitComponent) -> None:
    #     if not self.components:
    #         self._components = [component]
    #     else:
    #         self._components[-1] >> component
    #         self._components.append(component)

    def __rshift__(self, other: Union[CircuitComponent, Circuit]):
        # circ >> other
        if isinstance(other, CircuitComponent):
            ret = self.components
            ret[-1] >> other
            return Circuit(ret + [other])

    def __getitem__(self, idx: int) -> CircuitComponent:
        return self._components[idx]
