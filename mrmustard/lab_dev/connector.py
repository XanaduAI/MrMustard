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
Classed and methods to connect the components of quantum circuits.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ..utils.typing import Mode
from .circuit_components import CircuitComponent
from .wires import Wire

def connect_two(wire1: Wire, wire2: Wire) -> None:
    wire1.connect(wire2)

def connect_all(components: Sequence[CircuitComponent]) -> Sequence[CircuitComponent]:
    r"""
    Takes as input a sequence of unconnected circuit components and connects them together.
    """
    ret = []
    output_ket: dict[Mode, Optional[Wire]] = {m: None for c in components for m in c.modes}

    for component in components:
        new_component = component.light_copy()
        for m in new_component.modes:
            if output_ket[m]:
                connect_two(output_ket[m], new_component.wires.in_ket[m])
            output_ket[m] = new_component.wires.out_ket[m]
        ret += [new_component]
    return ret