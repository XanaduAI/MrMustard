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


def connect(components: Sequence[CircuitComponent]) -> Sequence[CircuitComponent]:
    r"""
    Takes as input a sequence of circuit components and connects their wires.

    In particular, it generates a list of light copies of the given components, then it modifies
    the wires' ``id``s so that connected wires have the same ``id``. It returns the list of light
    copies, leaving the input list unchanged.
    """
    # a dictionary mapping the each mode in ``components`` to the latest output wire on that
    # mode, or ``None`` if no wires have acted on that mode yet.
    output_ket: dict[Mode, Optional[Wire]] = {m: None for c in components for m in c.modes}

    ret = [component.light_copy() for component in components]

    for component in ret:
        for mode in component.modes:
            if output_ket[mode]:
                component.wires.in_ket[mode] = output_ket[mode]
            output_ket[mode] = component.wires.out_ket[mode]
    return ret
