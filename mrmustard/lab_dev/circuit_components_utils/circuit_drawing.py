# Copyright 2024 Xanadu Quantum Technologies Inc.

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
Helpers for drawing components as part of a circuit.
"""

from mrmustard import math, settings
from mrmustard.lab_dev.circuit_components import CircuitComponent

# update this when new controlled gates are added
control_gates = ["BSgate", "MZgate", "CZgate", "CXgate"]


def component_to_str(comp: CircuitComponent) -> list[str]:
    r"""
    Generates a list string-based representation for the given component.

    If ``comp`` is not a controlled gate, the list contains as many elements modes as in
    ``comp.modes``. For example, if ``comp=Sgate([0, 1, 5], r=[0.1, 0.2, 0.5])``, it returns
    ``['Sgate(0.1,0.0)', 'Sgate(0.2,0.0)', 'Sgate(0.5,0.0)']``.

    If ``comp`` is a controlled gate, the list contains the string that needs to be added to
    the target mode. For example, if``comp=BSgate([0, 1], 1, 2)``, it returns
    ``['BSgate(0.0,0.0)']``.

    Args:
        comp: A circuit component.
    """
    cc_name = comp.short_name
    parallel = isinstance(cc_name, list)
    if not comp.wires.input:
        cc_names = [f"◖{cc_name[i] if parallel else cc_name}◗" for i in range(len(comp.modes))]
    elif not comp.wires.output:
        cc_names = [f"|{cc_name[i] if parallel else cc_name})=" for i in range(len(comp.modes))]
    elif cc_name not in control_gates:
        cc_names = [f"{cc_name[i] if parallel else cc_name}" for i in range(len(comp.modes))]
    else:
        cc_names = [f"{cc_name}"]

    if comp.parameter_set.names and settings.DRAW_CIRCUIT_PARAMS:
        values = []
        for name in comp.parameter_set.names:
            param = comp.parameter_set.constants.get(name) or comp.parameter_set.variables.get(name)
            new_values = math.atleast_1d(param.value)
            if len(new_values) == 1 and cc_name not in control_gates:
                new_values = math.tile(new_values, (len(comp.modes),))
            values.append(math.asnumpy(new_values))
        return [cc_names[i] + str(val).replace(" ", "") for i, val in enumerate(zip(*values))]
    return cc_names
