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
A class to connect the components of quantum circuits.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from ..math.parameter_set import ParameterSet
from ..math.parameters import Constant, Variable
from .circuit_components import CircuitComponent
from .wires import Wire, Wires


class Network:
    r"""
    A book-keeping object used by ``Circuit``s to keep track of the unconnected output
    wires, so that these wires can be quickly connected to those of new components added
    to the circuit.
    """

    def __init__(self) -> None:
        self._ket = {}

    @property
    def ket(self) -> dict[int, Wire]:
        return self._ket


def make_connections(components: Sequence[CircuitComponent]) -> Sequence[CircuitComponent]:
    r"""
    Takes as input a sequence of unconnected circuit components and connects them together.
    """