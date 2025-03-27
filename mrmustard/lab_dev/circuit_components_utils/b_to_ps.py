# Copyright 2025 Xanadu Quantum Technologies Inc.

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
The class representing an operation that changes Bargmann into phase space.
"""

from __future__ import annotations

from mrmustard.physics import triples

from ..transformations.base import Map
from ...physics.ansatz import PolyExpAnsatz
from ...physics.wires import ReprEnum
from ..utils import make_parameter

__all__ = ["BtoPS"]


class BtoPS(Map):
    r"""
    The `s`-parametrized Wigner function` as a ``Map``.

    Used internally as a ``Channel`` for transformations between representations.

    Args:
        modes: The modes of this channel.
        s: The `s` parameter of this channel. The case `s=-1`  corresponds to Husimi, `s=0` to Wigner, and `s=1` to Glauber P function.
    """

    def __init__(
        self,
        modes: int | tuple[int, ...],
        s: float,
    ):
        modes = (modes,) if isinstance(modes, int) else modes
        super().__init__(name="BtoPS")
        self.parameters.add_parameter(make_parameter(False, s, "s", (None, None)))
        self._representation = self.from_ansatz(
            modes_in=modes,
            modes_out=modes,
            ansatz=PolyExpAnsatz.from_function(
                fn=triples.bargmann_to_wigner_Abc,
                s=self.parameters.s,
                n_modes=len(modes),
            ),
        ).representation
        for w in self.representation.wires.output.wires:
            w.repr = ReprEnum.CHARACTERISTIC
            w.repr_params_func = lambda: self.parameters.s
