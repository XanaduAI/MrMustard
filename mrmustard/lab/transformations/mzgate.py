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
The class representing a Mach-Zehnder gate.
"""

from __future__ import annotations

from collections.abc import Sequence

from mrmustard import math
from mrmustard.physics.ansatz import PolyExpAnsatz
from mrmustard.physics.wires import Wires

from ...physics import symplectics
from ..utils import make_parameter
from .base import Unitary

__all__ = ["MZgate"]


class MZgate(Unitary):
    r"""
    Mach-Zehnder gate.

    It supports two conventions:
        1. if ``internal=True``, both phases act inside the interferometer: ``phi_a`` on the upper arm, ``phi_b`` on the lower arm.
        2. if ``internal = False``, both phases act on the upper arm: ``phi_a`` before the first BS, ``phi_b`` after the first BS.

    Args:
        modes: The pair of modes of the MZ gate.
        phi_a: The phase in the upper arm of the MZ interferometer.
        phi_b: The phase in the lower arm or external of the MZ interferometer.
        phi_a_trainable: Whether ``phi_a`` is trainable.
        phi_b_trainable: Whether ``phi_b`` is trainable.
        phi_a_bounds: The bounds for ``phi_a``.
        phi_b_bounds: The bounds for ``phi_b``.
        internal: Whether phases are both in the internal arms.

    .. code-block::

        >>> from mrmustard.lab import MZgate

        >>> mz = MZgate((0, 1), phi_a=0.1, phi_b=0.2)
        >>> assert mz.modes == (0, 1)
        >>> assert mz.parameters.phi_a.value == 0.1
        >>> assert mz.parameters.phi_b.value == 0.2
    """

    short_name = "MZ"

    def __init__(
        self,
        modes: tuple[int, int],
        phi_a: float | Sequence[float] = 0.0,
        phi_b: float | Sequence[float] = 0.0,
        phi_a_trainable: bool = False,
        phi_b_trainable: bool = False,
        phi_a_bounds: tuple[float | None, float | None] = (None, None),
        phi_b_bounds: tuple[float | None, float | None] = (None, None),
        internal: bool = False,
    ):
        super().__init__(name="MZgate")
        self.parameters.add_parameter(
            make_parameter(phi_a_trainable, phi_a, "phi_a", phi_a_bounds, dtype=math.float64)
        )
        self.parameters.add_parameter(
            make_parameter(phi_b_trainable, phi_b, "phi_b", phi_b_bounds, dtype=math.float64)
        )

        self._ansatz = PolyExpAnsatz.from_function(
            fn=lambda phi_a, phi_b, internal: Unitary.from_symplectic(
                modes,
                symplectics.mzgate_symplectic(phi_a, phi_b, internal),
            ).bargmann_triple(),
            phi_a=self.parameters.phi_a,
            phi_b=self.parameters.phi_b,
            internal=internal,
        )
        self._wires = Wires(modes_in_ket=set(modes), modes_out_ket=set(modes))
