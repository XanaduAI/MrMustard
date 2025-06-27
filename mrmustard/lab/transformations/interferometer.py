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
The class representing an Interferometer gate.
"""

from __future__ import annotations

from mrmustard import math
from mrmustard.math.parameters import update_unitary
from mrmustard.physics.ansatz import PolyExpAnsatz
from mrmustard.physics.wires import Wires
from mrmustard.utils.typing import ComplexMatrix

from ...physics import symplectics
from ..utils import make_parameter
from .base import Unitary

__all__ = ["Interferometer"]


class Interferometer(Unitary):
    r"""
    N-mode interferometer.

    It corresponds to a Ggate with zero mean and a ``2N x 2N`` unitary symplectic matrix.

    Args:
        modes: The modes this gate is applied to.
        unitary: A unitary matrix. For N modes it must have shape `(N,N)`. If ``None``, a random unitary is generated.
        unitary_trainable: Whether ``unitary`` is trainable.

    Raises:
        ValueError: If the size of the unitary does not match the number of modes.

    .. code-block::

        >>> from mrmustard import math
        >>> from mrmustard.lab import Interferometer

        >>> unitary = Interferometer(modes=(1, 2), unitary=math.eye(2))
        >>> assert unitary.modes == (1, 2)
        >>> assert math.allclose(unitary.symplectic, math.eye(4))
    """

    short_name = "I"

    def __init__(
        self,
        modes: int | tuple[int, ...],
        unitary: ComplexMatrix | None = None,
        unitary_trainable: bool = False,
    ):
        modes = (modes,) if isinstance(modes, int) else modes
        num_modes = len(modes)
        unitary = unitary if unitary is not None else math.random_unitary(num_modes)
        if unitary.shape[-1] != num_modes:
            raise ValueError(
                f"The size of the unitary must match the number of modes: {unitary.shape[-1]} =/= {num_modes}",
            )
        super().__init__(name="Interferometer")
        self.parameters.add_parameter(
            make_parameter(unitary_trainable, unitary, "unitary", (None, None), update_unitary),
        )
        self._ansatz = PolyExpAnsatz.from_function(
            fn=lambda uni: Unitary.from_symplectic(
                modes,
                symplectics.interferometer_symplectic(uni),
            ).bargmann_triple(),
            uni=self.parameters.unitary,
        )
        self._wires = Wires(modes_in_ket=set(modes), modes_out_ket=set(modes))
