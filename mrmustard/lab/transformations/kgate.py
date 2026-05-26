# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The class representing a Kerr gate."""

from __future__ import annotations

import numpy as np

from mrmustard import math
from mrmustard.lab.circuit_components import CircuitComponent
from mrmustard.parameters import Parameter
from mrmustard.physics.ansatz.array_ansatz import ArrayAnsatz
from mrmustard.physics.wires import Wires

from .base import Unitary

__all__ = ["Kgate"]


class Kgate(Unitary):
    r"""The Kerr gate.

    A non-Gaussian single-mode unitary, diagonal in the Fock basis. By default
    the generator is the normal-ordered :math:`a^\dagger a^\dagger a a = n(n-1)`,
    giving :math:`U|n\rangle = e^{i\kappa n(n-1)}|n\rangle`. With
    ``normal_ordered=False`` the generator is :math:`n^2`, giving
    :math:`U|n\rangle = e^{i\kappa n^2}|n\rangle`.

    >>> from mrmustard.lab import Kgate, Number
    >>> gate = Kgate(mode=0, kappa=0.1)
    >>> assert gate.modes == (0,)

    Args:
        mode: The mode the gate is applied to.
        kappa: The Kerr nonlinearity strength.
        normal_ordered: If ``True`` (default), the generator is
            :math:`a^\dagger a^\dagger a a = n(n-1)`. If ``False``, the generator
            is :math:`n^2`. The two differ only by a linear rotation
            :math:`e^{i\kappa n}`, i.e. an  ``Rgate`` .
    """

    short_name = "K"

    def __init__(
        self,
        mode: int | tuple[int],
        kappa: float | Parameter = 0.0,
        normal_ordered: bool = True,
    ):
        mode = (mode,) if not isinstance(mode, tuple) else mode
        super().__init__(
            wires=Wires(modes_in_ket=set(mode), modes_out_ket=set(mode)),
            name=self.__class__.__name__,
        )
        self.parameters["kappa"] = Parameter.from_cc_init(kappa, "float64", f"{self.name}/kappa")
        self._normal_ordered = normal_ordered

    @property
    def normal_ordered(self) -> bool:
        r"""Whether the generator is :math:`n(n-1)` (True) or :math:`n^2` (False)."""
        return self._normal_ordered

    def __custom_rrshift__(self, other: CircuitComponent) -> CircuitComponent:
        r"""Kerr is diagonal in Fock, so we implement its right-shift directly:
        multiply the ket-side photon-number axis by :math:`e^{i\kappa g(n)}`, and
        (when present) the bra-side axis by :math:`e^{-i\kappa g(n)}`, where
        :math:`g(n) = n(n-1)` if ``normal_ordered`` else :math:`n^2`.

        Args:
            other: the component other than the Kgate in the contraction.

        Output:
            the result of the contraction.
        """
        other = other.to_fock()
        array = other.fock_array()
        core_shape = other.ansatz.core_shape
        state_batch_dims = other.ansatz.batch_dims
        n_modes = other.n_modes
        has_ket = bool(other.wires.ket)
        has_bra = bool(other.wires.bra)

        mode_indices = np.indices(core_shape)
        kappa = math.cast(math.astensor(self.parameters.kappa.value), "complex128")
        kappa_shape = kappa.shape
        kappa = math.reshape(kappa, (*kappa_shape, *((1,) * (state_batch_dims + len(core_shape)))))

        def g(n: np.ndarray) -> np.ndarray:
            return n * (n - 1) if self._normal_ordered else n * n

        (mode,) = self.modes
        phase_exp = math.astensor(0, dtype="complex128")
        if has_ket:
            ket_axis = (n_modes if has_bra else 0) + mode
            phase_exp = phase_exp + math.astensor(g(mode_indices[ket_axis]), dtype="complex128")
        if has_bra:
            phase_exp = phase_exp - math.astensor(g(mode_indices[mode]), dtype="complex128")
        array = array * math.exp(1j * kappa * phase_exp)

        return CircuitComponent._from_attributes(
            ArrayAnsatz(array, batch_dims=state_batch_dims + len(kappa_shape)),
            other.wires,
            self.name,
        )
