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
Classes representing Gaussian states.
"""

from __future__ import annotations

from collections.abc import Sequence

from mrmustard.physics import triples
from mrmustard.physics.ansatz import PolyExpAnsatz
from mrmustard.physics.wires import Wires
from mrmustard.utils.typing import RealMatrix

from ..circuit_components_utils import TraceOut
from ..utils import reshape_params
from .dm import DM
from .ket import Ket

__all__ = ["GDM", "GKet"]


class GKet(Ket):
    r"""
    The `N`-mode pure state described by a Gaussian gate that acts on Vacuum.

    Args:
        modes: the modes over which the state is defined.
        symplectic: the symplectic representation of the unitary that acts on
        vacuum to produce the desired state. Use ``math.random_symplectic(len(modes))``
        to generate a random symplectic matrix if needed.

    Returns:
        A ``Ket``.

    .. code-block::

        >>> from mrmustard.lab import GKet, Ket

        >>> psi = GKet([0])

        >>> assert isinstance(psi, Ket)

    .. details::

        For a given Gaussian unitary U (that is determined by its symplectic
        representation), produces the state

        .. math::

            |\psi\rangle = U |0\rangle
    """

    short_name = "Gk"

    def __init__(
        self,
        modes: int | tuple[int, ...],
        symplectic: RealMatrix,
    ) -> None:
        modes = (modes,) if isinstance(modes, int) else modes

        A, b, c = triples.gket_state_Abc(symplectic=symplectic)
        ansatz = PolyExpAnsatz(A, b, c)
        wires = Wires(modes_out_ket=set(modes))

        super().__init__(ansatz=ansatz, wires=wires, name="GKet")

    def __getitem__(self, idx: int | Sequence[int]) -> GKet:
        r"""
        Override the default ``__getitem__`` method to handle symplectic slicing.

        Args:
            idx: The modes to keep.

        Returns:
            A new GKet with the modes indexed by `idx`.
        """
        idx = (idx,) if isinstance(idx, int) else tuple(idx)
        if not set(idx).issubset(self.modes):
            raise ValueError(f"Expected a subset of ``{self.modes}``, found ``{idx}``.")
        trace_out_modes = tuple(mode for mode in self.modes if mode not in idx)
        return self >> TraceOut(trace_out_modes)


class GDM(DM):
    r"""
    The `N`-mode mixed state described by a Gaussian gate that acts on a given
    thermal state.

    Args:
        modes: The modes over which the state is defined.
        beta: the set of temperatures determining the thermal states. If only a
        float is provided for a multi-mode state, the same temperature is considered
        across all modes.
        symplectic: The symplectic representation of the unitary that acts on a
        vacuum to produce the desired state. Use ``math.random_symplectic(len(modes))``
        to generate a random symplectic matrix if needed.

    Returns:
        A ``DM``.

    .. code-block::

        >>> from mrmustard.lab import GDM, DM

        >>> rho = GDM([0], beta = 1.0)

        >>> assert isinstance(rho, DM)

    .. details::

        For a given Gaussian unitary U (that is determined by its symplectic
        representation), and a set of temperatures, produces the state

        .. math::

            \rho = U (\bigotimes_i \rho_t(\beta_i))

        where rho_t are thermal states with temperatures determined by beta.
    """

    short_name = "Gd"

    def __init__(
        self,
        modes: int | tuple[int, ...],
        beta: float | Sequence[float],
        symplectic: RealMatrix | Sequence[RealMatrix],
    ) -> None:
        modes = (modes,) if isinstance(modes, int) else tuple(modes)
        (betas,) = list(reshape_params(len(modes), betas=beta))

        A, b, c = triples.gdm_state_Abc(
            betas=betas,
            symplectic=symplectic,
        )
        ansatz = PolyExpAnsatz(A, b, c)
        wires = Wires(modes_out_bra=set(modes), modes_out_ket=set(modes))

        super().__init__(ansatz=ansatz, wires=wires, name="GDM")

    def __getitem__(self, idx: int | Sequence[int]) -> GDM:
        r"""
        Override the default ``__getitem__`` method to handle symplectic slicing.

        Args:
            idx: The modes to keep.

        Returns:
            A new GDM with the modes indexed by `idx`.
        """
        idx = (idx,) if isinstance(idx, int) else idx
        if not set(idx).issubset(set(self.modes)):
            raise ValueError(f"Expected a subset of ``{self.modes}``, found ``{idx}``.")
        trace_out_modes = tuple(mode for mode in self.modes if mode not in idx)
        return self >> TraceOut(trace_out_modes)
