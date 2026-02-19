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

from mrmustard import math, settings
from mrmustard.parameters import Parameter
from mrmustard.physics.ansatz_factory import AnsatzFactory
from mrmustard.physics.wires import ReprEnum, Wires
from mrmustard.utils.typing import RealMatrix

from ..circuit_components_utils import TraceOut
from ..utils import reshape_params
from .builtins import gdm_state, gket_state
from .dm import DM
from .ket import Ket

__all__ = ["GaussianDM", "GaussianKet"]


class GaussianKet(Ket):
    r"""
    The `N`-mode pure state described by a Gaussian gate that acts on Vacuum.

    >>> from mrmustard.lab import GaussianKet, Ket
    >>> psi = GaussianKet.random(modes=0)
    >>> assert isinstance(psi, Ket)

    Args:
        modes: The modes over which the state is defined.
        symplectic: The symplectic representation of the unitary that acts on
        vacuum to produce the desired state.

    Returns:
        A ``GaussianKet``.

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
        symplectic: RealMatrix | Parameter,
    ) -> None:
        modes = (modes,) if isinstance(modes, int) else modes
        super().__init__(
            ansatz_factory=AnsatzFactory(
                ansatz_dict={ReprEnum.BARGMANN: (gket_state, ("symplectic", "lin_sup"))}
            ),
            wires=Wires(modes_out_ket=set(modes)),
            name=self.__class__.__name__,
        )
        self.parameters["symplectic"] = Parameter.from_cc_init(
            symplectic, "float64", f"{self.name}/symplectic"
        )

    @classmethod
    def random(
        cls, modes: int | tuple[int, ...], max_r: float = 1.0, seed: int | None = None
    ) -> GaussianKet:
        r"""
        Returns a random GaussianKet.

        Args:
            modes: The modes of the GaussianKet.
            max_r: Maximum squeezing parameter over which we make random choices.
            seed: The random seed. If ``None``, the global seed is used.

        Returns:
            The random GaussianKet.

        Raises:
            ValueError: if ``modes`` is an empty tuple.
        """
        modes = (modes,) if isinstance(modes, int) else modes
        if len(modes) == 0:
            raise ValueError("Cannot create a random GaussianKet with no modes.")
        symplectic = math.random_symplectic(len(modes), max_r, seed=seed)
        return cls(modes, symplectic)

    def get_modes(self, modes: int | Sequence[int]) -> GaussianKet:
        r"""
        Keep the given modes and trace out the rest. Returns a density matrix.

        Args:
            modes: The modes to keep.

        Returns:
            A new GaussianKet with the modes indexed by `modes`.
        """
        keep = {modes} if isinstance(modes, int) else set(modes)
        if not keep.issubset(self.modes):
            raise ValueError(f"Expected a subset of ``{self.modes}``, found ``{keep}``.")
        trace_out_modes = tuple(mode for mode in self.modes if mode not in keep)
        return self >> TraceOut(trace_out_modes)


class GaussianDM(DM):
    r"""
    The `N`-mode mixed state described by a Gaussian gate that acts on a given
    thermal state. This corresponds to the Williamson form. It is equivalent to
    the construction ``Thermal(beta) >> Ggate(symplectic)``.

    >>> from mrmustard.lab import GaussianDM, DM
    >>> rho = GaussianDM.random(modes=0)
    >>> assert isinstance(rho, DM)

    Args:
        modes: The modes over which the state is defined.
        beta: the set of temperatures determining the thermal states. If only a float
            is provided for a multi-mode state, the same temperature is considered
            across all modes.
        symplectic: The symplectic representation of the unitary that acts on a
            vacuum to produce the desired state.

    Returns:
        A ``GaussianDM``.

    .. details::

        For a given Gaussian unitary U (that is determined by its symplectic
        representation), and a set of temperatures, produces the state

        .. math::

            \rho = U (\bigotimes_i \rho_t(\beta_i))

        where rho_t are thermal states with temperatures determined by beta.
    """

    short_name = "Gdm"

    def __init__(
        self,
        modes: int | tuple[int, ...],
        beta: float | Sequence[float] | Parameter,
        symplectic: RealMatrix | Parameter,
    ) -> None:
        modes = (modes,) if isinstance(modes, int) else modes
        super().__init__(
            ansatz_factory=AnsatzFactory(
                ansatz_dict={ReprEnum.BARGMANN: (gdm_state, ("beta", "symplectic", "lin_sup"))}
            ),
            wires=Wires(modes_out_bra=set(modes), modes_out_ket=set(modes)),
            name=self.__class__.__name__,
        )
        (betas,) = list(
            reshape_params(len(modes), betas=beta.value if isinstance(beta, Parameter) else beta)
        )
        self.parameters["symplectic"] = Parameter.from_cc_init(
            symplectic, "float64", f"{self.name}/symplectic"
        )
        self.parameters["beta"] = Parameter.from_cc_init(betas, "float64", f"{self.name}/beta")

    @classmethod
    def random(
        cls,
        modes: int | tuple[int, ...],
        max_r: float = 1.0,
        min_beta: float = 0.2,
        max_beta: float = 1.0,
        seed: int | None = None,
    ) -> GaussianDM:
        r"""
        Returns a random ``GaussianDM`` constructed as ``Thermal(beta) >> Ggate(symplectic)``
        for a random ``beta`` and a random ``symplectic``.
        If a specific ``beta`` is required, use the construction ``Thermal(beta) >> Ggate.random()``
        where beta is the desired inverse temperature.
        To obtain random Gaussian density matrices from the induced trace measure,
        use the construction ``GaussianKet.random().get_modes()`` with appropriate modes.

        Args:
            modes: The modes of the GaussianDM.
            max_r: Maximum squeezing parameter for the Gaussian unitary. Default is 1.0.
            min_beta: Minimum inverse temperature that will be sampled. Default is 0.5.
            max_beta: Maximum inverse temperature that will be sampled. Default is 1.0.
            seed: The random seed. If ``None``, the global seed is used.

        Returns:
            The random GaussianDM.

        Raises:
            ValueError: if ``modes`` is an empty tuple.
        """
        modes = (modes,) if isinstance(modes, int) else modes
        if (m := len(modes)) == 0:
            raise ValueError("Cannot create a random GaussianDM with no modes.")
        rng = settings.get_rng(seed)
        beta = rng.uniform(low=min_beta, high=max_beta, size=m)
        symplectic = math.random_symplectic(m, max_r, seed=seed)
        return cls(modes, beta, symplectic)

    def get_modes(self, modes: int | Sequence[int]) -> GaussianDM:
        r"""
        Keep the given modes and trace out the rest.

        Args:
            modes: The modes to keep.

        Returns:
            A new GaussianDM with the modes indexed by `modes`.
        """
        keep = {modes} if isinstance(modes, int) else set(modes)
        if not keep.issubset(set(self.modes)):
            raise ValueError(f"Expected a subset of ``{self.modes}``, found ``{keep}``.")
        trace_out_modes = tuple(mode for mode in self.modes if mode not in keep)
        return self >> TraceOut(trace_out_modes)
