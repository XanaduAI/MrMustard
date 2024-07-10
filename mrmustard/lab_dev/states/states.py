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

# pylint: disable=abstract-method

"""
The classes representing states in quantum circuits.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

from mrmustard import math
from mrmustard.physics.representations import Bargmann, Fock
from mrmustard.physics.fock import fock_state
from mrmustard.physics import triples
from .base import Ket, DM
from ..utils import make_parameter, reshape_params

__all__ = [
    "Coherent",
    "DisplacedSqueezed",
    "Number",
    "SqueezedVacuum",
    "Thermal",
    "TwoModeSqueezedVacuum",
    "Vacuum",
]

#  ~~~~~~~~~~~
#  Pure States
#  ~~~~~~~~~~~


class Coherent(Ket):
    r"""The `N`-mode coherent state.

    If ``x`` and/or ``y`` are ``Sequence``\s, their length must be equal to `1` or `N`. If their length is equal to `1`,
    all the modes share the same parameters.

    .. code-block::

        >>> from mrmustard.lab_dev import Coherent, Vacuum, Dgate

        >>> state = Coherent(modes=[0, 1, 2], x=[0.3, 0.4, 0.5], y=0.2)
        >>> assert state == Vacuum([0, 1, 2]) >> Dgate([0, 1, 2], x=[0.3, 0.4, 0.5], y=0.2)

    Args:
        modes: The modes of the coherent state.
        x: The `x` displacement of the coherent state.
        y: The `y` displacement of the coherent state.
        x_trainable: Whether the `x` displacement is trainable.
        y_trainable: Whether the `y` displacement is trainable.
        x_bounds: The bounds of the `x` displacement.
        y_bounds: The bounds of the `y` displacement.

    .. details::

        For any :math:`\bar{\alpha} = \bar{x} + i\bar{y}` of length :math:`N`, the :math:`N`-mode
        coherent state displaced :math:`N`-mode vacuum state is defined by

        .. math::
            V = \frac{\hbar}{2}I_N \text{and } r = \sqrt{2\hbar}[\text{Re}(\bar{\alpha}), \text{Im}(\bar{\alpha})].

        Its ``(A,b,c)`` triple is given by

        .. math::
            A = O_{N\text{x}N}\text{, }b=\bar{\alpha}\text{, and }c=\text{exp}\big(-|\bar{\alpha}^2|/2\big).
    """

    short_name = "Coh"

    def __init__(
        self,
        modes: Sequence[int],
        x: Union[float, Sequence[float]] = 0.0,
        y: Union[float, Sequence[float]] = 0.0,
        x_trainable: bool = False,
        y_trainable: bool = False,
        x_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        y_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
    ):
        super().__init__(modes=modes, name="Coherent")
        self._add_parameter(make_parameter(x_trainable, x, "x", x_bounds))
        self._add_parameter(make_parameter(y_trainable, y, "y", y_bounds))

    @property
    def representation(self) -> Bargmann:
        n_modes = len(self.modes)
        xs, ys = list(reshape_params(n_modes, x=self.x.value, y=self.y.value))
        return Bargmann(*triples.coherent_state_Abc(xs, ys))


class DisplacedSqueezed(Ket):
    r"""The `N`-mode displaced squeezed vacuum state.

    If ``x``, ``y``, ``r``, and/or ``phi`` are ``Sequence``\s, their length must be equal to `1`
    or `N`. If their length is equal to `1`, all the modes share the same parameters.

    .. code-block::

        >>> from mrmustard.lab_dev import DisplacedSqueezed, Vacuum, Sgate, Dgate

        >>> state = DisplacedSqueezed(modes=[0, 1, 2], x=1, phi=0.2)
        >>> assert state == Vacuum([0, 1, 2]) >> Sgate([0, 1, 2], phi=0.2) >> Dgate([0, 1, 2], x=1)

    Args:
        modes: The modes of the coherent state.
        x: The displacements along the `x` axis, which represents position axis in phase space.
        y: The displacements along the `y` axis.
        r: The squeezing magnitude.
        phi: The squeezing angles.
        x_trainable: Whether `x` is a trainable variable.
        y_trainable: Whether `y` is a trainable variable.
        r_trainable: Whether `r` is trainable.
        phi_trainable: Whether `phi` is trainable.
        x_bounds: The bounds for the displacement along the `x` axis.
        y_bounds: The bounds for the displacement along the `y` axis, which represents momentum axis in phase space.
        r_bounds: The bounds of `r`.
        phi_bounds: The bounds of `phi`.
    """

    short_name = "DSq"

    def __init__(
        self,
        modes: Sequence[int],
        x: Union[float, Sequence[float]] = 0.0,
        y: Union[float, Sequence[float]] = 0.0,
        r: Union[float, Sequence[float]] = 0.0,
        phi: Union[float, Sequence[float]] = 0.0,
        x_trainable: bool = False,
        y_trainable: bool = False,
        r_trainable: bool = False,
        phi_trainable: bool = False,
        x_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        y_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        r_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        phi_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
    ):
        super().__init__(modes=modes, name="DisplacedSqueezed")
        self._add_parameter(make_parameter(x_trainable, x, "x", x_bounds))
        self._add_parameter(make_parameter(y_trainable, y, "y", y_bounds))
        self._add_parameter(make_parameter(r_trainable, r, "r", r_bounds))
        self._add_parameter(make_parameter(phi_trainable, phi, "phi", phi_bounds))

    @property
    def representation(self) -> Bargmann:
        n_modes = len(self.modes)
        params = reshape_params(
            n_modes, x=self.x.value, y=self.y.value, r=self.r.value, phi=self.phi.value
        )
        xs, ys, rs, phis = list(params)
        return Bargmann(*triples.displaced_squeezed_vacuum_state_Abc(xs, ys, rs, phis))


class Number(Ket):
    r"""The `N`-mode number state.

    .. code-block::

        >>> from mrmustard.lab_dev import Number

        >>> state = Number(modes=[0, 1], n=[10, 20])
        >>> assert state.modes == [0, 1]

    Args:
        modes: The modes of the number state.
        n: The number of photons in each mode.
        cutoffs: The cutoffs for the various modes. If ``cutoffs`` is given as
            an ``int``, it is broadcasted to all the states. If ``None``, it
            defaults to ``[n1+1, n2+1, ...]``, where ``ni`` is the photon number
            of the ``i``th mode.

    .. details::

        For any :math:`\bar{n} = (n_1,\:\ldots,\:n_N)`, the :math:`N`-mode number state is defined
        by

        .. math::
            \ket{\bar{n}} = \ket{n_1}\otimes\ldots\otimes\ket{n_N}\:,

        where :math:`\ket{n_j}` is the eigenstate of the number operator on mode `j` with eigenvalue
        :math:`n_j`.

    """

    short_name = "N"

    def __init__(
        self,
        modes: Sequence[int],
        n: Union[int, Sequence[int]],
        cutoffs: Optional[Union[int, Sequence[int]]] = None,
    ) -> None:
        super().__init__(modes=modes, name=f"{n}")

        if isinstance(n, int):
            n = (n,) * len(modes)
        elif len(n) != len(modes):
            raise ValueError(f"The number of modes is {len(modes)}, but {n} has length {len(n)}.")
        if isinstance(cutoffs, int):
            cutoffs = (cutoffs,) * len(modes)
        if cutoffs is not None and len(cutoffs) != len(modes):
            raise ValueError(
                f"The number of modes is {len(modes)}, but found {len(cutoffs)} cutoffs."
            )
        self.n = n
        for i, num in enumerate(n):
            self.fock_shape[i] = num + 1 if cutoffs is None else cutoffs[i] + 1

        self._representation = Fock(fock_state(self.n, [s - 1 for s in self.fock_shape]))


class SqueezedVacuum(Ket):
    r"""The `N`-mode squeezed vacuum state.

    If ``r`` and/or ``phi`` are ``Sequence``\s, their length must be equal to `1` or `N`. If their length is equal to `1`,
    all the modes share the same parameters.

    .. code-block::

        >>> from mrmustard.lab_dev import SqueezedVacuum, Vacuum, Sgate

        >>> state = SqueezedVacuum(modes=[0, 1, 2], r=[0.3, 0.4, 0.5], phi=0.2)
        >>> assert state == Vacuum([0, 1, 2]) >> Sgate([0, 1, 2], r=[0.3, 0.4, 0.5], phi=0.2)

    Args:
        modes: The modes of the squeezed vacuum state.
        r: The squeezing magnitude.
        phi: The squeezing angles.
        r_trainable: Whether `r` is trainable.
        phi_trainable: Whether `phi` is trainable.
        r_bounds: The bounds of `r`.
        phi_bounds: The bounds of `phi`.
    """

    short_name = "Sq"

    def __init__(
        self,
        modes: Sequence[int],
        r: Union[float, Sequence[float]] = 0.0,
        phi: Union[float, Sequence[float]] = 0.0,
        r_trainable: bool = False,
        phi_trainable: bool = False,
        r_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        phi_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
    ):
        super().__init__(modes=modes, name="SqueezedVacuum")
        self._add_parameter(make_parameter(r_trainable, r, "r", r_bounds))
        self._add_parameter(make_parameter(phi_trainable, phi, "phi", phi_bounds))

    @property
    def representation(self) -> Bargmann:
        n_modes = len(self.modes)
        rs, phis = list(reshape_params(n_modes, r=self.r.value, phi=self.phi.value))
        return Bargmann(*triples.squeezed_vacuum_state_Abc(rs, phis))


class TwoModeSqueezedVacuum(Ket):
    r"""The two-mode squeezed vacuum state.

    If ``r`` and/or ``phi`` are ``Sequence``\s, their length must be equal to `1`.

    .. code-block::

        >>> from mrmustard.lab_dev import TwoModeSqueezedVacuum, S2gate

        >>> state = TwoModeSqueezedVacuum(modes=[0, 1], r=0.3, phi=0.2)
        >>> assert state == Vacuum([0, 1]) >> S2gate([0, 1], r=0.3, phi=0.2)


    Args:
        modes: The modes of the coherent state.
        r: The squeezing magnitude.
        phi: The squeezing angles.
        r_trainable: Whether `r` is trainable.
        phi_trainable: Whether `phi` is trainable.
        r_bounds: The bounds of `r`.
        phi_bounds: The bounds of `phi`.
    """

    def __init__(
        self,
        modes: Tuple[int, int],
        r: float = 0.0,
        phi: float = 0.0,
        r_trainable: bool = False,
        phi_trainable: bool = False,
        r_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        phi_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
    ):
        super().__init__(modes=modes, name="TwoModeSqueezedVacuum")
        self._add_parameter(make_parameter(r_trainable, r, "r", r_bounds))
        self._add_parameter(make_parameter(phi_trainable, phi, "phi", phi_bounds))

    @property
    def representation(self) -> Bargmann:
        n_modes = len(self.modes)
        rs, phis = list(reshape_params(int(n_modes / 2), r=self.r.value, phi=self.phi.value))
        return Bargmann(*triples.two_mode_squeezed_vacuum_state_Abc(rs, phis))


class Vacuum(Ket):
    r"""
    The `N`-mode vacuum state.

    .. code-block ::

        >>> from mrmustard.lab_dev import Vacuum

        >>> state = Vacuum([1, 2])
        >>> assert state.modes == [1, 2]

    Args:
        modes: A list of modes.

    .. details::

        The :math:`N`-mode vacuum state is defined by

        .. math::
            V = \frac{\hbar}{2}I_N \text{and } r = \bar{0}_N.

        Its ``(A,b,c)`` triple is given by

        .. math::
            A = O_{N\text{x}N}\text{, }b = O_N\text{, and }c = 1.
    """

    short_name = "Vac"

    def __init__(
        self,
        modes: Sequence[int],
    ) -> None:
        super().__init__(modes=tuple(modes), name="Vac")
        for i in range(len(modes)):
            self.fock_shape[i] = 1

    @property
    def representation(self) -> Bargmann:
        n_modes = len(self.modes)
        return Bargmann(*triples.vacuum_state_Abc(n_modes))


#  ~~~~~~~~~~~~
#  Mixed States
#  ~~~~~~~~~~~~


class Thermal(DM):
    r"""
    The `N`-mode thermal state.

    If ``nbar`` is a ``Sequence``, its length must be equal to `1` or `N`. If its length is equal to `1`,
    all the modes share the same ``nbar``.

    .. code-block ::

        >>> from mrmustard.lab_dev import Vacuum

        >>> state = Thermal([1, 2], nbar=3)
        >>> assert state.modes == [1, 2]

    Args:
        modes: A list of modes.
        nbar: The expected number of photons in each mode.
        nbar_trainable: Whether ``nbar`` is trainable.
        nbar_bounds: The bounds of ``nbar``.
    """

    short_name = "Th"

    def __init__(
        self,
        modes: Sequence[int],
        nbar: Union[int, Sequence[int]] = 0,
        nbar_trainable: bool = False,
        nbar_bounds: Tuple[Optional[float], Optional[float]] = (0, None),
    ) -> None:
        super().__init__(modes=modes, name="Thermal")
        self._add_parameter(make_parameter(nbar_trainable, nbar, "nbar", nbar_bounds))

    @property
    def representation(self) -> Bargmann:
        n_modes = len(self.modes)
        nbars = list(reshape_params(n_modes, nbar=self.nbar.value))[0]
        return Bargmann(*triples.thermal_state_Abc(nbars))
