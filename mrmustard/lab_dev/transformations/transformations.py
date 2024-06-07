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
The classes representing transformations in quantum circuits.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

from .base import Unitary, Channel
from ...physics.representations import Bargmann
from ...physics import triples
from ..utils import make_parameter, reshape_params

__all__ = ["Attenuator", "BSgate", "Dgate", "Rgate", "Sgate", "Igate"]


class BSgate(Unitary):
    r"""The beam splitter gate.

    It applies to a single pair of modes.

    .. code-block ::

        >>> import numpy as np
        >>> from mrmustard.lab_dev import BSgate

        >>> unitary = BSgate(modes=[1, 2], theta=0.1)
        >>> assert unitary.modes == [1, 2]
        >>> assert np.allclose(unitary.theta.value, 0.1)
        >>> assert np.allclose(unitary.phi.value, 0.0)

    Args:
        modes: The modes this gate is applied to.
        theta: The transmissivity angle.
        theta_bounds: The bounds for the transmissivity angle.
        theta_trainable: Whether theta is a trainable variable.
        phi: The phase angle.
        phi_bounds: The bounds for the phase angle.
        phi_trainable: Whether phi is a trainable variable.

    Raises:
        ValueError: If ``modes`` is not a pair of modes.

    .. details::

        The beamsplitter gate is a Gaussian gate defined by

        .. math::
            S = \begin{bmatrix}
                    \text{Re}(U) & -\text{Im}(U)\\
                    \text{Im}(U) & \text{Re}(U)
                \end{bmatrix} \text{ and }
            d = O_4\:,

        with

        .. math::
            U &= \begin{bmatrix}
                    \text{cos}(\theta) & -e^{-i\phi}\text{sin}(\theta)\\
                    e^{i\phi}\text{sin}(\theta) & \text{cos}(\theta)
                \end{bmatrix} \\

        Its ``(A,b,c)`` triple is given by 

        .. math::
            A = \begin{bmatrix}
                    O_2 & U \\
                    U^{T} & O_2
                \end{bmatrix} \text{, }
            b = O_{4} \text{, and }
            c = 1
    """

    def __init__(
        self,
        modes: Tuple[int, int],
        theta: float = 0.0,
        phi: float = 0.0,
        theta_trainable: bool = False,
        phi_trainable: bool = False,
        theta_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        phi_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
    ):
        if len(modes) != 2:
            raise ValueError(f"Expected a pair of modes, found {modes}.")

        super().__init__(modes=modes, name="BSgate")
        self._add_parameter(
            make_parameter(theta_trainable, theta, "theta", theta_bounds)
        )
        self._add_parameter(make_parameter(phi_trainable, phi, "phi", phi_bounds))

    @property
    def representation(self) -> Bargmann:
        return Bargmann(
            *triples.beamsplitter_gate_Abc(self.theta.value, self.phi.value)
        )


class Dgate(Unitary):
    r"""
    The displacement gate.

    If ``x`` and/or ``y`` are iterables, their length must be equal to `1` or `N`. If their length is equal to `1`,
    all the modes share the same parameters.

    .. code-block ::

        >>> import numpy as np
        >>> from mrmustard.lab_dev import Dgate

        >>> unitary = Dgate(modes=[1, 2], x=0.1, y=[0.2, 0.3])
        >>> assert unitary.modes == [1, 2]
        >>> assert np.allclose(unitary.x.value, [0.1, 0.1])
        >>> assert np.allclose(unitary.y.value, [0.2, 0.3])

    Args:
        modes: The modes this gate is applied to.
        x: The displacements along the `x` axis, which represents position axis in phase space.
        y: The displacements along the `y` axis.
        x_bounds: The bounds for the displacement along the `x` axis.
        y_bounds: The bounds for the displacement along the `y` axis, which represents momentum axis in phase space.
        x_trainable: Whether `x` is a trainable variable.
        y_trainable: Whether `y` is a trainable variable.

    .. details::

        For any :math:`\bar{\alpha} = \bar{x} + i\bar{y}` of length :math:`N`, the :math:`N`-mode
        displacement gate is defined by

        .. math:: 
            S = I_N \text{ and } r = \sqrt{2\hbar}\big[\text{Re}(\bar{\alpha}), \text{Im}(\bar{\alpha})\big].

        Its ``(A,b,c)`` triple is given by 

        .. math::
            A &= \begin{bmatrix}
                    O_N & I_N\\
                    I_N & O_N
                \end{bmatrix} \\ \\
            b &= \begin{bmatrix}
                    \bar{\alpha} & -\bar{\alpha}^*
                \end{bmatrix} \\ \\
            c &= \text{exp}\big(-|\bar{\alpha}^2|/2\big).
    """

    def __init__(
        self,
        modes: Sequence[int] = None,
        x: Union[float, Sequence[float]] = 0.0,
        y: Union[float, Sequence[float]] = 0.0,
        x_trainable: bool = False,
        y_trainable: bool = False,
        x_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        y_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
    ) -> None:
        super().__init__(modes=modes, name="Dgate")
        self._add_parameter(make_parameter(x_trainable, x, "x", x_bounds))
        self._add_parameter(make_parameter(y_trainable, y, "y", y_bounds))

    @property
    def representation(self) -> Bargmann:
        n_modes = len(self.modes)
        xs, ys = list(reshape_params(n_modes, x=self.x.value, y=self.y.value))
        return Bargmann(*triples.displacement_gate_Abc(xs, ys))


class Rgate(Unitary):
    r"""
    The rotation gate.

    If ``theta`` is an iterable, its length must be equal to `1` or `N`. If its length is equal to `1`,
    all the modes share the same ``theta``.

    .. code-block ::

        >>> import numpy as np
        >>> from mrmustard.lab_dev import Rgate

        >>> unitary = Rgate(modes=[1, 2], theta=0.1)
        >>> assert unitary.modes == [1, 2]

    Args:
        modes: The modes this gate is applied to.
        theta: The rotation angles.
        theta_bounds: The bounds for ``theta``.
        theta_trainable: Whether ``theta`` is a trainable variable.
    """

    def __init__(
        self,
        modes: Sequence[int],
        theta: Union[float, list[float]] = 0.0,
        theta_trainable: bool = False,
        theta_bounds: Tuple[Optional[float], Optional[float]] = (0.0, None),
    ):
        super().__init__(modes=modes, name="Rgate")
        self._add_parameter(
            make_parameter(theta_trainable, theta, "theta", theta_bounds)
        )

    @property
    def representation(self) -> Bargmann:
        n_modes = len(self.modes)
        thetas = list(reshape_params(n_modes, theta=self.theta.value))[0]
        return Bargmann(*triples.rotation_gate_Abc(thetas))


class Sgate(Unitary):
    r"""
    The squeezing gate.

    If ``r`` and/or ``phi`` are iterables, their length must be equal to `1` or `N`. If their length is equal to `1`,
    all the modes share the same parameters.

    .. code-block ::

        >>> import numpy as np
        >>> from mrmustard.lab_dev import Sgate

        >>> unitary = Sgate(modes=[1, 2], r=0.1, phi=[0.2, 0.3])
        >>> assert unitary.modes == [1, 2]
        >>> assert np.allclose(unitary.r.value, [0.1, 0.1])
        >>> assert np.allclose(unitary.phi.value, [0.2, 0.3])

    Args:
        modes: The modes this gate is applied to.
        r: The list of squeezing magnitudes.
        r_bounds: The bounds for the squeezing magnitudes.
        r_trainable: Whether r is a trainable variable.
        phi: The list of squeezing angles.
        phi_bounds: The bounds for the squeezing angles.
        phi_trainable: Whether phi is a trainable variable.

    .. details::

        For any :math:`\bar{r}` and :math:`\bar{\phi}` of length :math:`N`, the :math:`N`-mode
        squeezing gate is defined by

        .. math::
            S = \begin{bmatrix}
                    \text{diag}_N(\text{cosh}(\bar{r})) & \text{diag}_N(e^{-i\bar{\phi}}\text{sinh}(\bar{r}))\\
                    -\text{diag}_N(e^{i\bar{\phi}}\text{sinh}(\bar{r})) & \text{diag}_N(\text{cosh}(\bar{r}))
                \end{bmatrix} \text{ and }
            d = O_{2N},

        where :math:`\text{diag}_N(\bar{a})` is the :math:`N\text{x}N` matrix with diagonal :math:`\bar{a}`.
        Its ``(A,b,c)`` triple is given by 

        .. math::
            A &= \begin{bmatrix}
                    -\text{diag}_N(e^{i\bar{\phi}}\text{tanh}(\bar{r})) & \text{diag}_N(\text{sech}(\bar{r}))\\
                    \text{diag}_N(\text{sech}(\bar{r})) & \text{diag}_N(e^{-i\bar{\phi}}\text{tanh}(\bar{r}))
                \end{bmatrix} \\ \\
            b &= O_{2N} \\ \\
            c &= \prod_{i=1}^N\sqrt{\text{sech}{\:r_i}}\:.
    """

    def __init__(
        self,
        modes: Sequence[int],
        r: Union[float, list[float]] = 0.0,
        phi: Union[float, list[float]] = 0.0,
        r_trainable: bool = False,
        phi_trainable: bool = False,
        r_bounds: Tuple[Optional[float], Optional[float]] = (0.0, None),
        phi_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
    ):
        super().__init__(modes_in=modes, modes_out=modes, name="Sgate")
        self._add_parameter(make_parameter(r_trainable, r, "r", r_bounds))
        self._add_parameter(make_parameter(phi_trainable, phi, "phi", phi_bounds))

    @property
    def representation(self) -> Bargmann:
        n_modes = len(self.modes)
        rs, phis = list(reshape_params(n_modes, r=self.r.value, phi=self.phi.value))
        return Bargmann(*triples.squeezing_gate_Abc(rs, phis))


class Igate(Unitary):
    r"""
    The identity gate.

    Applied to a single or multiple modes

    .. code-block ::

        >>> import numpy as np
        >>> from mrmustard.lab_dev import Igate

        >>> unitary = Igate(modes=[1, 2])
        >>> assert unitary.modes == [1, 2]

    Args:
        modes: The modes this gate is applied to.
    """

    def __init__(
        self,
        modes: Sequence[int],
    ):
        super().__init__(modes=modes, name="Igate")

    @property
    def representation(self) -> Bargmann:
        n_modes = len(self.modes)
        return Bargmann(*triples.identity_Abc(n_modes))


class Attenuator(Channel):
    r"""The noisy attenuator channel.

    If ``transmissivity`` is an iterable, its length must be equal to `1` or `N`. If it length is equal to `1`,
    all the modes share the same transmissivity.

    .. code-block ::

        >>> import numpy as np
        >>> from mrmustard.lab_dev import Attenuator

        >>> channel = Attenuator(modes=[1, 2], transmissivity=0.1)
        >>> assert channel.modes == [1, 2]
        >>> assert np.allclose(channel.transmissivity.value, [0.1, 0.1])

    Args:
        modes: The modes this gate is applied to.
        transmissivity: The transmissivity.
        transmissivity_trainable: Whether the transmissivity is a trainable variable.
        transmissivity_bounds: The bounds for the transmissivity.

    .. details::

        The :math:`N`-mode attenuator is defined as

        .. math::
            X = \text{cos}(\theta)I_{2N} \text{ , }
            Y = \text{sin}^2(\theta)I_{2N} \text{ , and }
            d = O_{4N}\:,

        where the :math:`\theta=\text{arcos}(\sqrt{\bar{\eta}})`, :math:`\eta` is the transmissivity, and 
        :math:`\text{diag}_N(\bar{\eta})` is the :math:`N\text{x}N` matrix with diagonal :math:`\bar{\eta}`.

        Its ``(A,b,c)`` triple is given by 

        .. math::
            A &= \begin{bmatrix}
                    O_N & \text{diag}_N(\sqrt{\bar{\eta}}) & O_N & O_N \\
                    \text{diag}_N(\sqrt{\bar{\eta}}) & O_N & O_N & \text{diag}_N(1-\sqrt{\bar{\eta}})\\
                    O_N & O_N & O_N & \text{diag}_N(\sqrt{\bar{\eta}})\\
                    O_N & \text{diag}_N(1-\sqrt{\bar{\eta}}) & \text{diag}_N(\sqrt{\bar{\eta}}) & O_N
                \end{bmatrix} \\ \\
            b &= O_{4N} \\ \\
            c &= 1\:.
    """

    def __init__(
        self,
        modes: Sequence[int],
        transmissivity: Union[Optional[float], Optional[list[float]]] = 1.0,
        transmissivity_trainable: bool = False,
        transmissivity_bounds: Tuple[Optional[float], Optional[float]] = (0.0, 1.0),
    ):
        super().__init__(modes=modes, name="Att")
        self._add_parameter(
            make_parameter(
                transmissivity_trainable,
                transmissivity,
                "transmissivity",
                transmissivity_bounds,
                None,
            )
        )

    @property
    def representation(self) -> Bargmann:
        n_modes = len(self.modes)
        eta = list(reshape_params(n_modes, eta=self.transmissivity.value))[0]
        return Bargmann(*triples.attenuator_Abc(eta))
