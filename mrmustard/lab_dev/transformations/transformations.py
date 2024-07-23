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

__all__ = [
    "Amplifier",
    "Attenuator",
    "BSgate",
    "Dgate",
    "Rgate",
    "Sgate",
    "S2gate",
    "Identity",
]


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

    short_name = "BS"

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

        super().__init__(modes_out=modes, modes_in=modes, name="BSgate")
        self._add_parameter(
            make_parameter(theta_trainable, theta, "theta", theta_bounds)
        )
        self._add_parameter(make_parameter(phi_trainable, phi, "phi", phi_bounds))

        self._representation = Bargmann.from_function(
            fn=triples.beamsplitter_gate_Abc, theta=self.theta, phi=self.phi
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

    short_name = "D"

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
        super().__init__(modes_out=modes, modes_in=modes, name="Dgate")
        xs, ys = list(reshape_params(len(modes), x=x, y=y))
        self._add_parameter(make_parameter(x_trainable, xs, "x", x_bounds))
        self._add_parameter(make_parameter(y_trainable, ys, "y", y_bounds))

        self._representation = Bargmann.from_function(
            fn=triples.displacement_gate_Abc, x=self.x, y=self.y
        )


class Rgate(Unitary):
    r"""
    The rotation gate.

    If ``theta`` is an iterable, its length must be equal to `1` or `N`. If its length is equal to `1`,
    all the modes share the same ``theta``.

    .. code-block ::

        >>> import numpy as np
        >>> from mrmustard.lab_dev import Rgate

        >>> unitary = Rgate(modes=[1, 2], phi=0.1)
        >>> assert unitary.modes == [1, 2]

    Args:
        modes: The modes this gate is applied to.
        theta: The rotation angles.
        theta_bounds: The bounds for ``theta``.
        theta_trainable: Whether ``theta`` is a trainable variable.
    """

    short_name = "R"

    def __init__(
        self,
        modes: Sequence[int],
        phi: Union[float, list[float]] = 0.0,
        phi_trainable: bool = False,
        phi_bounds: Tuple[Optional[float], Optional[float]] = (0.0, None),
    ):
        super().__init__(modes_out=modes, modes_in=modes, name="Rgate")
        (phis,) = list(reshape_params(len(modes), phi=phi))
        self._add_parameter(make_parameter(phi_trainable, phis, "phi", phi_bounds))

        self._representation = Bargmann.from_function(
            fn=triples.rotation_gate_Abc, theta=self.phi
        )


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

    short_name = "S"

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
        super().__init__(modes_out=modes, modes_in=modes, name="Sgate")
        rs, phis = list(reshape_params(len(modes), r=r, phi=phi))
        self._add_parameter(make_parameter(r_trainable, rs, "r", r_bounds))
        self._add_parameter(make_parameter(phi_trainable, phis, "phi", phi_bounds))

        self._representation = Bargmann.from_function(
            fn=triples.squeezing_gate_Abc, r=self.r, delta=self.phi
        )


class Identity(Unitary):
    r"""
    The identity gate.

    Applied to a single or multiple modes

    .. code-block ::

        >>> import numpy as np
        >>> from mrmustard.lab_dev import Identity

        >>> unitary = Identity(modes=[1, 2])
        >>> assert unitary.modes == [1, 2]

    Args:
        modes: The modes this gate is applied to.
    """

    short_name = "I"

    def __init__(
        self,
        modes: Sequence[int],
    ):
        rep = Bargmann.from_function(fn=triples.identity_Abc, n_modes=len(modes))
        super().__init__(
            modes_out=modes, modes_in=modes, representation=rep, name="Identity"
        )


class S2gate(Unitary):
    r"""The two mode squeezing gate.

    It applies to a single pair of modes.

    .. code-block ::

        >>> import numpy as np
        >>> from mrmustard.lab_dev import S2gate

        >>> unitary = S2gate(modes=[1, 2], r=1)
        >>> assert unitary.modes == [1, 2]
        >>> assert np.allclose(unitary.r.value, 1)
        >>> assert np.allclose(unitary.phi.value, 0.0)

    Args:
        modes: The modes this gate is applied to.
        r: The squeezing amplitude.
        r_bounds: The bounds for the squeezing amplitude.
        r_trainable: Whether r is a trainable variable.
        phi: The phase angle.
        phi_bounds: The bounds for the phase angle.
        phi_trainable: Whether phi is a trainable variable.

    Raises:
        ValueError: If ``modes`` is not a pair of modes.

    .. details::

        Its ``(A,b,c)`` triple is given by

        .. math::
            A = \begin{bmatrix}
                    O & e^{i\phi}\tanh(r) & \sech(r) & 0 \\
                    e^{i\phi}\tanh(r) & 0 & 0 & \sech(r) \\
                    \sech(r) & & 0 & 0 e^{i\phi}\tanh(r) \\
                    O & \sech(r) & e^{i\phi}\tanh(r) & 0
                \end{bmatrix} \text{, }
            b = O_{4} \text{, and }
            c = \sech(r)
    """

    def __init__(
        self,
        modes: Tuple[int, int],
        r: float = 0.0,
        phi: float = 0.0,
        r_trainable: bool = False,
        phi_trainable: bool = False,
        r_bounds: Tuple[Optional[float], Optional[float]] = (0, None),
        phi_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
    ):
        if len(modes) != 2:
            raise ValueError(f"Expected a pair of modes, found {modes}.")

        super().__init__(modes_out=modes, modes_in=modes, name="S2gate")
        self._add_parameter(make_parameter(r_trainable, r, "r", r_bounds))
        self._add_parameter(make_parameter(phi_trainable, phi, "phi", phi_bounds))

        self._representation = Bargmann.from_function(
            fn=triples.twomode_squeezing_gate_Abc, r=self.r, phi=self.phi
        )


class Amplifier(Channel):
    r"""The noisy amplifier channel.

    If ``gain`` is an iterable, its length must be equal to `1` or `N`. If it length is equal to `1`,
    all the modes share the same gain.

    .. code-block ::

        >>> import numpy as np
        >>> from mrmustard.lab_dev import Amplifier, Coherent

        >>> amp = Amplifier([0], gain=4)
        >>> coh = Coherent([0], x=1.0, y=2.0)  # units of hbar=2 (default)
        >>> _, mu, _ = (coh >> amp).phase_space(0)
        >>> assert np.allclose(mu[0], np.array([4.0, 8.0]))

    Args:
        modes: The modes this gate is applied to.
        gain: The gain.
        gain_trainable: Whether the gain is a trainable variable.
        gain_bounds: The bounds for the gain.

    .. details::

        The :math:`N`-mode attenuator is defined as

        .. math::
            X = /sqrt{/bar{g}}I_{2N} \text{ , }
            Y = (/bar{g}-1)I_{2N} \text{ , and }
            d = O_{4N}\:,

        where :math:`/bar{g}` is the gain and
        :math:`\text{diag}_N(\bar{g})` is the :math:`N\text{x}N` matrix with diagonal :math:`\bar{g}`.

        Its ``(A,b,c)`` triple is given by

        .. math::
            A &= \begin{bmatrix}
                    O_N & \text{diag}_N(1/(\sqrt{\bar{g}}) & \text{diag}_N(1-1/\bar{g}) & O_N \\
                    \text{diag}_N(1/(\sqrt{\bar{g}}) & O_N & O_N & O_N \\
                    \text{diag}_N(1-1/\bar{g})  & O_N & O_N &  \text{diag}_N(1/(\bar{g})\\
                    O_N & O_N &  \text{diag}_N(1/(\sqrt{\bar{g}}) & O_N
                \end{bmatrix} \\ \\
            b &= O_{4N} \\ \\
            c &= 1//bar{g}\:.
    """

    def __init__(
        self,
        modes: Sequence[int],
        gain: Union[Optional[float], Optional[list[float]]] = 1.0,
        gain_trainable: bool = False,
        gain_bounds: Tuple[Optional[float], Optional[float]] = (1.0, None),
    ):
        super().__init__(modes_out=modes, modes_in=modes, name="Amp")
        (gs,) = list(reshape_params(len(modes), gain=gain))
        self._add_parameter(
            make_parameter(
                gain_trainable,
                gs,
                "gain",
                gain_bounds,
                None,
            )
        )
        self._representation = Bargmann.from_function(
            fn=triples.amplifier_Abc, g=self.gain
        )


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

    short_name = "Att"

    def __init__(
        self,
        modes: Sequence[int],
        transmissivity: Union[Optional[float], Optional[list[float]]] = 1.0,
        transmissivity_trainable: bool = False,
        transmissivity_bounds: Tuple[Optional[float], Optional[float]] = (0.0, 1.0),
    ):
        super().__init__(modes_out=modes, modes_in=modes, name="Att")
        (etas,) = list(reshape_params(len(modes), transmissivity=transmissivity))
        self._add_parameter(
            make_parameter(
                transmissivity_trainable,
                etas,
                "transmissivity",
                transmissivity_bounds,
                None,
            )
        )
        self._representation = Bargmann.from_function(
            fn=triples.attenuator_Abc, eta=self.transmissivity
        )
