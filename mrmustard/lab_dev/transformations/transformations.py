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

from typing import Optional, Iterable, Tuple, Union

from mrmustard import math
from .base import Unitary, Channel
from ...physics.representations import Bargmann
from ...physics import triples
from ..utils import make_parameter

__all__ = ["Attenuator", "BSgate", "Dgate", "Sgate"]


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
            S = \\
            d = 

        The ``(A, b, c)`` triple of a beamsplitter is given by   

        .. math::
            A = \begin{bmatrix}
                1 & 4 & 7 \\
                2 & 5 & 8 \\
                3 & 6 & 9
            \end{bmatrix} \\
            b = \\
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
        self._add_parameter(make_parameter(theta_trainable, theta, "theta", theta_bounds))
        self._add_parameter(make_parameter(phi_trainable, phi, "phi", phi_bounds))

    @property
    def representation(self) -> Bargmann:
        return Bargmann(*triples.beamsplitter_gate_Abc(self.theta.value, self.phi.value))


class Dgate(Unitary):
    r"""
    The displacement gate in phase space.

    If ``x`` and/or ``y`` are iterables, their length must be equal to `1` or `N`. If their length is equal to `1`,
    all the modes share the same parameters.

    .. code-block ::

        >>> import numpy as np
        >>> from mrmustard.lab_dev import Dgate

        >>> unitary = Dgate(modes=[1, 2], x=0.1, y=[0.2, 0.3])
        >>> assert unitary.modes == [1, 2]
        >>> assert np.allclose(unitary.x.value, [0.1, 0.1])
        >>> assert np.allclose(unitary.y.value, [0.2, 0.3])

    To apply mode-specific values use a list of floats, one can optionally set bounds for each
    parameter, which the optimizer will respect.

    Args:
        modes: The modes this gate is applied to.
        x: The displacements along the `x` axis.
        x_bounds: The bounds for the displacement along the `x` axis.
        x_trainable: Whether `x` is a trainable variable.
        y: The displacements along the `y` axis.
        y_bounds: The bounds for the displacement along the `y` axis.
        y_trainable: Whether `y` is a trainable variable.

    .. details::

        The displacement gate is a Gaussian gate defined as

        .. math::
            D(\alpha) = \exp(\alpha a^\dagger -\alpha^* a) = \exp\left(-i\sqrt{2}(\re(\alpha) \hat{p} -\im(\alpha) \hat{x})/\sqrt{\hbar}\right)

        where :math:`\alpha = x + iy`.
    """

    def __init__(
        self,
        modes: Iterable[int] = None,
        x: Union[float, Iterable[float]] = 0.0,
        y: Union[float, Iterable[float]] = 0.0,
        x_trainable: bool = False,
        y_trainable: bool = False,
        x_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        y_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
    ) -> None:
        super().__init__("Dgate", modes=modes)
        self._add_parameter(make_parameter(x_trainable, x, "x", x_bounds))
        self._add_parameter(make_parameter(y_trainable, y, "y", y_bounds))

    @property
    def representation(self) -> Bargmann:
        num_modes = len(self.modes)

        xs = math.atleast_1d(self.x.value)
        if len(xs) == 1:
            xs = math.astensor([xs[0] for _ in range(num_modes)])
        ys = math.atleast_1d(self.y.value)
        if len(ys) == 1:
            ys = math.astensor([ys[0] for _ in range(num_modes)])

        return Bargmann(*triples.displacement_gate_Abc(xs, ys))


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

        The squeezing gate is defined as

        .. math::
            S(z) = \exp\left(\frac{1}{2}\left(z^* \a^2-z {\ad}^{2} \right) \right)
            = \exp\left(\frac{r}{2}\left(e^{-i\phi}\a^2 -e^{i\phi}{\ad}^{2} \right) \right)

        Its action on the annihilation and creation operators is given by

        .. math::
            S^\dagger(z) \a S(z) = \a \cosh(r) -\ad e^{i \phi} \sinh r\\
            S^\dagger(z) \ad S(z) = \ad \cosh(r) -\a e^{-i \phi} \sinh(r)\:,

        where :math:`z=r e^{i \phi}` with :math:`r \geq 0` and :math:`\phi \in [0,2 \pi)`.

        The squeezing gate affects the position and momentum operators as

        .. math::
            S^\dagger(z) \x_{\phi} S(z) = e^{-r}\x_{\phi}, ~~~ S^\dagger(z) \p_{\phi} S(z) = e^{r}\p_{\phi}

        The Fock basis decomposition of displacement is given by

        .. math::
            f_{n,m}(r,\phi,\beta)&=\bra{n}\exp\left(\frac{r}{2}\left(e^{i \phi} \a^2
                -e^{-i \phi} \ad \right) \right) D(\beta) \ket{m} = \bra{n}S(z^*) D(\beta) \ket{m}\\
            &=\sqrt{\frac{n!}{\mu  m!}} e^{\frac{\beta ^2 \nu ^*}{2\mu }-\frac{\left| \beta \right| ^2}{2}}
            \sum_{i=0}^{\min(m,n)}\frac{\binom{m}{i} \left(\frac{1}{\mu  \nu }\right)^{i/2}2^{\frac{i-m}{2}
                +\frac{i}{2}-\frac{n}{2}} \left(\frac{\nu }{\mu }\right)^{n/2}
                \left(-\frac{\nu ^*}{\mu }\right)^{\frac{m-i}{2}} H_{n-i}\left(\frac{\beta }{\sqrt{2}
                \sqrt{\mu  \nu }}\right) H_{m-i}\left(-\frac{\alpha ^*}{\sqrt{2}\sqrt{-\mu  \nu ^*}}\right)}{(n-i)!}\:,

        where :math:`\nu=e^{- i\phi} \sinh(r), \mu=\cosh(r), \alpha=\beta \mu - \beta^* \nu`. Two important special
        cases of the last formula are obtained when :math:`r \to 0` and when :math:`\beta \to 0`:

        * For :math:`r \to 0`, we can take :math:`\nu \to 1, \mu \to r, \alpha \to \beta` and use
          the fact that for large :math:`x \gg 1` the leading order term of the Hermite
          polynomials is  :math:`H_n(x) = 2^n x^n +O(x^{n-2})` to obtain

          .. math::
              f_{n,m}(0,\phi,\beta) = \bra{n}D(\beta) \ket{m}=\sqrt{\frac{n!}{  m!}}
              e^{-\frac{\left| \beta \right| ^2}{2}} \sum_{i=0}^{\min(m,n)}
              \frac{(-1)^{m-i}}{(n-i)!} \binom{m}{i} \beta^{n-i} (\beta^*)^{m-i}

        * On the other hand, if we let :math:`\beta\to 0` we use the fact that

          .. math::
              H_n(0) =\begin{cases}0,  & \mbox{if }n\mbox{ is odd} \\
              (-1)^{\tfrac{n}{2}} 2^{\tfrac{n}{2}} (n-1)!! , & \mbox{if }n\mbox{ is even} \end{cases}

          to deduce that :math:`f_{n,m}(r,\phi,0)` is zero if :math:`n` is even and
          :math:`m` is odd or vice versa.

        When writing the Bloch-Messiah reduction of a Gaussian state in the Fock basis, one often needs
        the following matrix element

        .. math::
            \bra{k} D(\alpha) R(\theta) S(r) \ket{l}  = e^{i \theta l }
            \bra{k} D(\alpha) S(r e^{2i \theta}) \ket{l} = e^{i \theta l}
            f^*_{l,k}(-r,-2\theta,-\alpha)
    """

    def __init__(
        self,
        modes: list[int],
        r: Union[float, list[float]] = 0.0,
        phi: Union[float, list[float]] = 0.0,
        r_trainable: bool = False,
        phi_trainable: bool = False,
        r_bounds: Tuple[Optional[float], Optional[float]] = (0.0, None),
        phi_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
    ):
        super().__init__(modes=modes, name="Sgate")
        self._add_parameter(make_parameter(r_trainable, r, "r", r_bounds))
        self._add_parameter(make_parameter(phi_trainable, phi, "phi", phi_bounds))

    @property
    def representation(self) -> Bargmann:
        num_modes = len(self.modes)

        rs = math.atleast_1d(self.r.value)
        if len(rs) == 1:
            rs = math.astensor([rs[0] for _ in range(num_modes)])
        phis = math.atleast_1d(self.phi.value)
        if len(phis) == 1:
            phis = math.astensor([phis[0] for _ in range(num_modes)])

        return Bargmann(*triples.squeezing_gate_Abc(rs, phis))


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

        The attenuator is defined as

        .. math::
            ??@yuan
    """

    def __init__(
        self,
        modes: Optional[Iterable[int]] = None,
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
        eta = self.transmissivity.value
        return Bargmann(*triples.attenuator_Abc(eta))
