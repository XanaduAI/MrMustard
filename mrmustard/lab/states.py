# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from mrmustard.types import *
from mrmustard import settings
from mrmustard.math import Math

math = Math()
from mrmustard.lab.abstract import State, Transformation
from mrmustard.physics import gaussian, fock
from mrmustard.utils.parametrized import Parametrized
from mrmustard.utils import training

__all__ = [
    "Vacuum",
    "SqueezedVacuum",
    "Coherent",
    "Thermal",
    "DisplacedSqueezed",
    "TMSV",
    "Gaussian",
    "Fock",
]


class Vacuum(State):
    r"""
    The N-mode vacuum state.
    """

    def __init__(self, num_modes: int):
        cov = gaussian.vacuum_cov(num_modes, settings.HBAR)
        means = gaussian.vacuum_means(num_modes, settings.HBAR)
        State.__init__(self, cov=cov, means=means)


class Coherent(Parametrized, State):
    r"""
    The N-mode coherent state. Equivalent to applying a displacement to the vacuum state:
    >>> Coherent(x=0.5, y=0.2) == Vacuum(1) >> Dgate(x=0.5, y=0.3)
    True

    Parallelizable over x and y:
    >>> Coherent(x=[1.0, 2.0], y=[-1.0, -2.0]) == Coherent(x=1.0, y=-1.0) & Coherent(x=2.0, y=-2.0)
    True

    Can be used to model a heterodyne detection:
    >>> Gaussian(2) << Coherent(x=1.0, y=0.0)[1]  # e.g. heterodyne on mode 1
    # leftover state on mode 0

    When used as a measurement, the returned state is always normalized,
    but the probability of the measurement is available as an attribute of the leftover state:
    >>> leftover = Gaussian(2) << Coherent(x=1.0, y=0.0)[1]
    >>> leftover.prob < 1.0
    True

    Note that the values of x and y are automatically rescaled by 1/(2*sqrt(mrmustard.settings.HBAR)).

    Args:
        x (float or List[float]): The x-displacement of the coherent state.
        y (float or List[float]): The y-displacement of the coherent state.
        x_trainable (bool): Whether the x-displacement is trainable.
        y_trainable (bool): Whether the y-displacement is trainable.
        x_bounds (float or None, float or None): The bounds of the x-displacement.
        y_bounds (float or None, float or None): The bounds of the y-displacement.
        modes (optional List[int]): The modes of the coherent state.
        normalize (bool, default True): When projecting onto Coherent, whether to normalize the leftover state.
    """

    def __init__(
        self,
        x: Union[Optional[float], Optional[List[float]]] = 0.0,
        y: Union[Optional[float], Optional[List[float]]] = 0.0,
        x_trainable: bool = False,
        y_trainable: bool = False,
        x_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        y_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        modes: Optional[Sequence[int]] = None,
        normalize: bool = True,
    ):
        Parametrized.__init__(
            self,
            x=x,
            y=y,
            x_trainable=x_trainable,
            y_trainable=y_trainable,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            modes=modes,
            normalize=normalize,
        )
        means = gaussian.displacement(self.x, self.y, settings.HBAR)
        cov = gaussian.vacuum_cov(means.shape[-1] // 2, settings.HBAR)
        State.__init__(self, cov=cov, means=means)

    @property
    def means(self):
        return gaussian.displacement(self.x, self.y, settings.HBAR)


class SqueezedVacuum(Parametrized, State):
    r"""
    The N-mode squeezed vacuum state. Equivalent to applying a squeezing gate to the vacuum state:
    >>> SqueezedVacuum(x=0.5, y=0.2) == Vacuum(1) >> Sgate(x=0.5, y=0.2)
    True

    Parallelizable over r and phi:
    >>> SqueezedVacuum(r=[1.0, 2.0], phi=[-1.0, -2.0]) == SqueezedVacuum(r=1.0, phi=-1.0) & SqueezedVacuum(r=2.0, phi=-2.0)
    True

    Can be used to model a heterodyne detection with result 0.0:
    >>> Gaussian(2) << SqueezedVacuum(r=10.0, phi=0.0)[1]  # e.g. homodyne on x quadrature on mode 1 with result 0.0
    # leftover state on mode 0

    When used as a measurement, the returned state is always normalized,
    but the probability of the measurement is available as an attribute of the leftover state:
    >>> leftover = Gaussian(2) << SqueezedVacuum(r=10.0, phi=0.0)[1]
    >>> leftover.prob < 1.0
    True

    Args:
        r (float): The squeezing magnitude.
        phi (float): The squeezing phase.
        r_trainable (bool): Whether the squeezing magnitude is trainable.
        phi_trainable (bool): Whether the squeezing phase is trainable.
        r_bounds (tuple): The bounds of the squeezing magnitude.
        phi_bounds (tuple): The bounds of the squeezing phase.
        modes (list): The modes of the squeezed vacuum state.
        normalize (bool, default True): When projecting onto SqueezedVacuum, whether to normalize the leftover state.
    """

    def __init__(
        self,
        r: Union[Scalar, Vector] = 0.0,
        phi: Union[Scalar, Vector] = 0.0,
        r_trainable: bool = False,
        phi_trainable: bool = False,
        r_bounds: Tuple[Optional[float], Optional[float]] = (0, None),
        phi_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        modes: Optional[Sequence[int]] = None,
        normalize: bool = True,
    ):
        Parametrized.__init__(
            self,
            r=r,
            phi=phi,
            r_trainable=r_trainable,
            phi_trainable=phi_trainable,
            r_bounds=r_bounds,
            phi_bounds=phi_bounds,
            modes=modes,
            normalize=normalize,
        )
        cov = gaussian.squeezed_vacuum_cov(self.r, self.phi, settings.HBAR)
        means = gaussian.vacuum_means(cov.shape[-1] // 2, settings.HBAR)
        State.__init__(self, cov=cov, means=means)

    @property
    def cov(self):
        return gaussian.squeezed_vacuum_cov(self.r, self.phi, settings.HBAR)


class TMSV(Parametrized, State):
    r"""
    The 2-mode squeezed vacuum state.
    Equivalent to applying a 50/50 beam splitter to a pair of squeezed vacuum states:
    >>> TMSV(r=0.5, phi=0.0) == Vacuum(2) >> Sgate(r=[0.5,0.5], phi=[0.0, np.pi]) >> BSgate(theta=-np.pi/4)
    True

    Args:
        r (float): The squeezing magnitude.
        phi (float): The squeezing phase.
        r_trainable (bool): Whether the squeezing magnitude is trainable.
        phi_trainable (bool): Whether the squeezing phase is trainable.
        r_bounds (tuple): The bounds of the squeezing magnitude.
        phi_bounds (tuple): The bounds of the squeezing phase.
        modes (list): The modes of the two-mode squeezed vacuum state. Must be of length 2.
        normalize (bool, default True): When projecting onto TMSV, whether to normalize the leftover state.
    """

    def __init__(
        self,
        r: Union[Scalar, Vector] = 0.0,
        phi: Union[Scalar, Vector] = 0.0,
        r_trainable: bool = False,
        phi_trainable: bool = False,
        r_bounds: Tuple[Optional[float], Optional[float]] = (0, None),
        phi_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        modes: Optional[Sequence[int]] = [0,1],
        normalize: bool = True,
    ):
        Parametrized.__init__(
            self,
            r=r,
            phi=phi,
            r_trainable=r_trainable,
            phi_trainable=phi_trainable,
            r_bounds=r_bounds,
            phi_bounds=phi_bounds,
            modes=modes,
            normalize=normalize,
        )
        cov = gaussian.two_mode_squeezed_vacuum_cov(self.r, self.phi, settings.HBAR)
        means = gaussian.vacuum_means(2, settings.HBAR)
        State.__init__(self, cov=cov, means=means)

    @property
    def cov(self):
        return gaussian.two_mode_squeezed_vacuum_cov(self.r, self.phi, settings.HBAR)


class Thermal(Parametrized, State):
    r"""
    The N-mode thermal state.
    Equivalent to applying an added noise channel to the vacuum state:
    >>> Thermal(nbar=0.5) == Vacuum(1) >> AdditiveNoise(noise=0.5)
    """

    def __init__(
        self,
        nbar: Union[Scalar, Vector] = 0.0,
        nbar_trainable: bool = False,
        nbar_bounds: Tuple[Optional[float], Optional[float]] = (0, None),
        **kwargs,
    ):
        Parametrized.__init__(
            self, nbar=nbar, nbar_trainable=nbar_trainable, nbar_bounds=nbar_bounds, **kwargs
        )
        cov = gaussian.thermal_cov(self.nbar, settings.HBAR)
        means = gaussian.vacuum_means(cov.shape[-1] // 2, settings.HBAR)
        State.__init__(self, cov=cov, means=means)

    @property
    def cov(self):
        return gaussian.thermal_cov(self.nbar, settings.HBAR)


class DisplacedSqueezed(Parametrized, State):
    r"""
    The N-mode displaced squeezed state.
    """

    def __init__(
        self,
        r: Union[Scalar, Vector] = 0.0,
        phi: Union[Scalar, Vector] = 0.0,
        x: Union[Scalar, Vector] = 0.0,
        y: Union[Scalar, Vector] = 0.0,
        r_trainable: bool = False,
        phi_trainable: bool = False,
        x_trainable: bool = False,
        y_trainable: bool = False,
        r_bounds: Tuple[Optional[float], Optional[float]] = (0, None),
        phi_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        x_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        y_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        **kwargs,
    ):
        Parametrized.__init__(
            self,
            r=r,
            phi=phi,
            x=x,
            y=y,
            r_trainable=r_trainable,
            phi_trainable=phi_trainable,
            x_trainable=x_trainable,
            y_trainable=y_trainable,
            r_bounds=r_bounds,
            phi_bounds=phi_bounds,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            **kwargs,
        )
        cov = gaussian.squeezed_vacuum_cov(self.r, self.phi, settings.HBAR)
        means = gaussian.displacement(self.x, self.y, settings.HBAR)
        State.__init__(self, cov=cov, means=means)

    @property
    def cov(self):
        return gaussian.squeezed_vacuum_cov(self.r, self.phi, settings.HBAR)

    @property
    def means(self):
        return gaussian.displacement(self.x, self.y, settings.HBAR)


class Gaussian(Parametrized, State):
    r"""
    The N-mode Gaussian state.
    """

    def __init__(
        self,
        num_modes: int,
        symplectic: Matrix = None,
        eigenvalues: Vector = None,
        symplectic_trainable: bool = False,
        eigenvalues_trainable: bool = False,
        **kwargs,
    ):
        if symplectic is None:
            symplectic = training.new_symplectic(num_modes=num_modes)
        if eigenvalues is None:
            eigenvalues = gaussian.math.ones(num_modes) * settings.HBAR / 2
        Parametrized.__init__(
            self,
            symplectic=symplectic,
            eigenvalues=eigenvalues,
            eigenvalues_trainable=eigenvalues_trainable,
            symplectic_trainable=symplectic_trainable,
            eigenvalues_bounds=(settings.HBAR / 2, None),
            symplectic_bounds=(None, None),
            **kwargs,
        )
        cov = gaussian.gaussian_cov(self.symplectic, self.eigenvalues, settings.HBAR)
        means = gaussian.vacuum_means(cov.shape[-1] // 2, settings.HBAR)
        State.__init__(self, cov=cov, means=means)

    @property
    def cov(self):
        return gaussian.gaussian_cov(self.symplectic, self.eigenvalues, settings.HBAR)

    @property
    def is_mixed(self):
        return any(self.eigenvalues > settings.HBAR / 2)

    @property
    def trainable_parameters(self) -> Dict[str, List[Trainable]]:
        return {
            "symplectic": [self.symplectic] * self._symplectic_trainable,
            "orthogonal": [],
            "euclidean": ([self.eigenvalues] * self._eigenvalues_trainable),
        }


class Fock(Parametrized, State):
    r"""
    The N-mode Fock state.

    """

    def __init__(self, n: Sequence[int], **kwargs):
        State.__init__(self, ket=fock.fock_state(n))
        Parametrized.__init__(self, n=[n] if isinstance(n, int) else n, **kwargs)

    def _preferred_projection(self, other: State, mode_indices: Sequence[int]):
        r"""
        Preferred method to perform a projection onto this state (rather than the default one).
        E.g. ket << Fock(1, modes=[3]) is equivalent to ket[:,:,:,1] if ket has 4 modes
        E.g. dm << Fock(1, modes=[1]) is equivalent to dm[:,1,:,1] if dm has 2 modes
        Args:
            other: The state to project onto this state.
            mode_indices: The indices of the modes of other that we want to project onto self.
        """
        getitem = []
        used = 0
        for i, m in enumerate(other.modes):
            if i in mode_indices:
                getitem.append(self._n[used])
                used += 1
            else:
                getitem.append(slice(None))
        return other.fock[tuple(getitem)] if self.is_pure else other.fock[tuple(getitem) * 2]
