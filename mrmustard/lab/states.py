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
    r"""The N-mode vacuum state."""

    def __init__(self, num_modes: int):
        cov = gaussian.vacuum_cov(num_modes, settings.HBAR)
        means = gaussian.vacuum_means(num_modes, settings.HBAR)
        State.__init__(self, cov=cov, means=means)


class Coherent(Parametrized, State):
    r"""The N-mode coherent state.

    Equivalent to applying a displacement to the vacuum state:

    .. code-block::

        Coherent(x=0.5, y=0.2) == Vacuum(1) >> Dgate(x=0.5, y=0.3)    # True

    Parallelizable over x and y:

    .. code-block::

        Coherent(x=[1.0, 2.0], y=[-1.0, -2.0]) == Coherent(x=1.0, y=-1.0) & Coherent(x=2.0, y=-2.0)  # True

    Can be used to model a heterodyne detection:

    .. code-block::

        Gaussian(2) << Coherent(x=1.0, y=0.0)[1]  # e.g. heterodyne on mode 1, leftover state on mode 0

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
    r"""The N-mode squeezed vacuum state.

    Equivalent to applying a squeezing gate to the vacuum state:

    .. code::

      >>> SqueezedVacuum(r=0.5, phi=0.2) == Vacuum(1) >> Sgate(r=0.5, phi=0.2)
      True

    Parallelizable over r and phi:
    .. code::

      >>> SqueezedVacuum(r=[1.0, 2.0], phi=[-1.0, -2.0]) == SqueezedVacuum(r=1.0, phi=-1.0) & SqueezedVacuum(r=2.0, phi=-2.0)
      True

    Can be used to model a heterodyne detection with result 0.0:
    .. code::

      >>> Gaussian(2) << SqueezedVacuum(r=10.0, phi=0.0)[1]  # e.g. homodyne on x quadrature on mode 1 with result 0.0
      # leftover state on mode 0

    Args:
        r (float): the squeezing magnitude
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
    r"""The 2-mode squeezed vacuum state.

    Equivalent to applying a 50/50 beam splitter to a pair of squeezed vacuum states:

    .. code::

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
        modes: Optional[Sequence[int]] = [0, 1],
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
    r"""The N-mode thermal state.

    Equivalent to applying additive noise to the vacuum:

    .. code::

        >>> Thermal(nbar=0.31) == Vacuum(1) >> AdditiveNoise(0.62)  # i.e. 2*nbar + 1 (from vac) in total
        True

    Parallelizable over ``nbar``:

    .. code::

        >>> Thermal(nbar=[0.1, 0.2]) == Thermal(nbar=0.1) & Thermal(nbar=0.2)
        True

    Args:
        nbar (float or List[float]): the expected number of photons in each mode
        nbar_trainable (bool): whether the ``nbar`` is trainable
        nbar_bounds (tuple): the bounds of the ``nbar``
        modes (list): the modes of the thermal state
        normalize (bool, default True): when projecting onto Thermal, whether to normalize the leftover state
    """

    def __init__(
        self,
        nbar: Union[Scalar, Vector] = 0.0,
        nbar_trainable: bool = False,
        nbar_bounds: Tuple[Optional[float], Optional[float]] = (0, None),
        modes: Optional[Sequence[int]] = None,
        normalize: bool = True,
    ):
        Parametrized.__init__(
            self,
            nbar=nbar,
            nbar_trainable=nbar_trainable,
            nbar_bounds=nbar_bounds,
            modes=modes,
            normalize=normalize,
        )
        cov = gaussian.thermal_cov(self.nbar, settings.HBAR)
        means = gaussian.vacuum_means(cov.shape[-1] // 2, settings.HBAR)
        State.__init__(self, cov=cov, means=means)

    @property
    def cov(self):
        return gaussian.thermal_cov(self.nbar, settings.HBAR)


class DisplacedSqueezed(Parametrized, State):
    r"""The N-mode displaced squeezed state.

    Equivalent to applying a displacement to the squeezed vacuum state:

    .. code::

        >>> DisplacedSqueezed(r=0.5, phi=0.2, x=0.3, y=-0.7) == SqueezedVacuum(r=0.5, phi=0.2) >> Dgate(x=0.3, y=-0.7)
        True

    Parallelizable over ``r``, ``phi``, ``x``, ``y``:

    .. code::

        >>> DisplacedSqueezed(r=[0.1, 0.2], phi=[0.3, 0.4], x=[0.5, 0.6], y=[0.7, 0.8]) == DisplacedSqueezed(r=0.1, phi=0.3, x=0.5, y=0.7) & DisplacedSqueezed(r=0.2, phi=0.4, x=0.6, y=0.8)
        True

    Can be used to model homodyne detection:

    .. code::

      >>> Gaussian(2) << DisplacedSqueezed(r=10, phi=np.pi, y=0.3)[1]  # e.g. homodyne on mode 1, p quadrature, result 0.3
      # leftover state on mode 0

    Args:
        r (float or List[float]): the squeezing magnitude
        phi (float or List[float]): the squeezing phase
        x (float or List[float]): the displacement in the x direction
        y (float or List[float]): the displacement in the y direction
        r_trainable (bool): whether the squeezing magnitude is trainable
        phi_trainable (bool): whether the squeezing phase is trainable
        x_trainable (bool): whether the displacement in the x direction is trainable
        y_trainable (bool): whether the displacement in the y direction is trainable
        r_bounds (tuple): the bounds of the squeezing magnitude
        phi_bounds (tuple): the bounds of the squeezing phase
        x_bounds (tuple): the bounds of the displacement in the x direction
        y_bounds (tuple): the bounds of the displacement in the y direction
        modes (list): the modes of the displaced squeezed state.
        normalize (bool, default True): when projecting onto DisplacedSqueezed, whether to normalize the leftover state.
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
        modes: Optional[Sequence[int]] = None,
        normalize: bool = True,
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
            modes=modes,
            normalize=normalize,
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
    r"""The N-mode Gaussian state parametrized by a symplectic matrix and N symplectic eigenvalues.

    The (mixed) Gaussian state is equivalent to applying a Gaussian symplectic transformation to a Thermal state:

    .. code::

        >>> G = Gaussian(num_modes=1, eigenvalues = np.random.uniform(settings.HBAR/2, 10.0))
        >>> G == Thermal(nbar=(G.eigenvalues*2/settings.HBAR  - 1)/2) >> Ggate(1, symplectic=G.symplectic)
        True

    Note that the 1st moments are zero unless a Dgate is applied to the Gaussian state:

    .. code::

        >>> np.allclose(Gaussian(num_modes=1).means, 0.0)
        True

    Args:
        num_modes (int): the number of modes
        eigenvalues (float or List[float]): the symplectic eigenvalues of the Gaussian state
        symplectic (np.ndarray or List[np.ndarray]): the symplectic matrix of the Gaussian state
        eigenvalues_trainable (bool): whether the eigenvalues are trainable
        symplectic_trainable (bool): whether the symplectic matrix is trainable
        eigenvalues_bounds (tuple): the bounds of the eigenvalues
        modes (optional, List[int]): the modes of the Gaussian state.
        normalize (bool, default True): when projecting onto Gaussian, whether to normalize the leftover state.
    """

    def __init__(
        self,
        num_modes: int,
        symplectic: Matrix = None,
        eigenvalues: Vector = None,
        symplectic_trainable: bool = False,
        eigenvalues_trainable: bool = False,
        modes: List[int] = None,
        normalize: bool = True,
    ):
        if symplectic is None:
            symplectic = training.new_symplectic(num_modes=num_modes)
        if eigenvalues is None:
            eigenvalues = gaussian.math.ones(num_modes) * settings.HBAR / 2
        if math.any(math.atleast_1d(eigenvalues) < settings.HBAR / 2):
            raise ValueError(
                f"Eigenvalues cannot be smaller than hbar/2 = {settings.HBAR}/2 = {settings.HBAR/2}"
            )
        Parametrized.__init__(
            self,
            symplectic=symplectic,
            eigenvalues=eigenvalues,
            eigenvalues_trainable=eigenvalues_trainable,
            symplectic_trainable=symplectic_trainable,
            eigenvalues_bounds=(settings.HBAR / 2, None),
            symplectic_bounds=(None, None),
            modes=modes,
            normalize=normalize,
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
    r"""The N-mode Fock state.

    Args:
        n (int or List[int]): the number of photons in each mode
        modes (optional, List[int]): the modes of the Fock state
        normalize (bool, default True): when projecting onto Fock, whether to normalize the
            leftover state
    """

    def __init__(self, n: Sequence[int], modes: Sequence[int] = None, normalize: bool = True):
        State.__init__(self, ket=fock.fock_state(n))
        Parametrized.__init__(
            self, n=[n] if isinstance(n, int) else n, modes=modes, normalize=normalize
        )

    def _preferred_projection(self, other: State, mode_indices: Sequence[int]):
        r"""Preferred method to perform a projection onto this state (rather than the default one).

        E.g. ``ket << Fock(1, modes=[3])`` is equivalent to ``ket[:,:,:,1]`` if ``ket`` has 4 modes
        E.g. ``dm << Fock(1, modes=[1])`` is equivalent to ``dm[:,1,:,1]`` if ``dm`` has 2 modes

        Args:
            other: the state to project onto this state
            mode_indices: the indices of the modes of other that we want to project onto self
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
