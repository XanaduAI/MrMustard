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

from mrmustard.utils.types import *
from mrmustard import settings
from mrmustard.lab.abstract import Transformation
from mrmustard.utils.parametrized import Parametrized
from mrmustard.utils import training
from mrmustard.physics import gaussian, fock

__all__ = [
    "Dgate",
    "Sgate",
    "Rgate",
    "Ggate",
    "BSgate",
    "MZgate",
    "S2gate",
    "Interferometer",
    "LossChannel",
]


class Dgate(Parametrized, Transformation):
    r"""
    Displacement gate.

    If len(modes) > 1 the gate is applied in parallel to all of the modes provided.

    If a parameter is a single float, the parallel instances of the gate share that parameter.

    To apply mode-specific values use a list of floats.

    One can optionally set bounds for each parameter, which the optimizer will respect.

    Args:
        x (float or List[float]): the list of displacements along the x axis
        x_bounds (float, float): bounds for the displacement along the x axis
        x_trainable (bool): whether x is a trainable variable
        y (float or List[float]): the list of displacements along the y axis
        y_bounds (float, float): bounds for the displacement along the y axis
        y_trainable bool: whether y is a trainable variable
        modes (optional, List[int]): the list of modes this gate is applied to
    """

    def __init__(
        self,
        x: Union[Optional[float], Optional[List[float]]] = 0.0,
        y: Union[Optional[float], Optional[List[float]]] = 0.0,
        x_trainable: bool = False,
        y_trainable: bool = False,
        x_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        y_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        modes: Optional[List[int]] = None,
    ):
        super().__init__(
            x=x,
            y=y,
            x_trainable=x_trainable,
            y_trainable=y_trainable,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            modes=modes,
        )
        self.is_gaussian = True

    @property
    def d_vector(self):
        return gaussian.displacement(self.x, self.y, settings.HBAR)


class Sgate(Parametrized, Transformation):
    r"""
    Squeezing gate.

    If len(modes) > 1 the gate is applied in parallel to all of the modes provided.

    If a parameter is a single float, the parallel instances of the gate share that parameter.

    To apply mode-specific values use a list of floats.

    One can optionally set bounds for each parameter, which the optimizer will respect.

    Args:
        r (float or List[float]): the list of squeezing magnitudes
        r_bounds (float, float): bounds for the squeezing magnitudes
        r_trainable (bool): whether r is a trainable variable
        phi (float or List[float]): the list of squeezing angles
        phi_bounds (float, float): bounds for the squeezing angles
        phi_trainable bool: whether phi is a trainable variable
        modes (optional, List[int]): the list of modes this gate is applied to
    """

    def __init__(
        self,
        r: Union[Optional[float], Optional[List[float]]] = 0.0,
        phi: Union[Optional[float], Optional[List[float]]] = 0.0,
        r_trainable: bool = False,
        phi_trainable: bool = False,
        r_bounds: Tuple[Optional[float], Optional[float]] = (0.0, None),
        phi_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        modes: Optional[List[int]] = None,
    ):
        super().__init__(
            r=r,
            phi=phi,
            r_trainable=r_trainable,
            phi_trainable=phi_trainable,
            r_bounds=r_bounds,
            phi_bounds=phi_bounds,
            modes=modes,
        )
        self.is_gaussian = True

    @property
    def X_matrix(self):
        return gaussian.squeezing_symplectic(self.r, self.phi)


class Rgate(Parametrized, Transformation):
    r"""
    Rotation gate. If len(modes) > 1 the gate is applied in parallel to all of the modes provided.

    If a parameter is a single float, the parallel instances of the gate share that parameter.

    To apply mode-specific values use a list of floats.

    One can optionally set bounds for each parameter, which the optimizer will respect.

    Args:
        modes (List[int]): the list of modes this gate is applied to
        angle (float or List[float]): the list of rotation angles
        angle_bounds (float, float): bounds for the rotation angles
        angle_trainable bool: whether angle is a trainable variable
        modes (optional, List[int]): the list of modes this gate is applied to
    """

    def __init__(
        self,
        angle: Union[Optional[float], Optional[List[float]]] = 0.0,
        angle_trainable: bool = False,
        angle_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        modes: Optional[List[int]] = None,
    ):
        super().__init__(
            angle=angle,
            angle_trainable=angle_trainable,
            angle_bounds=angle_bounds,
            modes=modes,
        )
        self.is_gaussian = True

    @property
    def X_matrix(self):
        return gaussian.rotation_symplectic(self.angle)


class BSgate(Parametrized, Transformation):
    r"""
    Beam splitter gate.

    It applies to a single pair of modes.

    One can optionally set bounds for each parameter, which the optimizer will respect.

    Args:
        theta (float): the transmissivity angle
        theta_bounds (float, float): bounds for the transmissivity angle
        theta_trainable (bool): whether theta is a trainable variable
        phi (float): the phase angle
        phi_bounds (float, float): bounds for the phase angle
        phi_trainable bool: whether phi is a trainable variable
        modes (optional, List[int]): the list of modes this gate is applied to
    """

    def __init__(
        self,
        theta: Optional[float] = 0.0,
        phi: Optional[float] = 0.0,
        theta_trainable: bool = False,
        phi_trainable: bool = False,
        theta_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        phi_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        modes: Optional[List[int]] = None,
    ):
        super().__init__(
            theta=theta,
            phi=phi,
            theta_trainable=theta_trainable,
            phi_trainable=phi_trainable,
            theta_bounds=theta_bounds,
            phi_bounds=phi_bounds,
            modes=modes,
        )
        self.is_gaussian = True

    @property
    def X_matrix(self):
        return gaussian.beam_splitter_symplectic(self.theta, self.phi)

    def _validate_modes(self, modes):
        if len(modes) != 2:
            raise ValueError(
                f"Invalid number of modes: {len(modes)} (should be 2). Perhaps you are looking for Interferometer."
            )


class MZgate(Parametrized, Transformation):
    r"""
    Mach-Zehnder gate.

    sIt supports two conventions:
        1. if `internal=True`, both phases act iside the interferometer: `phi_a` on the upper arm, `phi_b` on the lower arm;
        2. if `internal = False`, both phases act on the upper arm: `phi_a` before the first BS, `phi_b` after the first BS.

    One can optionally set bounds for each parameter, which the optimizer will respect.

    Args:
        phi_a (float): the phase in the upper arm of the MZ interferometer
        phi_a_bounds (float, float): bounds for phi_a
        phi_a_trainable (bool): whether phi_a is a trainable variable
        phi_b (float): the phase in the lower arm or external of the MZ interferometer
        phi_b_bounds (float, float): bounds for phi_b
        phi_b_trainable (bool): whether phi_b is a trainable variable
        internal (bool): whether phases are both in the internal arms (default is False)
        modes (optional, List[int]): the list of modes this gate is applied to
    """

    def __init__(
        self,
        phi_a: Optional[float] = 0.0,
        phi_b: Optional[float] = 0.0,
        phi_a_trainable: bool = False,
        phi_b_trainable: bool = False,
        phi_a_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        phi_b_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        internal: bool = False,
        modes: Optional[List[int]] = None,
    ):
        super().__init__(
            phi_a=phi_a,
            phi_b=phi_b,
            phi_a_trainable=phi_a_trainable,
            phi_b_trainable=phi_b_trainable,
            phi_a_bounds=phi_a_bounds,
            phi_b_bounds=phi_b_bounds,
            internal=internal,
            modes=modes,
        )
        self.is_gaussian = True

    @property
    def X_matrix(self):
        return gaussian.mz_symplectic(self.phi_a, self.phi_b, internal=self._internal)

    def _validate_modes(self, modes):
        if len(modes) != 2:
            raise ValueError(
                f"Invalid number of modes: {len(modes)} (should be 2). Perhaps you are looking for Interferometer?"
            )


class S2gate(Parametrized, Transformation):
    r"""
    Two-mode squeezing gate.

    It applies to a single pair of modes.

    One can optionally set bounds for each parameter, which the optimizer will respect.

    Args:
        r (float): the squeezing magnitude
        r_bounds (float, float): bounds for the squeezing magnitude
        r_trainable (bool): whether r is a trainable variable
        phi (float): the squeezing angle
        phi_bounds (float, float): bounds for the squeezing angle
        phi_trainable bool: whether phi is a trainable variable
        modes (optional, List[int]): the list of modes this gate is applied to
    """

    def __init__(
        self,
        r: Optional[float] = 0.0,
        phi: Optional[float] = 0.0,
        r_trainable: bool = False,
        phi_trainable: bool = False,
        r_bounds: Tuple[Optional[float], Optional[float]] = (0.0, None),
        phi_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        modes: Optional[List[int]] = None,
    ):
        super().__init__(
            r=r,
            phi=phi,
            r_trainable=r_trainable,
            phi_trainable=phi_trainable,
            r_bounds=r_bounds,
            phi_bounds=phi_bounds,
            modes=modes,
        )
        self.is_gaussian = True

    @property
    def X_matrix(self):
        return gaussian.two_mode_squeezing_symplectic(self.r, self.phi)

    def _validate_modes(self, modes):
        if len(modes) != 2:
            raise ValueError(f"Invalid number of modes: {len(modes)} (should be 2")


class Interferometer(Parametrized, Transformation):
    r"""
    N-mode interferometer. It corresponds to a Ggate with zero mean and a `2N x 2N` orthogonal symplectic matrix.

    Args:
        orthogonal (2d array): a valid orthogonal matrix. For N modes it must have shape `(2N,2N)`
        orthogonal_trainable (bool): whether orthogonal is a trainable variable
    """

    def __init__(
        self,
        num_modes: int,
        orthogonal: Optional[Tensor] = None,
        orthogonal_trainable: bool = False,
    ):
        if orthogonal is None:
            orthogonal = training.new_orthogonal(num_modes=num_modes)
        super().__init__(
            orthogonal=orthogonal,
            orthogonal_trainable=orthogonal_trainable,
            orthogonal_bounds=(None, None),
            modes=list(range(num_modes)),
        )
        self.is_gaussian = True

    @property
    def X_matrix(self):
        return self.orthogonal

    def _validate_modes(self, modes):
        if len(modes) != self.orthogonal.shape[1] // 2:
            raise ValueError(f"Invalid number of modes: {len(modes)} (should be {self.orthogonal.shape[1] // 2})")

    @property
    def trainable_parameters(self) -> Dict[str, List[Trainable]]:
        return {
            "symplectic": [],
            "orthogonal": [self.orthogonal] if self._orthogonal_trainable else [],
            "euclidean": [],
        }


class Ggate(Parametrized, Transformation):
    r"""
    A generic N-mode Gaussian unitary transformation with zero displacement.

    If a symplectic matrix is not provided, one will be picked at random with effective squeezing
    strength r in [0,1] for each mode.

    Args:
        num_modes (int): the number of modes this gate is acting on.
        symplectic (2d array): a valid symplectic matrix in XXPP order. For N modes it must have shape `(2N,2N)`.
        symplectic_trainable (bool): whether symplectic is a trainable variable.
    """

    def __init__(
        self,
        num_modes: int,
        symplectic: Optional[Tensor] = None,
        symplectic_trainable: bool = False,
    ):
        if symplectic is None:
            symplectic = training.new_symplectic(num_modes=num_modes)
        super().__init__(
            symplectic=symplectic,
            symplectic_trainable=symplectic_trainable,
            symplectic_bounds=(None, None),
            modes=list(range(num_modes)),
        )
        self.is_gaussian = True

    @property
    def X_matrix(self):
        return self.symplectic

    def _validate_modes(self, modes):
        if len(modes) != self.symplectic.shape[1] // 2:
            raise ValueError(f"Invalid number of modes: {len(modes)} (should be {self.symplectic.shape[1] // 2})")

    @property
    def trainable_parameters(self) -> Dict[str, List[Trainable]]:
        return {
            "symplectic": [self.symplectic] if self._symplectic_trainable else [],
            "orthogonal": [],
            "euclidean": [],
        }


# ~~~~~~~~~~~~~
# NON-UNITARY
# ~~~~~~~~~~~~~


class LossChannel(Parametrized, Transformation):
    r"""
    The lossy bosonic channel.

    If len(modes) > 1 the gate is applied in parallel to all of the modes provided.

    If `transmissivity` is a single float, the parallel instances of the gate share that parameter.

    To apply mode-specific values use a list of floats.

    One can optionally set bounds for `transmissivity`, which the optimizer will respect.

    Args:
        transmissivity (float or List[float]): the list of transmissivities
        transmissivity_bounds (float, float): bounds for the transmissivity
        transmissivity_trainable (bool): whether transmissivity is a trainable variable
        modes (optional, List[int]): the list of modes this gate is applied to
    """

    def __init__(
        self,
        transmissivity: Union[Optional[float], Optional[List[float]]] = 1.0,
        transmissivity_trainable: bool = False,
        transmissivity_bounds: Tuple[Optional[float], Optional[float]] = (0.0, 1.0),
        modes: Optional[List[int]] = None,
    ):
        super().__init__(
            transmissivity=transmissivity,
            transmissivity_trainable=transmissivity_trainable,
            transmissivity_bounds=transmissivity_bounds,
            modes=modes,
        )
        self.is_unitary = False
        self.is_gaussian = True

    @property
    def X_matrix(self):
        return gaussian.loss_X(self.transmissivity)

    @property
    def Y_matrix(self):
        return gaussian.loss_Y(self.transmissivity, settings.HBAR)
