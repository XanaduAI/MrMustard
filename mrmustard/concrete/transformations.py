from mrmustard._typing import *
from mrmustard.abstract import Parametrized, Transformation
from mrmustard.functionality import gaussian, train

__all__ = ["Dgate", "Sgate", "Rgate", "Ggate", "BSgate", "MZgate", "S2gate", "Interferometer", "LossChannel"]


class Dgate(Parametrized, Transformation):
    r"""
    Displacement gate. If len(modes) > 1 the gate is applied in parallel to all of the modes provided.
    If a parameter is a single float, the parallel instances of the gate share that parameter.
    To apply mode-specific values use a list of floats.
    One can optionally set bounds for each parameter, which the optimizer will respect.

    Arguments:
        x (float or List[float]): the list of displacements along the x axis
        x_bounds (float, float): bounds for the displacement along the x axis
        x_trainable (bool): whether x is a trainable variable
        y (float or List[float]): the list of displacements along the y axis
        y_bounds (float, float): bounds for the displacement along the y axis
        y_trainable bool: whether y is a trainable variable
    """

    def __init__(
        self,
        x: Union[Optional[float], Optional[List[float]]] = None,
        x_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        x_trainable: bool = True,
        y: Union[Optional[float], Optional[List[float]]] = None,
        y_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        y_trainable: bool = True,
    ):
        super().__init__(x=x, x_bounds=x_bounds, x_trainable=x_trainable, y=y, y_bounds=y_bounds, y_trainable=y_trainable)

    def d_vector(self, hbar: float):
        return gaussian.displacement(self.x, self.y, hbar=hbar)


class Sgate(Parametrized, Transformation):
    r"""
    Squeezing gate. If len(modes) > 1 the gate is applied in parallel to all of the modes provided.
    If a parameter is a single float, the parallel instances of the gate share that parameter.
    To apply mode-specific values use a list of floats.
    One can optionally set bounds for each parameter, which the optimizer will respect.

    Arguments:
        r (float or List[float]): the list of squeezing magnitudes
        r_bounds (float, float): bounds for the squeezing magnitudes
        r_trainable (bool): whether r is a trainable variable
        phi (float or List[float]): the list of squeezing angles
        phi_bounds (float, float): bounds for the squeezing angles
        phi_trainable bool: whether phi is a trainable variable
    """

    def __init__(
        self,
        r: Union[Optional[float], Optional[List[float]]] = None,
        r_bounds: Tuple[Optional[float], Optional[float]] = (0.0, None),
        r_trainable: bool = True,
        phi: Union[Optional[float], Optional[List[float]]] = None,
        phi_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        phi_trainable: bool = True,
    ):
        super().__init__(r=r, r_bounds=r_bounds, r_trainable=r_trainable, phi=phi, phi_bounds=phi_bounds, phi_trainable=phi_trainable)

    def X_matrix(self):
        return gaussian.squeezing_symplectic(self.r, self.phi)


class Rgate(Parametrized, Transformation):
    r"""
    Rotation gate. If len(modes) > 1 the gate is applied in parallel to all of the modes provided.
    If a parameter is a single float, the parallel instances of the gate share that parameter.
    To apply mode-specific values use a list of floats.
    One can optionally set bounds for each parameter, which the optimizer will respect.

    Arguments:
        angle (float or List[float]): the list of rotation angles
        angle_bounds (float, float): bounds for the rotation angles
        angle_trainable bool: whether angle is a trainable variable
    """

    def __init__(
        self,
        angle: Union[Optional[float], Optional[List[float]]] = None,
        angle_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        angle_trainable: bool = True,
    ):
        super().__init__(angle=angle, angle_bounds=angle_bounds, angle_trainable=angle_trainable)

    def X_matrix(self):
        return gaussian.rotation_symplectic(self.angle)


class BSgate(Parametrized, Transformation):
    r"""
    Beam splitter gate. It applies to a single pair of modes.
    One can optionally set bounds for each parameter, which the optimizer will respect.

    Arguments:
        theta (float): the transmissivity angle
        theta_bounds (float, float): bounds for the transmissivity angle
        theta_trainable (bool): whether theta is a trainable variable
        phi (float): the phase angle
        phi_bounds (float, float): bounds for the phase angle
        phi_trainable bool: whether phi is a trainable variable
    """

    def __init__(
        self,
        theta: Optional[float] = None,
        theta_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        theta_trainable: bool = True,
        phi: Optional[float] = None,
        phi_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        phi_trainable: bool = True,
    ):
        super().__init__(
            theta=theta,
            theta_bounds=theta_bounds,
            theta_trainable=theta_trainable,
            phi=phi,
            phi_bounds=phi_bounds,
            phi_trainable=phi_trainable,
        )

    def X_matrix(self):
        return gaussian.beam_splitter_symplectic(self.theta, self.phi)


class MZgate(Parametrized, Transformation):
    r"""
    Mach-Zehnder gate. It supports two conventions:
        1. if `internal=True`, both phases act iside the interferometer: `phi_a` on the upper arm, `phi_b` on the lower arm;
        2. if `internal = False`, both phases act on the upper arm: `phi_a` before the first BS, `phi_b` after the first BS.
    One can optionally set bounds for each parameter, which the optimizer will respect.

    Arguments:
        phi_a (float): the phase in the upper arm of the MZ interferometer
        phi_a_bounds (float, float): bounds for phi_a
        phi_a_trainable (bool): whether phi_a is a trainable variable
        phi_b (float): the phase in the lower arm or external of the MZ interferometer
        phi_b_bounds (float, float): bounds for phi_b
        phi_b_trainable (bool): whether phi_b is a trainable variable
        internal (bool): whether phases are both in the internal arms (default is False)
    """

    def __init__(
        self,
        phi_a: Optional[float] = None,
        phi_a_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        phi_a_trainable: bool = True,
        phi_b: Optional[float] = None,
        phi_b_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        phi_b_trainable: bool = True,
        internal: bool = False,
    ):
        super().__init__(
            phi_a=phi_a,
            phi_a_bounds=phi_a_bounds,
            phi_a_trainable=phi_a_trainable,
            phi_b=phi_b,
            phi_b_bounds=phi_b_bounds,
            phi_b_trainable=phi_b_trainable,
            internal=internal,
        )

    def X_matrix(self):
        return gaussian.mz_symplectic(self.phi_a, self.phi_b, internal=self._internal)


class S2gate(Parametrized, Transformation):
    r"""
    Two-mode squeezing gate. It applies to a single pair of modes.
    One can optionally set bounds for each parameter, which the optimizer will respect.

    Arguments:
        modes (List[int]): the list of modes the two-mode squeezing is applied to. Must be of length 2.
        r (float): the squeezing magnitude
        r_bounds (float, float): bounds for the squeezing magnitude
        r_trainable (bool): whether r is a trainable variable
        phi (float): the squeezing angle
        phi_bounds (float, float): bounds for the squeezing angle
        phi_trainable bool: whether phi is a trainable variable
    """

    def __init__(
        self,
        r: Optional[float] = None,
        r_bounds: Tuple[Optional[float], Optional[float]] = (0.0, None),
        r_trainable: bool = True,
        phi: Optional[float] = None,
        phi_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        phi_trainable: bool = True,
    ):
        super().__init__(r=r, r_bounds=r_bounds, r_trainable=r_trainable, phi=phi, phi_bounds=phi_bounds, phi_trainable=phi_trainable)

    def X_matrix(self):
        return gaussian.two_mode_squeezing_symplectic(self.r, self.phi)


class Interferometer(Parametrized, Transformation):
    r"""
    N-mode interferometer. It corresponds to a Ggate with zero mean and a `2N x 2N` orthogonal symplectic matrix.

    Arguments:
        num_modes (int): the number of modes this gate is acting on
        orthogonal (2d array): a valid orthogonal matrix. For N modes it must have shape `(2N,2N)`
        orthogonal_trainable (bool): whether orthogonal is a trainable variable
    """

    def __init__(self, num_modes: int, orthogonal: Optional[Tensor] = None, orthogonal_trainable: bool = True):
        if orthogonal is None:
            orthogonal = train.new_orthogonal(num_modes=num_modes)
        super().__init__(num_modes=num_modes, orthogonal=orthogonal, orthogonal_bounds=(None, None), orthogonal_trainable=orthogonal_trainable)

    def X_matrix(self):
        return self.orthogonal

    @property
    def trainable_parameters(self) -> Dict[str, List[Trainable]]:
        return {"symplectic": [], "orthogonal": [self.orthogonal] if self._orthogonal_trainable else [], "euclidean": []}


class Ggate(Parametrized, Transformation):
    r"""
    General Gaussian gate. If len(modes) == N the gate represents an N-mode Gaussian unitary transformation.
    If a symplectic matrix is not provided, one will be picked at random with effective squeezings between 0 and 1.

    Arguments:
        num_modes (int): the number of modes this gate is applied to
        symplectic (2d array): a valid symplectic matrix. For N modes it must have shape `(2N,2N)`
        symplectic_trainable (bool): whether symplectic is a trainable variable
        displacement (1d array): a displacement vector. For N modes it must have shape `(2N,)`
        displacement_trainable (bool): whether displacement is a trainable variable
    """

    def __init__(
        self,
        num_modes: int,
        symplectic: Optional[Tensor] = None,
        symplectic_trainable: bool = True,
        displacement: Optional[Tensor] = None,
        displacement_trainable: bool = True,
    ):
        if symplectic is None:
            symplectic = train.new_symplectic(num_modes=num_modes)
        if displacement is None:
            displacement = train.backend.zeros(num_modes * 2)  # TODO: gates should not know about the backend
        super().__init__(
            num_modes=num_modes,
            symplectic=symplectic,
            symplectic_bounds=(None, None),
            symplectic_trainable=symplectic_trainable,
            displacement=displacement,
            displacement_bounds=(None, None),
            displacement_trainable=displacement_trainable,
        )

    def X_matrix(self):
        return self.symplectic

    def d_vector(self, hbar: float):
        return self.displacement

    @property
    def trainable_parameters(self) -> Dict[str, List[Trainable]]:
        return {
            "symplectic": [self.symplectic] if self._symplectic_trainable else [],
            "orthogonal": [],
            "euclidean": [self.displacement] if self._displacement_trainable else [],
        }


#
#  NON-UNITARY
#


class LossChannel(Parametrized, Transformation):
    r"""
    The lossy bosonic channel. If len(modes) > 1 the gate is applied in parallel to all of the modes provided.
    If `transmissivity` is a single float, the parallel instances of the gate share that parameter.
    To apply mode-specific values use a list of floats.
    One can optionally set bounds for `transmissivity`, which the optimizer will respect.

    Arguments:
        modes (List[int]): the list of modes the loss is applied to
        transmissivity (float or List[float]): the list of transmissivities
        transmissivity_bounds (float, float): bounds for the transmissivity
        transmissivity_trainable (bool): whether transmissivity is a trainable variable
    """

    def __init__(
        self,
        transmissivity: Union[Optional[float], Optional[List[float]]] = 1.0,
        transmissivity_bounds: Tuple[Optional[float], Optional[float]] = (0.0, 1.0),
        transmissivity_trainable: bool = False,
    ):
        super().__init__(
            transmissivity=transmissivity,
            transmissivity_bounds=transmissivity_bounds,
            transmissivity_trainable=transmissivity_trainable,
        )

    def X_matrix(self):
        return gaussian.loss_X(self.transmissivity)

    def Y_matrix(self, hbar: float):
        return gaussian.loss_Y(self.transmissivity, hbar=hbar)
