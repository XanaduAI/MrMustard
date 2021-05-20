from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union, Callable
import numpy as np

from mrmustard._circuit import GateInterface
from mrmustard._states import State
from mrmustard._backends import MathBackendInterface


class GateBackendInterface(ABC):
    @abstractmethod
    def loss_X(self, transmissivity):
        pass

    @abstractmethod
    def loss_Y(self, transmissivity, hbar: float):
        pass

    @abstractmethod
    def thermal_X(self, nbar, hbar: float):
        pass

    @abstractmethod
    def thermal_Y(self, nbar, hbar: float):
        pass

    @abstractmethod
    def rotation_symplectic(self, angle):
        pass

    @abstractmethod
    def displacement(self, x, y, hbar: float):
        pass

    @abstractmethod
    def squeezing_symplectic(self, r, phi):
        pass

    @abstractmethod
    def beam_splitter_symplectic(self, theta, varphi):
        pass

    @abstractmethod
    def two_mode_squeezing_symplectic(self, r, phi):
        pass


class Gate(GateInterface):
    _math_backend: MathBackendInterface
    _gate_backend: GateBackendInterface

    def _apply_gaussian_channel(self, state, modes, symplectic=None, displacement=None, noise=None):
        output = State(state.num_modes, hbar=state.hbar, mixed=noise is not None)
        output.cov = self._math_backend.sandwich(bread=symplectic, filling=state.cov, modes=modes)
        output.cov = self._math_backend.add(old=output.cov, new=noise, modes=modes)
        output.means = self._math_backend.matvec(mat=symplectic, vec=state.means, modes=modes)
        output.means = self._math_backend.add(old=output.means, new=displacement, modes=modes)
        return output

    def __call__(self, state: State) -> State:
        return self._apply_gaussian_channel(
            state,
            self.modes,
            self.symplectic_matrix(state.hbar),
            self.displacement_vector(state.hbar),
            self.noise_matrix(state.hbar),
        )

    def __repr__(self):
        with np.printoptions(precision=3, suppress=True):
            return f"{self.__class__.__qualname__}({self._repr_string(*[str(np.atleast_1d(p)) for p in self._parameters])})"

    def symplectic_matrix(self, hbar: float) -> Optional:
        return None

    def displacement_vector(self, hbar: float) -> Optional:
        return None

    def noise_matrix(self, hbar: float) -> Optional:
        return None

    @property
    def euclidean_parameters(self) -> List:
        return [
            p for i, p in enumerate(self._parameters) if self._trainable[i]
        ]  # NOTE overridden in Ggate

    @property
    def symplectic_parameters(self) -> List:
        return []


class Dgate(Gate):
    r"""
    Displacement gate. If len(modes) > 1 the gate is applied in parallel to all of the modes provided.
    If a parameter is a single float, its value is applied to all of the parallel instances of the gate. To apply mode-specific values use a list of floats.
    If a parameter value is provided, that value will be used.
    If a parameter value is not provided its value is picked at random, unless it's non-trainable in which case its value is set to zero.
    One can optionally set bounds for each parameter, which the optimizer will respect.

    Arguments:
        modes (List[int]): the list of modes this gate is applied to
        x (float or List[float]): the list of displacements along the x axis
        x_bounds (float, float): bounds for the displacement along the x axis
        x_trainable (bool): whether x is a trainable variable
        y (float or List[float]): the list of displacements along the y axis
        y_bounds (float, float): bounds for the displacement along the y axis
        y_trainable bool: whether y is a trainable variable
    """

    def __init__(
        self,
        modes: List[int],
        x: Union[Optional[float], Optional[List[float]]] = None,
        x_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        x_trainable: bool = True,
        y: Union[Optional[float], Optional[List[float]]] = None,
        y_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        y_trainable: bool = True,
    ):
        self._repr_string: Callable[
            [float, float], str
        ] = (
            lambda x, y: f"modes={modes}, x={x}, x_bounds={x_bounds}, x_trainable={x_trainable}, y={y}, y_bounds={y_bounds}, y_trainable={y_trainable}"
        )
        self.modes = modes
        self.mixing = False
        self._trainable = [x_trainable, y_trainable]
        self._parameters = [
            self._math_backend.make_euclidean_parameter(
                x, x_trainable, x_bounds, (len(modes),), "x"
            ),
            self._math_backend.make_euclidean_parameter(
                y, y_trainable, y_bounds, (len(modes),), "y"
            ),
        ]

    def displacement_vector(self, hbar: float):
        return self._gate_backend.displacement(*self._parameters, hbar=hbar)


class Sgate(Gate):
    r"""
    Squeezing gate. If len(modes) > 1 the gate is applied in parallel to all of the modes provided.
    If a parameter is a single float, its value is applied to all of the parallel instances of the gate. To apply mode-specific values use a list of floats.
    If a parameter value is provided, that value will be used.
    If a parameter value is not provided its value is picked at random, unless it's non-trainable in which case its value is set to zero.
    One can optionally set bounds for each parameter, which the optimizer will respect.

    Arguments:
        modes (List[int]): the list of modes this gate is applied to
        r (float or List[float]): the list of squeezing magnitudes
        r_bounds (float, float): bounds for the squeezing magnitudes
        r_trainable (bool): whether r is a trainable variable
        phi (float or List[float]): the list of squeezing angles
        phi_bounds (float, float): bounds for the squeezing angles
        phi_trainable bool: whether phi is a trainable variable
    """

    def __init__(
        self,
        modes: List[int],
        r: Union[Optional[float], Optional[List[float]]] = None,
        r_bounds: Tuple[Optional[float], Optional[float]] = (0.0, None),
        r_trainable: bool = True,
        phi: Union[Optional[float], Optional[List[float]]] = None,
        phi_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        phi_trainable: bool = True,
    ):
        self._repr_string: Callable[
            [float, float], str
        ] = (
            lambda r, phi: f"modes={modes}, r={r}, r_bounds={r_bounds}, r_trainable={r_trainable}, phi={phi}, phi_bounds={phi_bounds}, phi_trainable={phi_trainable}"
        )
        self.modes = modes
        self.mixing = False
        self._trainable = [r_trainable, phi_trainable]
        self._parameters = [
            self._math_backend.make_euclidean_parameter(
                r, r_trainable, r_bounds, (len(modes),), "r"
            ),
            self._math_backend.make_euclidean_parameter(
                phi, phi_trainable, phi_bounds, (len(modes),), "phi"
            ),
        ]

    def symplectic_matrix(self, hbar: float):
        return self._gate_backend.squeezing_symplectic(*self._parameters)


class Rgate(Gate):
    r"""
    Rotation gate. If len(modes) > 1 the gate is applied in parallel to all of the modes provided.
    If a parameter is a single float, its value is applied to all of the parallel instances of the gate. To apply mode-specific values use a list of floats.
    If a parameter value is provided, that value will be used.
    If a parameter value is not provided its value is picked at random, unless it's non-trainable in which case its value is set to zero.
    One can optionally set bounds for each parameter, which the optimizer will respect.

    Arguments:
        modes (List[int]): the list of modes this gate is applied to
        angle (float or List[float]): the list of rotation angles
        angle_bounds (float, float): bounds for the rotation angles
        angle_trainable bool: whether angle is a trainable variable
    """

    def __init__(
        self,
        modes: List[int],
        angle: Union[Optional[float], Optional[List[float]]] = None,
        angle_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        angle_trainable: bool = True,
    ):
        self._repr_string: Callable[
            [float], str
        ] = (
            lambda angle: f"modes={modes}, angle={angle}, angle_bounds={angle_bounds}, angle_trainable={angle_trainable}"
        )
        self.modes = modes
        self.mixing = False
        self._trainable = [angle_trainable]
        self._parameters = [
            self._math_backend.make_euclidean_parameter(
                angle, angle_trainable, angle_bounds, (len(modes),), "angle"
            )
        ]

    def symplectic_matrix(self, hbar: float):
        return self._gate_backend.rotation_symplectic(*self._parameters)


class Ggate(Gate):
    r"""
    General Gaussian gate. If len(modes) == N the gate represents an N-mode Gaussian unitary transformation.
    If a displacement value is not provided its value is picked at random, unless it's non-trainable in which case its value is set to zero.

    Arguments:
        modes (List[int]): the list of modes this gate is applied to
        symplectic (2d array): a valid symplectic matrix. For N modes it must have shape `(2N,2N)`
        symplectic_trainable (bool): whether symplectic is a trainable variable
        displacement (1d array): a displacement vector. For N modes it must have shape `(2N,)`
        displacement_trainable (bool): whether displacement is a trainable variable
    """

    def __init__(
        self,
        modes: List[int],
        symplectic: Optional = None,
        symplectic_trainable: bool = True,
        displacement: Optional = None,
        displacement_trainable: bool = True,
    ):
        self._repr_string: Callable[
            [float, float], str
        ] = (
            lambda symp, disp: f"modes={modes}, symplectic={1}, symplectic_trainable={symplectic_trainable}, displacement={1}, displacement_trainable={displacement_trainable}"
        )
        self.modes = modes
        self.mixing = False
        self._trainable = [symplectic_trainable, displacement_trainable]
        self._parameters = [
            self._math_backend.make_symplectic_parameter(
                symplectic, symplectic_trainable, len(modes), "symplectic"
            ),
            self._math_backend.make_euclidean_parameter(
                displacement,
                displacement_trainable,
                (None, None),
                (2 * len(modes),),
                "displacement",
            ),
        ]

    def symplectic_matrix(self, hbar: float):
        return self._parameters[0]

    def displacement_vector(self, hbar: float):
        return self._parameters[1]

    @property
    def symplectic_parameters(self) -> List:
        return [self.symplectic_matrix(hbar=2.0)] if self._trainable[0] else []

    @property
    def euclidean_parameters(self) -> List:
        return [self.displacement_vector(hbar=2.0)] if self._trainable[1] else []


class BSgate(Gate):
    r"""
    Beam splitter gate. It applies to a single pair of modes.

    Arguments:
        modes (List[int]): the pair of modes to which the beamsplitter is applied to. Must be of length 2.
        theta (float): the transmissivity angle
        theta_bounds (float, float): bounds for the transmissivity angle
        theta_trainable (bool): whether theta is a trainable variable
        phi (float): the phase angle
        phi_bounds (float, float): bounds for the phase angle
        phi_trainable bool: whether phi is a trainable variable
    """

    def __init__(
        self,
        modes: List[int],
        theta: Optional[float] = None,
        theta_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        theta_trainable: bool = True,
        phi: Optional[float] = None,
        phi_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        phi_trainable: bool = True,
    ):
        self._repr_string: Callable[
            [float, float], str
        ] = (
            lambda theta, phi: f"modes={modes}, theta={theta}, theta_bounds={theta_bounds}, theta_trainable={theta_trainable}, phi={phi}, phi_bounds={phi_bounds}, phi_trainable={phi_trainable}"
        )
        if len(modes) > 2:
            raise ValueError(
                "Beam splitter cannot be applied to more than 2 modes. Perhaps you are looking for Interferometer."
            )
        self.modes = modes
        self.mixing = False
        self._trainable = [theta_trainable, phi_trainable]
        self._parameters = [
            self._math_backend.make_euclidean_parameter(
                theta, theta_trainable, theta_bounds, None, "theta"
            ),
            self._math_backend.make_euclidean_parameter(
                phi, phi_trainable, phi_bounds, None, "phi"
            ),
        ]

    def symplectic_matrix(self, hbar: float):
        return self._gate_backend.beam_splitter_symplectic(*self._parameters)


class S2gate(Gate):
    r"""
    Two-mode squeezing gate. It applies to a single pair of modes.

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
        modes: List[int],
        r: Optional[float] = None,
        r_bounds: Tuple[Optional[float], Optional[float]] = (0.0, None),
        r_trainable: bool = True,
        phi: Optional[float] = None,
        phi_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        phi_trainable: bool = True,
    ):
        self._repr_string: Callable[
            [float, float], str
        ] = (
            lambda r, phi: f"modes={modes}, r={r}, r_bounds={r_bounds}, r_trainable={r_trainable}, phi={phi}, phi_bounds={phi_bounds}, phi_trainable={phi_trainable}"
        )
        self.modes = modes
        self.mixing = False
        self._trainable = [r_trainable, phi_trainable]
        self._parameters = [
            self._math_backend.make_euclidean_parameter(r, r_trainable, r_bounds, None, "r"),
            self._math_backend.make_euclidean_parameter(
                phi, phi_trainable, phi_bounds, None, "phi"
            ),
        ]

    def symplectic_matrix(self, hbar: float):
        return self._gate_backend.two_mode_squeezing_symplectic(*self._parameters)


class LossChannel(Gate):
    r"""
    The lossy bosonic channel. If len(modes) > 1 the gate is applied in parallel to all of the modes provided.
    If `transmissivity` is a single float, its value is applied to all of the parallel instances of the gate. To apply mode-specific values use a list of floats.
    If `transmissivity` is not provided, its value is set to 1.
    One can optionally set bounds for each parameter, which the optimizer will respect.

    Arguments:
        modes (List[int]): the list of modes the loss is applied to
        transmissivity (float or List[float]): the list of transmissivities
        transmissivity_bounds (float, float): bounds for the transmissivity
        transmissivity_trainable (bool): whether transmissivity is a trainable variable
    """

    def __init__(
        self,
        modes: List[int],
        transmissivity: Union[Optional[float], Optional[List[float]]] = 1.0,
        transmissivity_bounds: Tuple[Optional[float], Optional[float]] = (0.0, 1.0),
        transmissivity_trainable: bool = False,
        hbar: float = 2.0,
    ):
        self._repr_string: Callable[
            [float], str
        ] = (
            lambda T: f"modes={modes}, transmissivity={T}, transmissivity_bounds={transmissivity_bounds}, transmissivity_trainable={transmissivity_trainable}"
        )
        self.modes = modes
        self.mixing = True
        self.hbar = hbar
        self._trainable = [transmissivity_trainable]
        self._parameters = [
            self._math_backend.make_euclidean_parameter(
                transmissivity,
                transmissivity_trainable,
                transmissivity_bounds,
                (len(modes),),
                "transmissivity",
            )
        ]

    def symplectic_matrix(self, hbar: float):
        return self._gate_backend.loss_X(*self._parameters)

    def noise_matrix(self, hbar: float):
        return self._gate_backend.loss_Y(*self._parameters, hbar=hbar)
