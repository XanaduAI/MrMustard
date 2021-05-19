from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union, Callable
import numpy as np

from mrmustard._circuit import GateInterface
from mrmustard._states import State
from mrmustard._backends import MathBackendInterface

################
#  INTERFACES  #
################


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


######################
#  CONCRETE CLASSES  #
######################


class Gate(GateInterface):
    _math_backend: MathBackendInterface
    _gate_backend: GateBackendInterface
    _parameters: List
    _trainable: List[bool]
    _param_names: List[str]
    mixing: bool

    def _apply_gaussian_channel(self, state, modes, symplectic=None, displacement=None, noise=None):
        output = State(state.num_modes, hbar=state.hbar)
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
        return [p for i, p in enumerate(self._parameters) if self._trainable[i]]

    @property
    def symplectic_parameters(self) -> List:
        return []


class Dgate(Gate):
    "Displacement gate"

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
    "Squeezing gate"

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
    "Rotation gate"

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
    "Gaussian gate"

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
    "Beam Splitter gate"

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
    "Two-mode Squeezing gate"

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
    "Lossy Bosonic Channel"

    def __init__(
        self,
        modes: List[int],
        transmissivity: Union[Optional[float], Optional[List[float]]] = None,
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
