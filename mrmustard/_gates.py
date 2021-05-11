from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Sequence, Union, Callable
from numpy.typing import ArrayLike
from dataclasses import dataclass
import numpy as np

from mrmustard._circuit import GateInterface
from mrmustard._states import State
from mrmustard._backends import MathBackendInterface

# NOTE Gates IMPLEMENT the GateInterface, and USE the GateBackendInterface

################
#  INTERFACES  #
################

class GateBackendInterface(ABC):

    @abstractmethod
    def loss_X(self, transmissivity:ArrayLike) -> ArrayLike: pass

    @abstractmethod
    def loss_Y(self, transmissivity:ArrayLike, hbar:float) -> ArrayLike: pass

    @abstractmethod
    def thermal_X(self, nbar:ArrayLike, hbar:float) -> ArrayLike: pass

    @abstractmethod
    def thermal_Y(self, nbar:ArrayLike, hbar:float) -> ArrayLike: pass

    @abstractmethod
    def rotation_symplectic(self, angle:ArrayLike) -> ArrayLike: pass

    @abstractmethod
    def displacement(self, x:ArrayLike, y:ArrayLike, hbar:float) -> ArrayLike: pass

    @abstractmethod
    def squeezing_symplectic(self, r:ArrayLike, phi:ArrayLike) -> ArrayLike: pass

    @abstractmethod
    def beam_splitter_symplectic(self, theta:ArrayLike, varphi:ArrayLike) -> ArrayLike: pass

    @abstractmethod
    def two_mode_squeezing_symplectic(self, r:ArrayLike, phi:ArrayLike) -> ArrayLike: pass
    

######################
#  CONCRETE CLASSES  #
######################

class Gate(GateInterface):
    _math_backend:MathBackendInterface
    _gate_backend:GateBackendInterface
    _parameters: List
    _trainable: List[bool]
    _param_names: List[str]
    mixing: bool

    def _apply_gaussian_channel(self, state, modes, symplectic=None, displacement=None, noise=None):
        output = State(state.num_modes)
        output.cov = self._math_backend.sandwich(bread=symplectic, filling=state.cov, modes=modes)
        output.cov = self._math_backend.add(old=output.cov, new=noise, modes=modes)
        output.means = self._math_backend.matvec(mat=symplectic, vec=state.means, modes=modes)
        output.means = self._math_backend.add(old=output.means, new=displacement, modes=modes)
        return output

    def __call__(self, state:State) -> State:
        return self._apply_gaussian_channel(state, self.modes, self.symplectic_matrix, self.displacement_vector, self.noise_matrix)

    def __repr__(self):
        with np.printoptions(precision=3, suppress=True):
            return f"{self.__class__.__qualname__}({self._repr_string(*[str(np.atleast_1d(p)) for p in self._parameters])})"

    @property
    def symplectic_matrix(self) -> Optional[ArrayLike]:
        return None

    @property
    def displacement_vector(self) -> Optional[ArrayLike]:
        return None

    @property
    def noise_matrix(self) -> Optional[ArrayLike]:
        return None

    @property
    def euclidean_parameters(self) -> List[ArrayLike]:
        return [p for i,p in enumerate(self._parameters) if self._trainable[i]]

    @property
    def symplectic_parameters(self) -> List[ArrayLike]:
        return []




class Dgate(Gate):
    "Displacement gate"
    def __init__(self, modes:List[int],
                       x:Union[Optional[float], Optional[List[float]]]=None,
                       x_bounds:Tuple[Optional[float], Optional[float]]=(None,None),
                       x_trainable:bool=True,
                       y:Union[Optional[float], Optional[List[float]]]=None,
                       y_bounds:Tuple[Optional[float], Optional[float]]=(None,None),
                       y_trainable:bool=True,
                       hbar:float=2.0):
        self._repr_string:Callable[[float,float],str] = lambda x,y : f'modes={modes}, x={x}, x_bounds={x_bounds}, x_trainable={x_trainable}, y={y}, y_bounds={y_bounds}, y_trainable={y_trainable}, hbar={hbar}'
        self.modes = modes
        self.mixing = False
        self.hbar = hbar
        self._trainable = [x_trainable, y_trainable]
        self._parameters = [self._math_backend.make_euclidean_parameter(x, x_trainable, x_bounds, (len(modes),), 'x'),
                            self._math_backend.make_euclidean_parameter(y, y_trainable, y_bounds, (len(modes),), 'y')]

    @property
    def displacement_vector(self) -> ArrayLike:
        return self._gate_backend.displacement(*self._parameters, hbar=self.hbar)
    



class Sgate(Gate):
    "Squeezing gate"
    def __init__(self, modes:List[int],
                       r:Union[Optional[float], Optional[List[float]]]=None,
                       r_bounds:Tuple[Optional[float], Optional[float]]=(0.0,None),
                       r_trainable:bool=True,
                       phi:Union[Optional[float], Optional[List[float]]]=None,
                       phi_bounds:Tuple[Optional[float], Optional[float]]=(None,None),
                       phi_trainable:bool=True):
        self._repr_string:Callable[[float,float],str] = lambda r,phi : f'modes={modes}, r={r}, r_bounds={r_bounds}, r_trainable={r_trainable}, phi={phi}, phi_bounds={phi_bounds}, phi_trainable={phi_trainable}'
        self.modes = modes
        self.mixing = False
        self._trainable = [r_trainable, phi_trainable]
        self._parameters = [self._math_backend.make_euclidean_parameter(r, r_trainable, r_bounds, (len(modes),), 'r'),
                            self._math_backend.make_euclidean_parameter(phi, phi_trainable, phi_bounds, (len(modes),), 'phi')]

    @property
    def symplectic_matrix(self) -> ArrayLike:
        return self._gate_backend.squeezing_symplectic(*self._parameters)



class Rgate(Gate):
    "Rotation gate"
    def __init__(self, modes:List[int],
                       angle:Union[Optional[float], Optional[List[float]]]=None,
                       angle_bounds:Tuple[Optional[float], Optional[float]]=(None,None),
                       angle_trainable:bool=True):
        self._repr_string:Callable[[float],str] = lambda angle : f'modes={modes}, angle={angle}, angle_bounds={angle_bounds}, angle_trainable={angle_trainable}'
        self.modes = modes
        self.mixing = False
        self._trainable = [angle_trainable]
        self._parameters = [self._math_backend.make_euclidean_parameter(angle, angle_trainable, angle_bounds, (len(modes),), 'angle')]

    @property
    def symplectic_matrix(self) -> ArrayLike:
        return self._gate_backend.rotation_symplectic(*self._parameters)



class Ggate(Gate):
    "Gaussian gate"
    def __init__(self, modes:List[int],
                       symplectic:Optional[ArrayLike]=None,
                       symplectic_trainable:bool=True,
                       displacement:Optional[ArrayLike]=None,
                       displacement_trainable:bool=True):
        self._repr_string:Callable[[float,float],str] = lambda symp,disp : f'modes={modes}, symplectic={1}, symplectic_trainable={symplectic_trainable}, displacement={1}, displacement_trainable={displacement_trainable}'
        self.modes = modes
        self.mixing = False
        self._trainable = [symplectic_trainable, displacement_trainable]
        self._parameters =[self._math_backend.make_symplectic_parameter(symplectic, symplectic_trainable, len(modes), 'symplectic'),
                            self._math_backend.make_euclidean_parameter(displacement, displacement_trainable, (None,None), (2*len(modes),), 'displacement')]

    @property
    def symplectic_matrix(self) -> ArrayLike:
        return self._parameters[0]

    @property
    def displacement_vector(self) -> ArrayLike:
        return self._parameters[1]



class BSgate(Gate):
    "Beam Splitter gate"
    def __init__(self, modes:List[int],
                       theta:Optional[float]=None,
                       theta_bounds:Tuple[Optional[float], Optional[float]]=(None,None),
                       theta_trainable:bool=True,
                       phi:Optional[float]=None,
                       phi_bounds:Tuple[Optional[float], Optional[float]]=(None,None),
                       phi_trainable:bool=True):
        self._repr_string:Callable[[float,float],str] = lambda theta,phi : f'modes={modes}, theta={theta}, theta_bounds={theta_bounds}, theta_trainable={theta_trainable}, phi={phi}, phi_bounds={phi_bounds}, phi_trainable={phi_trainable}'
        self.modes = modes
        self.mixing = False
        self._trainable = [theta_trainable, phi_trainable]
        self._parameters = [self._math_backend.make_euclidean_parameter(theta, theta_trainable, theta_bounds, None, 'theta'),
                            self._math_backend.make_euclidean_parameter(phi, phi_trainable, phi_bounds, None, 'phi')]

    @property
    def symplectic_matrix(self) -> ArrayLike:
        return self._gate_backend.beam_splitter_symplectic(*self._parameters)



class S2gate(Gate):
    "Two-mode Squeezing gate"
    def __init__(self, modes:List[int],
                       r:Optional[float]=None,
                       r_bounds:Tuple[Optional[float], Optional[float]]=(0.0,None),
                       r_trainable:bool=True,
                       phi:Optional[float]=None,
                       phi_bounds:Tuple[Optional[float], Optional[float]]=(None,None),
                       phi_trainable:bool=True):
        self._repr_string:Callable[[float,float],str] = lambda r,phi : f'modes={modes}, r={r}, r_bounds={r_bounds}, r_trainable={r_trainable}, phi={phi}, phi_bounds={phi_bounds}, phi_trainable={phi_trainable}'
        self.modes = modes
        self.mixing = False
        self._trainable = [r_trainable, phi_trainable]
        self._parameters = [self._math_backend.make_euclidean_parameter(r, r_trainable, r_bounds, None, 'r'),
                            self._math_backend.make_euclidean_parameter(phi, phi_trainable, phi_bounds, None, 'phi')]

    @property
    def symplectic_matrix(self) -> ArrayLike:
        return self._gate_backend.two_mode_squeezing_symplectic(*self._parameters)




class LossChannel(Gate):
    "Lossy Bosonic Channel"
    def __init__(self, modes:List[int],
                       transmissivity:Union[Optional[float], Optional[List[float]]]=None,
                       transmissivity_bounds:Tuple[Optional[float], Optional[float]]=(0.0,1.0),
                       transmissivity_trainable:bool=False,
                       hbar:float=2.0):
        self._repr_string:Callable[[float],str] = lambda T : f'modes={modes}, transmissivity={T}, transmissivity_bounds={transmissivity_bounds}, transmissivity_trainable={transmissivity_trainable}'
        self.modes = modes
        self.mixing = True
        self.hbar = hbar
        self._trainable = [transmissivity_trainable]
        self._parameters = [self._math_backend.make_euclidean_parameter(transmissivity, transmissivity_trainable, transmissivity_bounds, (len(modes),2), 'transmissivity')]

    @property
    def symplectic_matrix(self) -> ArrayLike:
        return self._gate_backend.loss_X(*self._parameters)

    @property
    def noise_matrix(self) -> ArrayLike:
        return self._gate_backend.loss_Y(*self._parameters, hbar=self.hbar)

