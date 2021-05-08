from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Sequence, Union
from numpy.typing import ArrayLike
from dataclasses import dataclass

from mrmustard._circuit import GateInterface
from mrmustard._states import State
from mrmustard._backends import MathBackendInterface

# NOTE Gates IMPLEMENT the GateInterface, and USE the GateBackendInterface

################
#  INTERFACES  #
################

class GateBackendInterface(ABC):

    @abstractmethod
    def _loss_X(self, transmissivity, hbar) -> ArrayLike: pass

    @abstractmethod
    def _loss_Y(self, transmissivity, hbar) -> ArrayLike: pass

    @abstractmethod
    def _thermal_X(self, nbar, hbar) -> ArrayLike: pass

    @abstractmethod
    def _thermal_Y(self, nbar, hbar) -> ArrayLike: pass

    @abstractmethod
    def _rotation_symplectic(self, angle) -> ArrayLike: pass

    @abstractmethod
    def _squeezing_symplectic(self, r, phi) -> ArrayLike: pass

    @abstractmethod
    def _beam_splitter_symplectic(self, theta, varphi) -> ArrayLike: pass

    @abstractmethod
    def _two_mode_squeezing_symplectic(self, r:float, phi:float) -> ArrayLike: pass
    

######################
#  CONCRETE CLASSES  #
######################

class BaseGate(GateInterface):
    _backend:MathBackendInterface
    _parameters: List
    _trainable: List

    def _apply_gaussian_channel(self, state, modes, symplectic=None, displacement=None, noise=None):
        output = State(state.num_modes)
        output.cov = self._backend.sandwich(bread=symplectic, filling=state.cov, modes=modes)
        output.cov = self._backend.add(old=output.cov, new=noise, modes=modes)
        output.means = self._backend.matvec(mat=symplectic, vec=state.means, modes=modes)
        output.means = self._backend.add(old=output.means, new=displacement, modes=modes)
        return output

    def __call__(self, state:State) -> State:
        return self._apply_gaussian_channel(state, self.modes, self.symplectic_matrix, self.displacement_vector, self.noise_matrix)

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

    


class BaseBSgate(BaseGate, GateBackendInterface):
    def __init__(self, modes:List[int],
                       theta:Optional[float]=None,
                       theta_bounds:Tuple[Optional[float], Optional[float]]=(None,None),
                       theta_trainable:bool=True,
                       phi:Optional[float]=None,
                       phi_bounds:Tuple[Optional[float], Optional[float]]=(None,None),
                       phi_trainable:bool=True):
        self.modes = modes
        self.mixing = False
        self._trainable = [theta_trainable, phi_trainable]
        self._parameters = [self._backend.make_parameter(theta, theta_trainable, theta_bounds, None, 'theta'),
                            self._backend.make_parameter(phi, phi_trainable, phi_bounds, None, 'phi')]

    @property
    def symplectic_matrix(self) -> ArrayLike:
        return self._beam_splitter_symplectic(*self._parameters)

    def __repr__(self):
        return f"BSgate(theta={float(self._parameters[0]):.4f}, varphi={float(self._parameters[1]):.4f}, modes={self.modes})"



class BaseDgate(BaseGate, GateBackendInterface):
    def __init__(self, modes:List[int],
                       x:Optional[float]=None,
                       x_bounds:Tuple[Optional[float], Optional[float]]=(None,None),
                       x_trainable:bool=True,
                       y:Optional[float]=None,
                       y_bounds:Tuple[Optional[float], Optional[float]]=(None,None),
                       y_trainable:bool=True):
        self.modes = modes
        self.mixing = False
        self._trainable = [x_trainable, y_trainable]
        self._parameters = [self._backend.make_parameter(x, x_trainable, x_bounds, None, 'x'),
                            self._backend.make_parameter(y, y_trainable, y_bounds, None, 'y')]

    @property
    def displacement_vector(self) -> ArrayLike:
        return self._backend.concat(self._parameters)

    def __repr__(self):
        return f"Dgate(x={float(self._parameters[0]):.4f}, y={float(self._parameters[1]):.4f}, modes={self.modes})"



class BaseSgate(BaseGate, GateBackendInterface):
    def __init__(self, modes:List[int],
                       r:Optional[float]=None,
                       r_bounds:Tuple[Optional[float], Optional[float]]=(0.0,None),
                       r_trainable:bool=True,
                       phi:Optional[float]=None,
                       phi_bounds:Tuple[Optional[float], Optional[float]]=(None,None),
                       phi_trainable:bool=True):
        self.modes = modes
        self.mixing = False
        self._trainable = [r_trainable, phi_trainable]
        self._parameters = [self._backend.make_parameter(r, r_trainable, r_bounds, None, 'r'),
                            self._backend.make_parameter(phi, phi_trainable, phi_bounds, None, 'phi')]

    @property
    def symplectic_matrix(self) -> ArrayLike:
        return self._squeezing_symplectic(*self._parameters)

    def __repr__(self):
        return f"Sgate(r={float(self._parameters[0]):.4f}, phi={float(self._parameters[1]):.4f}, modes={self.modes})"



class BaseGgate(BaseGate, GateBackendInterface):
    def __init__(self, modes:List[int],
                       symplectic:Optional[ArrayLike]=None,
                       symplectic_trainable:bool=True,
                       displacement:Optional[ArrayLike]=None,
                       displacement_trainable:bool=True):
        self.modes = modes
        self.mixing = False
        self._trainable = [symp_trainable, displacement_trainable]
        self._parameters =[self._backend.make_symplectic_parameter(symplectic, symplectic_trainable, len(self.modes), 'symplectic'),
                            self._backend.make_parameter(displacement, displacement_trainable, (None,None), (len(self.modes),), 'displacement')]

    @property
    def symplectic_matrix(self) -> ArrayLike:
        return self._parameters[0]

    @property
    def displacement_vector(self) -> ArrayLike:
        return self._parameters[1]
    


class BaseRgate(BaseGate, GateBackendInterface):
    def __init__(self, modes:List[int],
                       angle:Optional[float]=None,
                       angle_bounds:Tuple[Optional[float], Optional[float]]=(None,None),
                       angle_trainable:bool=True):
        self.modes = modes
        self.mixing = False
        self._trainable = [angle_trainable]
        self._parameters = [self._backend.make_parameter(angle, angle_trainable, angle_bounds, None, 'angle')]

    @property
    def symplectic_matrix(self) -> ArrayLike:
        return self._rotation_symplectic(*self._parameters)

    def __repr__(self):
        return f"Rgate(angle={float(self._parameters[0]):.4f}, modes={self.modes})"



class BaseLoss(BaseGate, GateBackendInterface):
    def __init__(self, modes:List[int],
                       transmissivity:Optional[float]=None,
                       transmissivity_bounds:Tuple[Optional[float], Optional[float]]=(0.0,1.0),
                       transmissivity_trainable:bool=False):
        self.modes = modes
        self.mixing = False
        self._trainable = [transmissivity_trainable]
        self._parameters = [self._backend.make_parameter(transmissivity, transmissivity_trainable, transmissivity_bounds, None, 'transmissivity')]

    def __repr__(self):
        str_params = ', '.join([f'{eta:.2f}' for eta in self._parameters])
        return f"Loss(transmissivity=[{str_params}], modes={self.modes})"

    @property
    def symplectic_matrix(self) -> ArrayLike:
        return self._loss_X(*self._parameters)

    @property
    def noise_matrix(self) -> ArrayLike:
        return self._loss_Y(*self._parameters)



class BaseS2gate(BaseGate, GateBackendInterface):
    def __init__(self, modes:List[int],
                       r:Optional[float]=None,
                       r_bounds:Tuple[Optional[float], Optional[float]]=(0.0,None),
                       r_trainable:bool=True,
                       phi:Optional[float]=None,
                       phi_bounds:Tuple[Optional[float], Optional[float]]=(None,None),
                       phi_trainable:bool=True):
        self.modes = modes
        self.mixing = False
        self._trainable = [r_trainable, phi_trainable]
        self._parameters = [self._backend.make_parameter(r, r_trainable, r_bounds, None, 'r'),
                            self._backend.make_parameter(phi, phi_trainable, phi_bounds, None, 'phi')]

    @property
    def symplectic_matrix(self) -> ArrayLike:
        return self._two_mode_squeezing_symplectic(*self._parameters)


    def __repr__(self):
        return f"S2gate(r={float(self._parameters[0]):.4f}, varphi={float(self._parameters[1]):.4f}, modes={self.modes})"
