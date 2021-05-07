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
    def _bosonic_loss(self, cov, means, transmissivity, modes) -> Tuple[ArrayLike, ArrayLike]: pass

    @abstractmethod
    def _rotation_symplectic(self, angle) -> ArrayLike: pass

    @abstractmethod
    def _squeezing_symplectic(self, r, phi) -> ArrayLike: pass

    @abstractmethod
    def _beam_splitter_symplectic(self, theta, varphi) -> ArrayLike: pass

    @abstractmethod
    def _two_mode_squeezing_symplectic(self, r:float, phi:float) -> ArrayLike: pass


@dataclass
class ParameterInfo:
    init_value: Optional[float] = None
    trainable: bool = True
    bounds: Tuple[Optional[float], Optional[float]] = (None,None)
    shape:Optional[Tuple[int,...]] = None
    name: str = ''
    

######################
#  CONCRETE CLASSES  #
######################


class BaseBSgate(GateInterface, GateBackendInterface, MathBackendInterface):
    def __init__(self, modes:List[int],
                       theta:Optional[float]=None,
                       theta_bounds:Tuple[Optional[float], Optional[float]]=(None,None),
                       theta_trainable:bool=True,
                       phi:Optional[float]=None,
                       phi_bounds:Tuple[Optional[float], Optional[float]]=(None,None),
                       phi_trainable:bool=True):
        self.modes = modes
        self.mixing = False
        _theta = ParameterInfo(theta, theta_trainable, theta_bounds, None, 'theta')
        _phi = ParameterInfo(phi, phi_trainable, phi_bounds, None, 'phi')
        self._parameters = [self._make_parameter(_theta), self._make_parameter(_phi)]
        self.euclidean_parameters = [p for p,info in zip(self._parameters, [_theta, _phi]) if info.trainable]

    def __call__(self, state:State) -> State:
        BS = self._beam_splitter_symplectic(*self._parameters) # (4x4)
        output = State(state.num_modes)
        output.cov = self._sandwich(bread=BS, filling=state.cov, modes=self.modes)
        output.means = self._matvec(mat=BS, vec=state.means, modes=self.modes)
        return output

    def __repr__(self):
        return f"BSgate(theta={float(self._parameters[0]):.4f}, varphi={float(self._parameters[1]):.4f}, modes={self.modes})"





class BaseDgate(GateInterface, GateBackendInterface):
    def __init__(self, modes:List[int],
                       x:Optional[float]=None,
                       x_bounds:Tuple[Optional[float], Optional[float]]=(None,None),
                       x_trainable:bool=True,
                       y:Optional[float]=None,
                       y_bounds:Tuple[Optional[float], Optional[float]]=(None,None),
                       y_trainable:bool=True):
        self.modes = modes
        self.mixing = False
        _x = ParameterInfo(x, x_trainable, x_bounds, None, 'x')
        _y = ParameterInfo(y, y_trainable, y_bounds, None, 'y')
        self._parameters = [self._make_parameter(_x), self._make_parameter(_y)]
        self.euclidean_parameters = [p for p,info in zip(self._parameters, [_x, _y]) if info.trainable]
            
    def __call__(self, state:State) -> State:
        x,y = self._parameters
        output = State(state.num_modes)
        output.cov = state.cov
        output.means = self._add_at_index(state.means, value=x, index=self.modes)
        output.means = self._add_at_index(output.means, value=y, index=[self.modes[0] + state.num_modes])
        return output

    def __repr__(self):
        return f"Dgate(x={float(self._parameters[0]):.4f}, y={float(self._parameters[1]):.4f}, modes={self.modes})"



class BaseSgate(GateInterface, GateBackendInterface):
    def __init__(self, modes:List[int],
                       r:Optional[float]=None,
                       r_bounds:Tuple[Optional[float], Optional[float]]=(0.0,None),
                       r_trainable:bool=True,
                       phi:Optional[float]=None,
                       phi_bounds:Tuple[Optional[float], Optional[float]]=(None,None),
                       phi_trainable:bool=True):
        self.modes = modes
        self.mixing = False
        _r = ParameterInfo(r, r_trainable, r_bounds, None, 'r')
        _phi = ParameterInfo(phi, phi_trainable, phi_bounds, None, 'phi')
        self._parameters = [self._make_parameter(_r), self._make_parameter(_phi)]
        self.euclidean_parameters = [p for p,info in zip(self._parameters, [_r, _phi]) if info.trainable]

    def __call__(self, state:State) -> State:
        S = self._squeezing_symplectic(*self._parameters)
        output = State(state.num_modes)
        output.cov = self._sandwich(bread=S, filling=state.cov, modes=self.modes)
        output.means = self._matvec(mat=S, vec=state.means, modes=self.modes)
        return output

    def __repr__(self):
        return f"Sgate(r={float(self._parameters[0]):.4f}, phi={float(self._parameters[1]):.4f}, modes={self.modes})"



class BaseGgate(GateInterface, GateBackendInterface):
    def __init__(self, modes:List[int],
                       symp:Optional[ArrayLike]=None,
                       symp_trainable:bool=True,
                       means:Optional[ArrayLike]=None,
                       means_trainable:bool=True):
        self.modes = modes
        self.mixing = False
        self._parameters = []
        if symp is None:
            symp = self._new_symplectic_variable(num_modes=len(self.modes), trainable = symp_trainable, name='CovMatrix')
        self._parameters.append(symp)
        self.euclidean_parameters = [symp] if symp_trainable else []
        _means = ParameterInfo(init_value=means, trainable=means_trainable, shape=(2*len(modes),), name='MeansVector')
        means = self._make_parameter(_means)
        self._parameters.append(means)
        self.euclidean_parameters = [means] if means_trainable else []

    def __call__(self, state:State) -> State:
        symp, means = self._parameters
        output = State(state.num_modes)
        output.cov = self._sandwich(bread=symp, filling=state.cov, modes=self.modes)
        output.means = self._matvec(mat=symp, vec=state.means) + means # TODO fix
        return output
    


class BaseRgate(GateInterface, GateBackendInterface):
    def __init__(self, modes:List[int],
                       angle:Optional[float]=None,
                       angle_bounds:Tuple[Optional[float], Optional[float]]=(None,None),
                       angle_trainable:bool=True):
        self.modes = modes
        self.mixing = False
        _angle = ParameterInfo(angle, angle_trainable, angle_bounds, None, 'angle')
        self._parameters = [self._make_parameter(_angle)]
        self.euclidean_parameters = [p for p,info in zip(self._parameters, [_angle]) if info.trainable]

    def __call__(self, state:State) -> State:
        R = self._rotation_symplectic(*self._parameters)
        output = State(state.num_modes)
        output.cov = self._sandwich(bread=R, filling=state.cov, modes=self.modes)
        output.means = self._matvec(mat=R, vec=state.means, modes=self.modes)
        return output

    def __repr__(self):
        return f"Rgate(angle={float(self._parameters[0]):.4f}, modes={self.modes})"



class BaseLoss(GateInterface, GateBackendInterface):
    def __init__(self, modes:List[int],
                       transmissivity:Optional[float]=None,
                       transmissivity_bounds:Tuple[Optional[float], Optional[float]]=(0.0,1.0),
                       transmissivity_trainable:bool=True):
        self.modes = modes
        self.mixing = False
        _transmissivity = ParameterInfo(transmissivity, transmissivity_trainable, transmissivity_bounds, None, 'transmissivity')
        self._parameters = [self._make_parameter(_transmissivity)]
            
    def __call__(self, state:State) -> State:
        transmissivity = self._parameters
        output = State(state.num_modes)
        output.cov, output.means = self._bosonic_loss(state.cov, state.means, transmissivity, self.modes)
        return output

    def __repr__(self):
        str_params = ', '.join([f'{eta:.2f}' for eta in self._parameters])
        return f"Loss(transmissivity=[{str_params}], modes={self.modes})"

    @property
    def euclidean_parameters(self) -> List[ArrayLike]:
        return [p for p in self._parameters if hasattr(p, 'trainable')]
