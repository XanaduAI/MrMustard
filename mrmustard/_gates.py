from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Sequence, Union
from numpy.typing import ArrayLike
from dataclasses import dataclass

from mrmustard._circuit import GateInterface
from mrmustard._states import State
from mrmustard._backends import MathBackendInterface



# NOTE Gates IMPLEMENT the GateInterface, and USE the GateBackendInterface


class GateBackendInterface(ABC):
    @abstractmethod
    def _bosonic_loss(self, cov, means, transmissivity, modes) -> Tuple[ArrayLike, ArrayLike]: pass

    @abstractmethod
    def _rotation_symplectic(self, angle) -> ArrayLike: pass

    @abstractmethod
    def _squeezing_symplectic(self, r, phi) -> ArrayLike: pass

    @abstractmethod
    def _beam_splitter_symplectic(self, theta, varphi) -> ArrayLike: pass


@dataclass
class ParameterInfo:
    init_value: Optional[float] = None
    trainable: bool = True
    bounds: Tuple[Optional[float], Optional[float]] = (None,None)
    shape:Optional[Tuple[int,...]] = None
    name: str = ''
    

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

    def __call__(self, state:State) -> State:
        BS = self._beam_splitter_symplectic(*self._parameters) # (4x4)
        output = State(state.num_modes)
        output.cov = self._sandwich(bread=BS, filling=state.cov, modes=self.modes)
        output.means = self._matvec(mat=BS, vec=state.means, modes=self.modes)
        return output

    def __repr__(self):
        return f"BSgate(theta={float(self._parameters[0]):.4f}, varphi={float(self._parameters[1]):.4f}, modes={self.modes})"

    @property
    def euclidean_parameters(self) -> List[ArrayLike]:
        return [p for p in self._parameters if hasattr(p, 'trainable')]



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
            
    def __call__(self, state:State) -> State:
        x,y = self._parameters
        output = State(state.num_modes)
        output.cov = state.cov
        output.means = self._add_at_index(state.means, value=x, index=self.modes)
        output.means = self._add_at_index(output.means, value=y, index=[self.modes[0] + state.num_modes])
        return output

    def __repr__(self):
        return f"Dgate(x={float(self._parameters[0]):.4f}, y={float(self._parameters[1]):.4f}, modes={self.modes})"

    @property
    def euclidean_parameters(self) -> List[ArrayLike]:
        return [p for p in self._parameters if hasattr(p, 'trainable')]



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

    def __call__(self, state:State) -> State:
        r, phi = self._parameters
        S = self._expand(self._squeezing_symplectic(r, phi), self.modes, state.num_modes)
        output = State(state.num_modes)
        output.cov = self._sandwich(bread=S, filling=state.cov, modes=self.modes)
        output.means = self._matvec(mat=S, vec=state.means, modes=self.modes)
        return output

    def __repr__(self):
        return f"Sgate(r={float(self._parameters[0]):.4f}, phi={float(self._parameters[1]):.4f}, modes={self.modes})"

    @property
    def euclidean_parameters(self) -> List:
        return [p for p in self._parameters if hasattr(p, 'trainable')]



class BaseGgate(GateInterface, GateBackendInterface):
    def __init__(self, modes:List[int], with_displacement:bool=False):
        self.modes = modes
        self.mixing = False
        self._parameters = []
        self._parameters[0] = self._new_symplectic_variable(num_modes=len(self.modes), name='CovMatrix')
        if with_displacement:
            _means_vector = ParameterInfo(trainable=True, shape=(2*len(modes),), name='MeansVector')
        else:
            _means_vector = ParameterInfo(init_value= np.zeros(2*len(modes)), trainable=False, name='MeansVector')
        self._parameters[1] = self._make_parameter(_means_vector)

    def __call__(self, state:State) -> State:
        cov, means = self._parameters
        output = State(state.num_modes)
        output.cov = self._sandwich(bread=cov, filling=state.cov, modes=self.modes)
        output.means = self._matvec(mat=S, vec=state.means) + means # TODO fix
        return output

    # def __repr__(self):
    #     return f"Ggate(cov={self._parameters[0]:.4f}, means={self._parameters[1]:.4f}, modes={self.modes})"

    @property
    def symplectic_parameters(self) -> List[ArrayLike]:
        cov = self._parameters[0]
        if hasattr(cov, 'trainable'):
            return [cov]
        return []
    @property
    def euclidean_parameters(self) -> List[ArrayLike]:
        means = self._parameters[1]
        if hasattr(means, 'trainable'):
            return [means]
        return []
    


class BaseRgate(GateInterface, GateBackendInterface):
    def __init__(self, modes:List[int],
                       angle:Optional[float]=None,
                       angle_bounds:Tuple[Optional[float], Optional[float]]=(None,None),
                       angle_trainable:bool=True):
        self.modes = modes
        self.mixing = False
        _angle = ParameterInfo(angle, angle_trainable, angle_bounds, None, 'angle')
        self._parameters = [self._make_parameter(_angle)]

    def __call__(self, state:State) -> State:
        angle, = self._parameters
        R = self._expand(self._rotation_symplectic(angle), self.modes, state.num_modes)
        output = State(state.num_modes)
        output.cov = self._sandwich(bread=R, filling=state.cov, modes=self.modes)
        output.means = self._matvec(mat=R, vec=state.means, modes=self.modes)
        return output

    def __repr__(self):
        return f"Rgate(angle={float(self._parameters[0]):.4f}, modes={self.modes})"

    @property
    def euclidean_parameters(self) -> List[ArrayLike]:
        return [p for p in self._parameters if hasattr(p, 'trainable')]



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
