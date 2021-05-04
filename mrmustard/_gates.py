from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Sequence, Union
from numpy.typing import ArrayLike

from mrmustard._circuit import GateInterface
from mrmustard._states import State

# NOTE Gates IMPLEMENT the GateInterface, and USE the GateBackendInterface


# TODO [improvement]: we may want to apply gates to states that live in a subset of modes
# rather than applying gates on states spanning all the modes

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
    def _expand(self, matrix: ArrayLike, modes: List[int], total_modes:int): pass

    @abstractmethod
    def _sandwich(self, bread:ArrayLike, filling:ArrayLike) -> ArrayLike: pass

    @abstractmethod
    def _matvec(self, mat:ArrayLike, vec:ArrayLike) -> ArrayLike: pass

    @abstractmethod
    def _add_at_index(self, array:ArrayLike, value:ArrayLike, index:Sequence[int]) -> ArrayLike: pass

    @abstractmethod
    def _new_real_variable(self, shape:Optional[Sequence[int]]=None, clip_min:Optional[float]=None,
                            clip_max:Optional[float]=None, name:str='') -> ArrayLike: pass

    @abstractmethod
    def _new_symplectic_variable(self, num_modes:int, name:str) -> ArrayLike: pass

    @abstractmethod
    def _constant(self, value:Optional[float]) -> Optional[ArrayLike]: pass




class BaseBSgate(GateInterface, GateBackendInterface):
    def __init__(self, modes:List[int], theta:Optional[float]=None, phi:Optional[float]=None):
        self.modes = modes
        self.mixing = False
        self._parameters = [self._constant(theta), self._constant(phi)]
        if theta is None:
            self._parameters[0] = self._new_real_variable(name='theta')
        if phi is None:
            self._parameters[1] = self._new_real_variable(name='varphi')

    def __call__(self, state:State) -> State:
        theta, phi = self._parameters
        BS = self._expand(self._beam_splitter_symplectic(theta, phi), self.modes, state.num_modes)
        output = State(state.num_modes)
        output.cov = self._sandwich(bread=BS, filling=state.cov)
        output.means = self._matvec(mat=BS, vec=state.means)
        return output

    def __repr__(self):
        return f"BSgate(theta={float(self._parameters[0]):.4f}, varphi={float(self._parameters[1]):.4f}, modes={self.modes})"

    @property
    def euclidean_parameters(self) -> List[ArrayLike]:
        return [p for p in self._parameters if hasattr(p, 'trainable')]



class BaseDgate(GateInterface, GateBackendInterface):
    def __init__(self, modes:List[int], x:Optional[float]=None, y:Optional[float]=None):
        self.modes = modes
        self.mixing = False
        self._parameters = [self._constant(x), self._constant(y)]
        if x is None:
            self._parameters[0] = self._new_real_variable(name='x')
        if y is None:
            self._parameters[1] = self._new_real_variable(name='y')
            
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
    def __init__(self, modes:List[int], r:Optional[float]=None, phi:Optional[float]=None):
        self.modes = modes
        self.mixing = False
        self._parameters = [self._constant(r), self._constant(phi)]
        if r is None:
            self._parameters[0] = self._new_real_variable(clip_min=0.0, name='r')
        if phi is None:
            self._parameters[1] = self._new_real_variable(name='phi')

    def __call__(self, state:State) -> State:
        r, phi = self._parameters
        S = self._expand(self._squeezing_symplectic(r, phi), self.modes, state.num_modes)
        output = State(state.num_modes)
        output.cov = self._sandwich(bread=S, filling=state.cov)
        output.means = self._matvec(mat=S, vec=state.means)
        return output

    def __repr__(self):
        return f"Sgate(r={float(self._parameters[0]):.4f}, phi={float(self._parameters[1]):.4f}, modes={self.modes})"

    @property
    def euclidean_parameters(self) -> List:
        return [p for p in self._parameters if hasattr(p, 'trainable')]



class BaseGgate(GateInterface, GateBackendInterface):
    def __init__(self, modes:List[int], cov:Optional[ArrayLike]=None, means:Optional[ArrayLike]=None):
        self.modes = modes
        self.mixing = False
        self._parameters = [self._constant(cov), self._constant(means)]
        if cov is None:
            # import pdb
            # pdb.set_trace()
            self._parameters[0] = self._new_symplectic_variable(num_modes=len(self.modes), name='CovMatrix')
        if means is None:
            self._parameters[1] = self._new_real_variable(shape=[2*len(self.modes)], name='MeansVector')

    def __call__(self, state:State) -> State:
        cov, means = self._parameters
        output = State(state.num_modes)
        output.cov = self._sandwich(bread=cov, filling=state.cov)
        output.means = state.means + means
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
    def __init__(self, modes:List[int], angle:Optional[float]=None):
        self.modes = modes
        self.mixing = False
        self._parameters = [self._constant(angle)]
        if angle is None:
            self._parameters[0] = self._new_real_variable(name='angle')

    def __call__(self, state:State) -> State:
        angle, = self._parameters
        R = self._expand(self._rotation_symplectic(angle), self.modes, state.num_modes)
        output = State(state.num_modes)
        output.cov = self._sandwich(bread=R, filling=state.cov)
        output.means = self._matvec(mat=R, vec=state.means)
        return output

    def __repr__(self):
        return f"Rgate(angle={float(self._parameters[0]):.4f}, modes={self.modes})"

    @property
    def euclidean_parameters(self) -> List[ArrayLike]:
        return [p for p in self._parameters if hasattr(p, 'trainable')]



class BaseLoss(GateInterface, GateBackendInterface):
    def __init__(self, modes:Sequence[int], transmissivity:Union[float,Sequence[float]]=None):
        self.modes = modes
        self.mixing = True
        if transmissivity is None:
            self._parameters = [self._new_real_variable(clip_min=0.0, clip_max=1.0, name=f'Transmissivity_{i}') for i in modes]
        if not isinstance(transmissivity, Sequence):
            self._parameters = [self._constant(transmissivity) for _ in modes]
        if isinstance(transmissivity, Sequence):
            self._parameters = [self._constant(t) for t in transmissivity]
            
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
