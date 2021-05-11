from abc import ABC, abstractproperty, abstractmethod
from numpy.typing import ArrayLike
from typing import List, Sequence, Optional, Tuple
# from mrmustard._gates import ParameterInfo

class MathBackendInterface(ABC):

    @abstractmethod
    def identity(self, size:int) -> ArrayLike: pass

    @abstractmethod
    def zeros(self, size:int) -> ArrayLike: pass

    @abstractmethod
    def sandwich(self, bread:Optional[ArrayLike], filling:ArrayLike, modes:List[int]) -> ArrayLike: pass

    @abstractmethod
    def matvec(self, mat:Optional[ArrayLike], vec:ArrayLike, modes:List[int]) -> ArrayLike: pass

    @abstractmethod
    def add(self, old:ArrayLike, new:Optional[ArrayLike], modes:List[int]) -> ArrayLike: pass

    @abstractmethod
    def concat(self, lst:List[ArrayLike]) -> ArrayLike: pass

    @abstractmethod
    def all_diagonals(self, rho: ArrayLike) -> ArrayLike: pass

    @abstractmethod
    def modsquare(self, array:ArrayLike) -> ArrayLike: pass

    @abstractmethod
    def make_symplectic_parameter(self, init_value: Optional[ArrayLike],
                                        trainable:bool,
                                        num_modes:int, 
                                        name:str) -> ArrayLike: pass

    @abstractmethod
    def make_euclidean_parameter(self, init_value: Optional[ArrayLike],
                                       trainable: bool,
                                       bounds: Tuple[Optional[float], Optional[float]],
                                       shape:Optional[Sequence[int]],
                                       name: str) -> ArrayLike: pass