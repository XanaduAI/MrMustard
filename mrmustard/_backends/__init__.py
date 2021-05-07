from abc import ABC, abstractproperty, abstractmethod
from numpy.typing import ArrayLike
from typing import List, Sequence, Optional
# from mrmustard._gates import ParameterInfo

class MathBackendInterface(ABC):

    @abstractmethod
    def _sandwich(self, bread:ArrayLike, filling:ArrayLike, modes:List[int]) -> ArrayLike: pass

    @abstractmethod
    def _matvec(self, mat:ArrayLike, vec:ArrayLike, modes:List[int]) -> ArrayLike: pass

    @abstractmethod
    def _all_diagonals(self, rho: ArrayLike) -> ArrayLike: pass

    @abstractmethod
    def _modsquare(self, array:ArrayLike) -> ArrayLike: pass

    @abstractmethod
    def _add_at_index(self, array:ArrayLike, value:ArrayLike, index:Sequence[int]) -> ArrayLike: pass

    @abstractmethod
    def _make_parameter(self, parinfo): pass