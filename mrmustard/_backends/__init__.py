from abc import ABC, abstractproperty, abstractmethod
from typing import List, Sequence, Optional, Tuple

import rich

#rich.pretty.install()


class MathBackendInterface(ABC):
    @abstractmethod
    def identity(self, size: int):
        pass

    @abstractmethod
    def zeros(self, size: int):
        pass

    @abstractmethod
    def sandwich(self, bread: Optional, filling, modes: List[int]):
        pass

    @abstractmethod
    def matvec(self, mat: Optional, vec, modes: List[int]):
        pass

    @abstractmethod
    def add(self, old, new: Optional, modes: List[int]):
        pass

    @abstractmethod
    def concat(self, lst: List):
        pass

    @abstractmethod
    def all_diagonals(self, rho):
        pass

    @abstractmethod
    def modsquare(self, array):
        pass

    @abstractmethod
    def make_symplectic_parameter(
        self, init_value: Optional, trainable: bool, num_modes: int, name: str
    ):
        pass

    @abstractmethod
    def make_euclidean_parameter(
        self,
        init_value: Optional,
        trainable: bool,
        bounds: Tuple[Optional[float], Optional[float]],
        shape: Optional[Sequence[int]],
        name: str,
    ):
        pass
