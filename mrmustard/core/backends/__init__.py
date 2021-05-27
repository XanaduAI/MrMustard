from abc import ABC, abstractmethod, abstractproperty
from typing import Optional, List, Tuple, Sequence, Callable


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
    def abs(self, array):
        pass

    @abstractmethod
    def new_symplectic_parameter(
        self, init_value: Optional, trainable: bool, num_modes: int, name: str
    ):
        pass

    @abstractmethod
    def new_euclidean_parameter(
        self,
        init_value: Optional,
        trainable: bool,
        bounds: Tuple[Optional[float], Optional[float]],
        shape: Optional[Sequence[int]],
        name: str,
    ):
        pass


class SymplecticBackendInterface(ABC):
    @abstractmethod
    def loss_X(self, transmissivity): ...

    @abstractmethod
    def loss_Y(self, transmissivity, hbar: float): ...

    @abstractmethod
    def thermal_X(self, nbar, hbar: float): ...

    @abstractmethod
    def thermal_Y(self, nbar, hbar: float): ...

    @abstractmethod
    def displacement(self, x, y, hbar: float): ...

    @abstractmethod
    def beam_splitter_symplectic(self, theta, phi): ...

    @abstractmethod
    def rotation_symplectic(self, phi): ...

    @abstractmethod
    def squeezing_symplectic(self, r, phi): ...

    @abstractmethod
    def two_mode_squeezing_symplectic(self, r, phi): ...


class OptimizerBackendInterface(ABC):
    @abstractmethod
    def _loss_and_gradients(self, symplectic_params: Sequence, euclidean_params: Sequence, cost_fn: Callable): ...

    @abstractmethod
    def _update_symplectic(self, symplectic_grads: Sequence, symplectic_params: Sequence): ...

    @abstractmethod
    def _update_euclidean(self, euclidean_grads: Sequence, euclidean_params: Sequence): ...

    @abstractmethod
    def _all_symplectic_parameters(self, items: Sequence): ...

    @abstractmethod
    def _all_euclidean_parameters(self, items: Sequence): ...

    @abstractproperty
    def euclidean_opt(self): ...


class StateBackendInterface(ABC):
    @abstractmethod
    def number_means(self, cov, means, hbar: float): ...

    @abstractmethod
    def number_cov(self, cov, means, hbar: float): ...

    @abstractmethod
    def ABC(self, cov, means, mixed: bool = False, hbar: float = 2.0): ...

    @abstractmethod
    def fock_state(self, A, B, C, cutoffs: Sequence[int]): ...
