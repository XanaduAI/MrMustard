from abc import ABC, abstractmethod
from numpy.typing import ArrayLike
from typing import Sequence, Optional
import numpy as np


class StateInterface(ABC):
    cov: Optional[ArrayLike]
    means: Optional[ArrayLike]
    modes: Sequence[int]
    num_modes: int


class State(StateInterface):
    def __init__(self, num_modes: int):
        self.num_modes = num_modes
        self.cov = None
        self.means = None
        # TODO: refactor generic functionality here


class Vacuum(State):
    def __init__(self, num_modes: int):
        super().__init__(num_modes)
        self.cov = np.identity(2 * self.num_modes)
        self.means = np.zeros(2 * self.num_modes)

    def __repr__(self) -> str:
        return f"Vacuum(cov = {self.num_modes}x{self.num_modes} identity, means = {self.num_modes}-dim zero vector)"


class SqueezedVacuum(State):
    def __init__(self, num_modes: int):
        super().__init__(num_modes)
        self.cov = np.identity(2 * self.num_modes)  # TODO
        self.means = np.zeros(2 * self.num_modes)  # TODO


class Coherent(State):
    def __init__(self, num_modes: int):
        super().__init__(num_modes)
        self.cov = np.identity(2 * self.num_modes)  # TODO
        self.means = np.zeros(2 * self.num_modes)  # TODO


class Thermal(State):
    def __init__(self, num_modes: int):
        super().__init__(num_modes)
        self.cov = np.identity(2 * self.num_modes)  # TODO
        self.means = np.zeros(2 * self.num_modes)  # TODO
