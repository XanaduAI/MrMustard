from abc import ABC, abstractmethod
import numpy as np


class StateBackendInterface(ABC):
    @abstractmethod
    def photon_number_mean(self, cov, means, hbar: float):
        pass

    @abstractmethod
    def photon_number_covariance(self, cov, means, hbar: float):
        pass


class State:
    def __init__(self, num_modes: int, hbar: float = 2.0):
        self.num_modes = num_modes
        self.hbar = hbar

    def photon_number_mean(self):
        return self._state_backend.photon_number_mean(self.cov, self.means, self.hbar)

    def photon_number_covariance(self):
        return self._state_backend.photon_number_covariance(self.cov, self.means, self.hbar)


class Vacuum(State):
    def __init__(self, num_modes: int, hbar: float = 2.0):
        super().__init__(num_modes, hbar)
        self.cov = hbar * np.identity(2 * self.num_modes) / 2.0
        self.means = np.zeros(2 * self.num_modes)

    def __repr__(self) -> str:
        return f"Vacuum(cov = {self.num_modes}x{self.num_modes} identity, means = {self.num_modes}-dim zero vector)"


class SqueezedVacuum(State):
    def __init__(self, num_modes: int, hbar: float = 2.0):
        super().__init__(num_modes, hbar)
        self.cov = np.identity(2 * self.num_modes)  # TODO
        self.means = np.zeros(2 * self.num_modes)  # TODO


class Coherent(State):
    def __init__(self, num_modes: int, hbar: float = 2.0):
        super().__init__(num_modes, hbar)
        self.cov = np.identity(2 * self.num_modes)  # TODO
        self.means = np.zeros(2 * self.num_modes)  # TODO


class Thermal(State):
    def __init__(self, num_modes: int, hbar: float = 2.0):
        super().__init__(num_modes, hbar)
        self.cov = np.identity(2 * self.num_modes)  # TODO
        self.means = np.zeros(2 * self.num_modes)  # TODO
