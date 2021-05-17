from abc import ABC, abstractmethod, abstractproperty
from typing import List, Sequence, Tuple, Optional
from numpy.typing import ArrayLike, DTypeLike
from mrmustard._states import Vacuum, State
from mrmustard._opt import CircuitInterface


################
#  INTERFACES  #
################


class CircuitBackendInterface(ABC):
    @abstractmethod
    def _Sigma_mu_C(
        self, cov: ArrayLike, means: ArrayLike, mixed: bool, hbar: float
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        pass

    @abstractmethod
    def _recursive_state(self, A: ArrayLike, B: ArrayLike, C: ArrayLike, cutoffs: Sequence[int]):
        pass

    @abstractmethod
    def _photon_number_mean(self, cov: ArrayLike, means: ArrayLike, hbar: float) -> ArrayLike:
        pass

    @abstractmethod
    def _photon_number_covariance(self, cov: ArrayLike, means: ArrayLike, hbar: float) -> ArrayLike:
        pass


class GateInterface(ABC):
    modes: List[int]
    mixing: bool
    _parameters: List[ArrayLike]

    @abstractmethod
    def __call__(self, state: State) -> State:
        pass

    @abstractproperty
    def symplectic_matrix(self) -> Optional[ArrayLike]:
        pass

    @abstractproperty
    def displacement_vector(self) -> Optional[ArrayLike]:
        pass

    @abstractproperty
    def noise_matrix(self) -> Optional[ArrayLike]:
        pass

    @abstractproperty
    def euclidean_parameters(self) -> List[ArrayLike]:
        pass

    @abstractproperty
    def symplectic_parameters(self) -> List[ArrayLike]:
        pass


######################
#  CONCRETE CLASSES  #
######################


class BaseCircuit(CircuitInterface, CircuitBackendInterface):
    def __init__(self, num_modes: int):
        self._gates: List[GateInterface] = []
        self.num_modes: int = num_modes
        self._input: State = Vacuum(num_modes=num_modes)
        self._mixed_output: bool = False

    def gaussian_output(self) -> State:
        state = self._input
        for gate in self._gates:
            state = gate(state)
        return state

    def fock_output(self, cutoffs: Sequence[int]) -> ArrayLike:
        output = self.gaussian_output()
        Sigma, mu, C = self._Sigma_mu_C(output.cov, output.means, mixed=self._mixed_output, hbar=2)
        return self._recursive_state(Sigma, mu, C, cutoffs=cutoffs)

    def fock_probabilities(self, cutoffs: Sequence[int]) -> ArrayLike:
        if self._mixed_output:
            rho = self.fock_output(cutoffs=cutoffs)
            return self._all_diagonals(rho)
        else:
            psi = self.fock_output(cutoffs=cutoffs)
            return self._modsquare(psi)

    def photon_number_mean(self, hbar: float = 2) -> ArrayLike:
        gaussian = self.gaussian_output()
        return self._photon_number_mean(gaussian.cov, gaussian.means, hbar)

    def photon_number_covariance(self, hbar: float = 2) -> ArrayLike:
        gaussian = self.gaussian_output()
        return self._photon_number_covariance(gaussian.cov, gaussian.means, hbar)

    @property
    def symplectic_parameters(self) -> List[ArrayLike]:
        return [par for gate in self._gates for par in gate.symplectic_parameters]

    @property
    def euclidean_parameters(self) -> List[ArrayLike]:
        return [par for gate in self._gates for par in gate.euclidean_parameters]

    def add_gate(self, gate: GateInterface) -> None:
        self._gates.append(gate)
        if gate.mixing:
            self._mixed_output = True

    def set_input(self, input: State) -> None:
        self._input = input

    def __repr__(self) -> str:
        return repr(self._input) + "\n" + "\n".join([repr(g) for g in self._gates])
