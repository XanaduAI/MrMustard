from abc import ABC, abstractmethod, abstractproperty
from typing import Callable, Sequence, List, Union
from mrmustard._states import State
from mrmustard.visual import Progressbar
from mrmustard._backends import MathBackendInterface

################
#  INTERFACES  #
################


class CircuitInterface(ABC):
    "Interface for the Circuit class. Implemented in _circuit.py"
    _math_backend = MathBackendInterface

    @abstractproperty
    def symplectic_parameters(self) -> List:
        pass

    @abstractproperty
    def euclidean_parameters(self) -> List:
        pass

    @abstractmethod
    def add_gate(self, gate) -> None:
        pass

    @abstractmethod
    def set_input(self, state) -> None:
        pass

    @abstractmethod
    def gaussian_output(self) -> State:
        pass

    @abstractmethod
    def fock_output(self, cutoffs: Sequence[int]):
        pass

    @abstractmethod
    def fock_probabilities(self, cutoffs: Sequence[int]):
        pass


class OptimizerBackendInterface(ABC):
    """Interface for the Circuit class.
    Implemented in backends/... and used by the BaseOptimizer"""

    _symplectic_lr: float
    _euclidean_lr: float
    _opt: type
    _backend_opt: type

    @abstractmethod
    def _loss_and_gradients(self, symplectic_params, euclidean_params, loss_fn):
        pass

    @abstractmethod
    def _update_symplectic(self, symplectic_grads, symplectic_params) -> None:
        pass

    @abstractmethod
    def _update_euclidean(self, euclidean_grads, euclidean_params) -> None:
        pass

    @abstractmethod
    def _all_symplectic_parameters(self, circuits) -> List:
        pass

    @abstractmethod
    def _all_euclidean_parameters(self, circuits) -> List:
        pass


class BaseOptimizer(OptimizerBackendInterface):
    _backend_opt: type

    def __init__(self, symplectic_lr: float = 1.0, euclidean_lr: float = 0.003):
        self.loss_history: List[float] = [0]
        self._symplectic_lr = symplectic_lr
        self._euclidean_lr = euclidean_lr
        self._opt = self._backend_opt(euclidean_lr)  # from specific backend

    def minimize(
        self,
        circuit: Union[Sequence[CircuitInterface], CircuitInterface],
        loss_fn: Callable,
        max_steps: int = 1000,
    ) -> Union[Sequence[CircuitInterface], CircuitInterface]:

        circuits = [circuit] if not isinstance(circuit, Sequence) else circuit

        symplectic_parameters = self._all_symplectic_parameters(circuits)
        euclidean_parameters = self._all_euclidean_parameters(circuits)

        bar = Progressbar(max_steps)
        with bar:
            while not self.should_stop(max_steps):
                loss, symp_grads, eucl_grads = self._loss_and_gradients(
                    symplectic_parameters, euclidean_parameters, loss_fn
                )
                self._update_symplectic(symp_grads, symplectic_parameters)
                self._update_euclidean(eucl_grads, euclidean_parameters)
                self.loss_history.append(loss)
                bar.step(loss)
        return circuit

    def should_stop(self, max_steps: int) -> bool:
        if max_steps != 0 and len(self.loss_history) > max_steps:
            return True
        if len(self.loss_history) > 5:
            # loss is stable for 5 steps
            if (
                sum(abs(self.loss_history[-i - 1] - self.loss_history[-i]) for i in range(1, 5))
                < 1e-6
            ):
                print("Loss looks stable, stopping here.")
                return True
        return False
