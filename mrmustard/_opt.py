from abc import ABC, abstractmethod, abstractproperty
from typing import Callable, Sequence, List, Union, Optional
from mrmustard.visual import Progressbar
from mrmustard._detectors import Detector


class CircuitInterface(ABC):
    "Interface for the Circuit class. Implemented in _circuit.py"

    @abstractproperty
    def symplectic_parameters(self) -> List:
        pass

    @abstractproperty
    def euclidean_parameters(self) -> List:
        pass


class OptimizerBackendInterface(ABC):
    """Interface for the Circuit class.
    Implemented in backends/... and used by the BaseOptimizer"""

    _symplectic_lr: float
    _euclidean_lr: float
    _opt: type
    _backend_opt: type

    @abstractmethod
    def _loss_and_gradients(self, symplectic_params, euclidean_params, cost_fn):
        pass

    @abstractmethod
    def _update_symplectic(self, symplectic_grads, symplectic_params) -> None:
        pass

    @abstractmethod
    def _update_euclidean(self, euclidean_grads, euclidean_params) -> None:
        pass

    @abstractmethod
    def _all_symplectic_parameters(self, items) -> List:
        pass

    @abstractmethod
    def _all_euclidean_parameters(self, items) -> List:
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
        cost_fn: Callable,
        by_optimizing: Sequence[Union[CircuitInterface, Detector]],
        max_steps: int = 1000,
    ) -> Union[Sequence[CircuitInterface], CircuitInterface]:
        r"""
        Optimizes circuits and/or detectors such that the given cost function is minimized.
        Arguments:
            cost_fn (Callable): a function that will be executed in a differentiable context in order to compute gradients as needed
            by_optimizing (list of circuits and/or detectors and/or gates): a list of elements that contain the parameters to optimize
            max_steps (int): the minimization keeps going until the loss is stable or max_steps are reached (if `max_steps=0` it will only stop when the loss is stable)
        """

        optimize = [by_optimizing] if not isinstance(by_optimizing, Sequence) else by_optimizing

        symplectic_parameters = self._all_symplectic_parameters(optimize)
        euclidean_parameters = self._all_euclidean_parameters(optimize)

        bar = Progressbar(max_steps)
        with bar:
            while not self.should_stop(max_steps):
                loss, symp_grads, eucl_grads = self._loss_and_gradients(
                    symplectic_parameters, euclidean_parameters, cost_fn
                )
                self._update_symplectic(symp_grads, symplectic_parameters)
                self._update_euclidean(eucl_grads, euclidean_parameters)
                self.loss_history.append(loss)
                bar.step(loss)

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
