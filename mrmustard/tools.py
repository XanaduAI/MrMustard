from collections.abc import MutableSequence
from typing import List, Callable, Sequence, Union

from mrmustard.core.backends import MathBackendInterface, OptimizerBackendInterface
from mrmustard.core.baseclasses import Gate, State, Detector
from mrmustard.core.visual import Progressbar


class Circuit(MutableSequence):
    _math_backend: MathBackendInterface

    def __init__(self):
        self._gates: List[Gate] = []

    def __call__(self, state: State) -> State:
        state_ = state
        for gate in self._gates:
            state_ = gate(state_)
        return state_

    def __getitem__(self, key):
        return self._gates.__getitem__(key)

    def __setitem__(self, key, value):
        if not isinstance(value, Gate):
            raise ValueError(f"Item {type(value)} is not a gate")
        return self._gates.__setitem__(key, value)

    def __delitem__(self, key):
        return self._gates.__delitem__(key)

    def __len__(self):
        return len(self._gates)

    def __repr__(self) -> str:
        return "\n".join([repr(g) for g in self._gates])

    def insert(self, index, object):
        return self._gates.insert(index, object)

    @property
    def symplectic_parameters(self) -> List:
        r"""
        Returns the list of symplectic parameters
        """
        return [par for gate in self._gates for par in gate.symplectic_parameters]

    @property
    def euclidean_parameters(self) -> List:
        r"""
        Returns the list of Euclidean parameters
        """
        return [par for gate in self._gates for par in gate.euclidean_parameters]


class Optimizer:
    _backend_opt: OptimizerBackendInterface

    def __init__(self, symplectic_lr: float = 1.0, euclidean_lr: float = 0.003):
        self.loss_history: List[float] = [0]
        self._symplectic_lr = symplectic_lr
        self._euclidean_lr = euclidean_lr
        self._opt = self._backend_opt(euclidean_lr)  # from specific backend

    def minimize(
        self,
        cost_fn: Callable,
        by_optimizing: Sequence[Union[Circuit, Detector]],
        max_steps: int = 1000,
    ) -> Union[Sequence[Circuit], Circuit]:
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
