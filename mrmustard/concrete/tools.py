from __future__ import annotations
from collections.abc import MutableSequence
from typing import List, Callable, Sequence, Union

from mrmustard.core.backends import MathBackendInterface, TrainingBackendInterface
from mrmustard.core.baseclasses import Op, State, Detector, Tensor
from mrmustard.core.visual import Progressbar

from typing import TypeVar

Trainable = TypeVar("Trainable", Circuit, Op)

class Circuit(MutableSequence):

    def __init__(self, ops: Sequence[Op] = []):
        try:
            if all(isinstance(o, Op) for o in ops):
                self._ops: List[Op] = [o for o in ops]
        except TypeError:
            raise TypeError(f"not a sequence of ops")
        except Exception as e:
            raise e

    def __call__(self, state: State) -> State:
        state_ = state  # NOTE: otherwise the next time we call the circuit, the state will be mutated
        for op in self._ops:
            state_ = op(state_)
        return state_

    def __getitem__(self, key):
        return self._ops.__getitem__(key)

    def __setitem__(self, key, value):
        if not isinstance(value, Gate):
            raise ValueError(f"Item {type(value)} is not a gate")
        return self._ops.__setitem__(key, value)

    def __delitem__(self, key):
        return self._ops.__delitem__(key)

    def __len__(self):
        return len(self._ops)

    def __repr__(self) -> str:
        return "\n".join([repr(g) for g in self._ops])

    def insert(self, index, object):
        return self._ops.insert(index, object)

    @property
    def symplectic_parameters(self) -> List:
        r"""
        Returns the list of symplectic parameters
        """
        return [par for op in self._ops for par in op.symplectic_parameters]

    @property
    def orthogonal_parameters(self) -> List:
        r"""
        Returns the list of orthogonal parameters
        """
        return [par for op in self._ops for par in op.orthogonal_parameters]

    @property
    def euclidean_parameters(self) -> List:
        r"""
        Returns the list of Euclidean parameters
        """
        return [par for op in self._ops for par in op.euclidean_parameters]


class Optimizer:
    _train: TrainPluginInterface

    def __init__(self, symplectic_lr: float = 1.0, euclidean_lr: float = 0.003):
        self.loss_history: List[float] = [0]
        self.symplectic_lr = symplectic_lr
        self.euclidean_lr = euclidean_lr
        self._by_optimizing: Sequence[Trainable] = []

    def minimize(self, cost_fn: Callable, by_optimizing: Sequence[Trainable], max_steps: int = 1000):
        r"""
        Optimizes circuits and/or detectors such that the given cost function is minimized.
        Arguments:
            cost_fn (Callable): a function that will be executed in a differentiable context in order to compute gradients as needed
            by_optimizing (list of circuits and/or detectors and/or gates): a list of elements that contain the parameters to optimize
            max_steps (int): the minimization keeps going until the loss is stable or max_steps are reached (if `max_steps=0` it will only stop when the loss is stable)
        """
        self._by_optimizing = by_optimizing
        if not isinstance(by_optimizing, Sequence):
            by_optimizing = [by_optimizing]
        self._train.store_symp_params(by_optimizing)
        self._train.store_orth_params(by_optimizing)
        self._train.store_eucl_params(by_optimizing)

        bar = Progressbar(max_steps)
        with bar:
            while not self.should_stop(max_steps):
                loss, symp_grads, orth_grads, eucl_grads = self._train.loss_and_gradients(cost_fn)
                self._train.update_symp(symp_grads, self.symplectic_lr)
                self._train.update_orth(orth_grads, self.symplectic_lr)
                self._train.update_eucl(eucl_grads, self.euclidean_lr)
                self.loss_history.append(loss)
                bar.step(loss)

    def should_stop(self, max_steps: int) -> bool:
        r"""
        Returns True if the optimization should stop
        (either because the loss is stable or because the maximum number of steps is reached)
        """
        if max_steps != 0 and len(self.loss_history) > max_steps:
            return True
        if len(self.loss_history) > 20:  # if loss varies less than 10e-6 over 20 steps
            if sum(abs(self.loss_history[-i - 1] - self.loss_history[-i]) for i in range(1, 20)) < 1e-6:
                print("Loss looks stable, stopping here.")
                return True
        return False
