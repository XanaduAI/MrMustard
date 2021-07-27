from typing import List, Sequence, Callable, Union
from mrmustard.backends import Trainable
from mrmustard.plugins import TrainPlugin
from mrmustard.plugins import GraphicsPlugin


class Optimizer:
    r"""An optimizer for any parametrized object.
    It can optimize euclidean, orthogonal and symplectic parameters.

    NOTE: In the future it will also include a compiler, so that it will be possible to
    simplify the circuit/detector/gate/etc before the optimization.
    """

    _train: TrainPlugin
    _graphics: GraphicsPlugin

    def __init__(self, symplectic_lr: float = 1.0, euclidean_lr: float = 0.003):
        self.symplectic_lr: float = 0.1
        self.orthogonal_lr: float = 0.1
        self.euclidean_lr: float = 0.003

        self.loss_history: List[float] = [0]
        self._by_optimizing: Sequence[Trainable] = []

    def minimize(self, cost_fn: Callable, by_optimizing: Union[Trainable, Sequence[Trainable]], max_steps: int = 1000):
        r"""
        Minimizes the given cost function by optimizing circuits and/or detectors.
        Arguments:
            cost_fn (Callable): a function that will be executed in a differentiable context in order to compute gradients as needed
            by_optimizing (list of circuits and/or detectors and/or gates): a list of elements that contain the parameters to optimize
            max_steps (int): the minimization keeps going until the loss is stable or max_steps are reached (if `max_steps=0` it will only stop when the loss is stable)
        """
        symplectic_params: List[Trainable] = self._train.get_symp_params(by_optimizing)
        orthogonal_params: List[Trainable] = self._train.get_orth_params(by_optimizing)
        euclidean_params:  List[Trainable] = self._train.get_eucl_params(by_optimizing)

        bar = self._graphics.Progressbar(max_steps)
        with bar:
            while not self.should_stop(max_steps):
                loss, symp_grads, orth_grads, eucl_grads = self._train.loss_and_gradients(cost_fn)
                self._train.update_symp(symplectic_params, symp_grads, self.symplectic_lr)
                self._train.update_orth(orthogonal_params, orth_grads, self.orthogonal_lr)
                self._train.update_eucl(euclidean_params, eucl_grads, self.euclidean_lr)
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
