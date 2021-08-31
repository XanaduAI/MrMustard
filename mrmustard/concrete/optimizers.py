from mrmustard._typing import *
from mrmustard.plugins import train, graphics

__all__ = ["Optimizer"]

# NOTE: there is no abstract optimizer class for the time being


class Optimizer:
    r"""An optimizer for any parametrized object.
    It can optimize euclidean, orthogonal and symplectic parameters.

    NOTE: In the future it will also include a compiler, so that it will be possible to
    simplify the circuit/detector/gate/etc before the optimization and also
    compile other types of structures like error correcting codes and encoders/decoders.
    """

    def __init__(self, symplectic_lr: float = 0.1, orthogonal_lr: float = 0.1, euclidean_lr: float = 0.001):
        self.symplectic_lr: float = symplectic_lr
        self.orthogonal_lr: float = orthogonal_lr
        self.euclidean_lr: float = euclidean_lr
        self.loss_history: List[float] = [0]

    def minimize(self, cost_fn: Callable, by_optimizing: Union[Trainable, Sequence[Trainable]], max_steps: int = 1000):
        r"""
        Minimizes the given cost function by optimizing circuits and/or detectors.
        Arguments:
            cost_fn (Callable): a function that will be executed in a differentiable context in order to compute gradients as needed
            by_optimizing (list of circuits and/or detectors and/or gates): a list of elements that contain the parameters to optimize
            max_steps (int): the minimization keeps going until the loss is stable or max_steps are reached (if `max_steps=0` it will only stop when the loss is stable)
        """
        params = {kind: train.extract_parameters(by_optimizing, kind) for kind in ("symplectic", "orthogonal", "euclidean")}
        bar = graphics.Progressbar(max_steps)
        with bar:
            while not self.should_stop(max_steps):
                loss, grads = train.loss_and_gradients(cost_fn, params)
                train.update_symplectic(params["symplectic"], grads["symplectic"], self.symplectic_lr)
                train.update_orthogonal(params["orthogonal"], grads["orthogonal"], self.orthogonal_lr)
                train.update_euclidean(params["euclidean"], grads["euclidean"], self.euclidean_lr)
                self.loss_history.append(loss)
                bar.step(train.numeric(loss))  # TODO

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
