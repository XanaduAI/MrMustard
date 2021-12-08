# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from scipy.linalg import expm
from mrmustard.types import *
from mrmustard.utils import graphics
from mrmustard.math import Math

math = Math()


class Optimizer:
    r"""An optimizer for any parametrized object: it can optimize euclidean, orthogonal and symplectic parameters.

    .. note::

        In the future it will also include a compiler, so that it will be possible to
        simplify the circuit/detector/gate/etc before the optimization and also
        compile other types of structures like error correcting codes and encoders/decoders.
    """

    def __init__(
        self, symplectic_lr: float = 0.1, orthogonal_lr: float = 0.1, euclidean_lr: float = 0.001
    ):
        self.symplectic_lr: float = symplectic_lr
        self.orthogonal_lr: float = orthogonal_lr
        self.euclidean_lr: float = euclidean_lr
        self.opt_history: List[float] = [0]

    def minimize(
        self, cost_fn: Callable, by_optimizing: Sequence[Trainable], max_steps: int = 1000
    ):
        r"""Minimizes the given cost function by optimizing circuits and/or detectors.

        Args:
            cost_fn (Callable): a function that will be executed in a differentiable context in
                order to compute gradients as needed
            by_optimizing (list of circuits and/or detectors and/or gates): a list of elements that
                contain the parameters to optimize
            max_steps (int): the minimization keeps going until the loss is stable or max_steps are
                reached (if ``max_steps=0`` it will only stop when the loss is stable)
        """
        try:
            params = {
                "symplectic": math.unique_tensors(
                    [p for item in by_optimizing for p in item.trainable_parameters["symplectic"]]
                ),
                "orthogonal": math.unique_tensors(
                    [p for item in by_optimizing for p in item.trainable_parameters["orthogonal"]]
                ),
                "euclidean": math.unique_tensors(
                    [p for item in by_optimizing for p in item.trainable_parameters["euclidean"]]
                ),
            }
            bar = graphics.Progressbar(max_steps)
            with bar:
                while not self.should_stop(max_steps):
                    cost, grads = math.value_and_gradients(cost_fn, params)
                    update_symplectic(params["symplectic"], grads["symplectic"], self.symplectic_lr)
                    update_orthogonal(params["orthogonal"], grads["orthogonal"], self.orthogonal_lr)
                    update_euclidean(params["euclidean"], grads["euclidean"], self.euclidean_lr)
                    self.opt_history.append(cost)
                    bar.step(math.asnumpy(cost))
        except KeyboardInterrupt:  # graceful exit
            return

    def should_stop(self, max_steps: int) -> bool:
        r"""Returns ``True`` if the optimization should stop (either because the loss is stable or because the maximum number of steps is reached)."""
        if max_steps != 0 and len(self.opt_history) > max_steps:
            return True
        if len(self.opt_history) > 20:  # if cost varies less than 10e-6 over 20 steps
            if (
                sum(abs(self.opt_history[-i - 1] - self.opt_history[-i]) for i in range(1, 20))
                < 1e-6
            ):
                print("Loss looks stable, stopping here.")
                return True
        return False


# ~~~~~~~~~~~~~~~~~
# Static functions
# ~~~~~~~~~~~~~~~~~


# def new_variable(
#     value, bounds: Tuple[Optional[float], Optional[float]], name: str, dtype=math.float64
# ) -> Trainable:
#     r"""Returns a new trainable variable from the current math backend
#     with initial value set by `value` and bounds set by `bounds`.
#
#     Args:
#         value (float): The initial value of the variable
#         bounds (Tuple[float, float]): The bounds of the variable
#         name (str): The name of the variable
#         dtype: The dtype of the variable
#
#     Returns:
#         variable (Trainable): The new variable
#     """
#     return math.new_variable(value, bounds, name, dtype)


# def new_constant(value, name: str, dtype=math.float64) -> Tensor:
#     r"""Returns a new constant (non-trainable) tensor from the current math backend
#     with initial value set by `value`.
#     Args:
#         value (numeric): The initial value of the tensor
#         name (str): The name of the constant
#         dtype: The dtype of the constant
#
#     Returns:
#         tensor (Tensor): The new constant tensor
#     """
#     return math.new_constant(value, name, dtype)


def new_symplectic(num_modes: int) -> Tensor:
    r"""Returns a new symplectic matrix from the current math backend with ``num_modes`` modes.

    Args:
        num_modes (int): the number of modes in the symplectic matrix

    Returns:
        Tensor: the new symplectic matrix
    """
    return math.random_symplectic(num_modes)


def new_orthogonal(num_modes: int) -> Tensor:
    return math.random_orthogonal(num_modes)


def update_symplectic(
    symplectic_params: Sequence[Trainable], symplectic_grads: Sequence[Tensor], symplectic_lr: float
):

    r"""Updates the symplectic parameters using the given symplectic gradients.

    Implemented from:
        Wang J, Sun H, Fiori S. A Riemannian‐steepest‐descent approach
        for optimization on the real symplectic group.
        Mathematical Methods in the Applied Sciences. 2018 Jul 30;41(11):4273-86.
    """
    for S, dS_euclidean in zip(symplectic_params, symplectic_grads):
        Y = math.euclidean_to_symplectic(S, dS_euclidean)
        YT = math.transpose(Y)
        new_value = math.matmul(
            S, math.expm(-symplectic_lr * YT) @ math.expm(-symplectic_lr * (Y - YT))
        )
        math.assign(S, new_value)


def update_orthogonal(
    orthogonal_params: Sequence[Trainable], orthogonal_grads: Sequence[Tensor], orthogonal_lr: float
):
    r"""Updates the orthogonal parameters using the given orthogonal gradients.

    Implemented from:
        Fiori S, Bengio Y. Quasi-Geodesic Neural Learning Algorithms
        Over the Orthogonal Group: A Tutorial.
        Journal of Machine Learning Research. 2005 May 1;6(5).
    """
    for O, dO_euclidean in zip(orthogonal_params, orthogonal_grads):
        dO_orthogonal = 0.5 * (
            dO_euclidean - math.matmul(math.matmul(O, math.transpose(dO_euclidean)), O)
        )
        new_value = math.matmul(
            O, math.expm(orthogonal_lr * math.matmul(math.transpose(dO_orthogonal), O))
        )
        math.assign(O, new_value)


def update_euclidean(
    euclidean_params: Sequence[Trainable], euclidean_grads: Sequence[Tensor], euclidean_lr: float
):
    math.euclidean_opt.lr = euclidean_lr
    math.euclidean_opt.apply_gradients(zip(euclidean_grads, euclidean_params))
