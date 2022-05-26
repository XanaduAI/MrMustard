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

"""This module contains the implementation of optimization classes and functions
used within Mr Mustard.
"""

import itertools
from mrmustard.types import List, Callable, Sequence, Tensor
from mrmustard.utils.parameter import Parameter, Trainable
from mrmustard.utils import graphics
from mrmustard.logger import create_logger
from mrmustard.math import Math
from mrmustard import settings
from mrmustard.utils.parametrized import Parametrized

math = Math()

# pylint: disable=disallowed-name
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
        self.learning_rate = {
            "euclidian": euclidean_lr,
            "symplectic": symplectic_lr,
            "orthogonal": orthogonal_lr,
        }
        self.opt_history: List[float] = [0]
        self.log = create_logger(__name__)

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
            # finding out which parameters are trainable from the ops
            trainable_params = list(
                itertools.chain(
                    *[
                        item.trainable_parameters
                        for item in by_optimizing
                        if isinstance(item, Parametrized)
                    ]
                )
            )

            bar = graphics.Progressbar(max_steps)
            with bar:

                while not self.should_stop(max_steps):
                    cost, grads = loss_and_gradients(cost_fn, trainable_params)
                    for param, grad in zip(trainable_params, grads):
                        param_lr = self.learning_rate[param.type]
                        param.update(grad, param_lr)

                    self.opt_history.append(cost)
                    bar.step(math.asnumpy(cost))

        except KeyboardInterrupt:  # graceful exit
            self.log.info("Optimizer execution halted due to keyboard interruption.")
            raise self.OptimizerInterruptedError() from None

    def should_stop(self, max_steps: int) -> bool:
        r"""Returns ``True`` if the optimization should stop (either because the loss is stable or because the maximum number of steps is reached)."""
        if max_steps != 0 and len(self.opt_history) > max_steps:
            return True
        if len(self.opt_history) > 20:  # if cost varies less than 10e-6 over 20 steps
            if (
                sum(abs(self.opt_history[-i - 1] - self.opt_history[-i]) for i in range(1, 20))
                < 1e-6
            ):
                self.log.info("Loss looks stable, stopping here.")
                return True
        return False

    class OptimizerInterruptedError(Exception):
        """A helper class to quietly stop execution without printing a traceback."""

        def _render_traceback_(self):
            pass


# ~~~~~~~~~~~~~~~~~
# Static functions
# ~~~~~~~~~~~~~~~~~


def loss_and_gradients(cost_fn: Callable, parameters: List[Parameter]):
    r"""Uses the backend to compute the loss and gradients of the parameters
    given a cost function.

    This functions is a wrapper around the backend optimizer to extract tensors
    from `parameters` and correctly compute the loss and gradients. Results of
    the calculation are associated back again with the given parameters.

    Args:
        cost_fn (Callable with no args): The cost function.
        parameters (List[Parameter]): The parameters to optimize.

    Returns:
        tuple(Tensor, List[Tensor]): The loss and the gradients.
    """
    param_tensors = [p.value for p in parameters]
    loss, grads = math.value_and_gradients(cost_fn, param_tensors)

    return loss, grads


def new_symplectic(num_modes: int) -> Tensor:
    r"""Returns a new symplectic matrix from the current math backend with ``num_modes`` modes.

    Args:
        num_modes (int): the number of modes in the symplectic matrix

    Returns:
        Tensor: the new symplectic matrix
    """
    return math.random_symplectic(num_modes)


def new_orthogonal(num_modes: int) -> Tensor:
    """Returns a random orthogonal matrix in :math:`O(2*num_modes)`."""
    return math.random_orthogonal(num_modes)
