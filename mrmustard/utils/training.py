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

from itertools import chain, groupby
from nntplib import GroupInfo
from mrmustard.types import List, Callable, Sequence, Tensor, Tuple
from mrmustard.utils.parameter import Parameter, Trainable
from mrmustard.utils import graphics
from mrmustard.logger import create_logger
from mrmustard.math import Math
from mrmustard.utils.parametrized import Parametrized

math = Math()


def update_symplectic(grads_and_vars: Sequence[Tuple[Tensor, Trainable]], symplectic_lr: float):

    r"""Updates the symplectic parameters using the given symplectic gradients.
    Implemented from:
        Wang J, Sun H, Fiori S. A Riemannian-steepest-descent approach
        for optimization on the real symplectic group.
        Mathematical Methods in the Applied Sciences. 2018 Jul 30;41(11):4273-86.
    """
    for dS_euclidean, S in grads_and_vars:
        Y = math.euclidean_to_symplectic(S, dS_euclidean)
        YT = math.transpose(Y)
        new_value = math.matmul(
            S, math.expm(-symplectic_lr * YT) @ math.expm(-symplectic_lr * (Y - YT))
        )
        math.assign(S, new_value)


def update_orthogonal(grads_and_vars: Sequence[Tuple[Tensor, Trainable]], orthogonal_lr: float):
    r"""Updates the orthogonal parameters using the given orthogonal gradients.
    Implemented from:
        Fiori S, Bengio Y. Quasi-Geodesic Neural Learning Algorithms
        Over the Orthogonal Group: A Tutorial.
        Journal of Machine Learning Research. 2005 May 1;6(5).
    """
    for dO_euclidean, O in grads_and_vars:
        dO_orthogonal = 0.5 * (
            dO_euclidean - math.matmul(math.matmul(O, math.transpose(dO_euclidean)), O)
        )
        new_value = math.matmul(
            O, math.expm(orthogonal_lr * math.matmul(math.transpose(dO_orthogonal), O))
        )
        math.assign(O, new_value)


def update_euclidean(grads_and_vars: Sequence[Tuple[Tensor, Trainable]], euclidean_lr: float):
    """Updates the parameters using the euclidian gradients."""
    math.euclidean_opt.lr = euclidean_lr
    math.euclidean_opt.apply_gradients(grads_and_vars)


update_method_dict = {
    "euclidean": update_euclidean,
    "symplectic": update_symplectic,
    "orthogonal": update_orthogonal,
}

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
            "euclidean": euclidean_lr,
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
                chain(
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
                    self.update_params(trainable_params, grads)

                    self.opt_history.append(cost)
                    bar.step(math.asnumpy(cost))

        except KeyboardInterrupt:  # graceful exit
            self.log.info("Optimizer execution halted due to keyboard interruption.")
            raise self.OptimizerInterruptedError() from None

    def update_params(self, trainable_params, grads):

        # group grads and vars by type
        grouped_vars_and_grads = self._group_vars_and_grads_by_type(trainable_params, grads)

        for param_type, grads_vars in grouped_vars_and_grads.items():
            param_lr = self.learning_rate[param_type]
            grads_and_vars = [(grad, p.value) for grad, p in grads_vars]
            update_method = update_method_dict.get(param_type)
            update_method(grads_and_vars, param_lr)

    @staticmethod
    def _group_vars_and_grads_by_type(trainable_params, grads):
        sorted_grads_and_vars = sorted(
            zip(grads, trainable_params), key=lambda grads_vars: grads_vars[1].type
        )
        grouped = {
            key: list(result)
            for key, result in groupby(
                sorted_grads_and_vars, key=lambda grads_vars: grads_vars[1].type
            )
        }

        return grouped

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
