# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
A Jax based optimizer for any parametrized object.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import chain

import equinox as eqx
import jax
from optax import GradientTransformation, OptState

from mrmustard import math, settings
from mrmustard.lab import Circuit, CircuitComponent
from mrmustard.math.parameters import Variable
from mrmustard.training.progress_bar import ProgressBar
from mrmustard.utils.logger import create_logger

__all__ = ["OptimizerJax"]


def get_trainable_params(
    trainable_items: Sequence[Variable | CircuitComponent | Circuit],
    root_tag: str = "optimized",
) -> dict[str, Variable]:
    r"""
    Traverses all instances of ``CircuitComponent``\s or trainable items that belong to the backend
    and return a dict of trainables of the form `{tags: trainable_parameters}` where the `tags`
    are traversal paths of collecting all parent tags for reaching each parameter.

    Args:
        trainable_items: A list of trainable items.
        root_tag: The root tag for the trainable items.

    Returns:
        A dict of trainables of the form `{tags: trainable_parameters}`.
    """
    trainables = []
    for i, item in enumerate(trainable_items):
        owner_tag = f"{root_tag}[{i}]"
        if isinstance(item, Circuit):
            for j, op in enumerate(item.components):
                tag = f"{owner_tag}:{item.__class__.__qualname__}/_ops[{j}]"
                tagged_vars = op.parameters.tagged_variables(tag)
                trainables.append(tagged_vars.items())
        elif hasattr(item, "parameters"):
            tag = f"{owner_tag}:{item.__class__.__qualname__}"
            tagged_vars = item.parameters.tagged_variables(tag)
            trainables.append(tagged_vars.items())
        elif math.from_backend(item) and math.is_trainable(item):
            # the created parameter is wrapped into a list because the case above
            # returns a list, hence ensuring we have a list of lists
            tag = f"{owner_tag}:{math.__class__.__name__}/{getattr(item, 'name', item.__class__.__name__)}"
            trainables.append([(tag, Variable(item, name="from _backend"))])

    return dict(chain(*trainables))


class Objective(eqx.Module):
    r"""
    A dataclass used by equinox to store the Jax arrays of the trainable parameters.

    Args:
        static: The static parameters of the model. These are the keys associated with the trainable parameters.
        dynamic: The dynamic parameters of the model. These are the arrays of the trainable parameters.
    """

    static: list[str]
    dynamic: list[jax.Array]

    def __init__(self, trainable_params: dict[str, Variable]):
        self.static = list(trainable_params.keys())
        self.dynamic = [array.value for array in trainable_params.values()]

    def __call__(self, cost_fn: Callable, by_optimizing: Sequence[CircuitComponent]) -> float:
        r"""
        Updates the parameters in ``by_optimizing`` with the values in ``self.dynamic``
        and calls the cost function. This is necessary because Jax does not support
        in-place updates.

        Args:
            cost_fn: The cost function to minimize.
            by_optimizing: The parameters to optimize.

        Returns:
            The loss value.
        """
        trainable_params = get_trainable_params(by_optimizing)
        for key, val in zip(self.static, self.dynamic):
            trainable_params[key].value = val
        return cost_fn(*by_optimizing)


class OptimizerJax:
    r"""
    A Jax based optimizer for any parametrized object.

    Note that this optimizer currently only supports Euclidean optimizations.

    Args:
        learning_rate: The learning rate of the optimizer.
        stable_threshold: The threshold for the loss to be considered stable.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        stable_threshold: float = 1e-6,
    ):
        self.learning_rate = learning_rate
        self.opt_history = [0]
        self.log = create_logger(__name__)
        self.stable_threshold = stable_threshold

    @eqx.filter_jit
    def make_step(
        self,
        optim: GradientTransformation,
        loss: Callable,
        model: eqx.Module,
        opt_state: OptState,
    ) -> tuple[eqx.Module, OptState, float]:
        r"""
        Make a step of the optimization.

        Args:
            optim: The optimizer to use.
            loss: The loss function to minimize.
            model: The model to optimize.
            opt_state: The current state of the optimizer.

        Returns:
            The updated model, the updated optimizer state, and the loss value.
        """
        params, static = eqx.partition(model, eqx.is_array)
        loss_value, grads = jax.value_and_grad(loss)(params, static)
        updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    def minimize(
        self,
        cost_fn: Callable,
        by_optimizing: Sequence[Variable | CircuitComponent | Circuit],
        max_steps: int = 1000,
    ) -> Sequence[Variable | CircuitComponent | Circuit]:
        r"""
        Minimizes the given cost function by optimizing ``Variable``s either on their own or within a ``CircuitComponent`` / ``Circuit``.

        Args:
            cost_fn: A function that will be executed in a differentiable context in
                order to compute gradients as needed.
            by_optimizing: A list of elements that contain the parameters to optimize.
            max_steps: The minimization keeps going until the loss is stable or max_steps are
                reached (if ``max_steps=0`` it will only stop when the loss is stable).

        Returns:
            The list of elements optimized.
        """
        if settings.PROGRESSBAR:
            progress_bar = ProgressBar(max_steps)
            with progress_bar:
                by_optimizing = self._optimization_loop(
                    cost_fn,
                    by_optimizing,
                    max_steps=max_steps,
                    progress_bar=progress_bar,
                )
        else:
            by_optimizing = self._optimization_loop(cost_fn, by_optimizing, max_steps=max_steps)
        return by_optimizing

    def should_stop(self, max_steps: int) -> bool:
        r"""
        Returns a boolean indicating whether the optimization should stop.
        An optimization should stop either because the loss is stable or because
        the maximum number of steps is reached.

        Args:
            max_steps: The maximum number of steps to run.

        Returns:
            A boolean indicating whether the optimization should stop.
        """
        if max_steps != 0 and len(self.opt_history) > max_steps:
            return True
        # if cost varies less than threshold over 20 steps
        if (
            len(self.opt_history) > 20
            and sum(abs(self.opt_history[-i - 1] - self.opt_history[-i]) for i in range(1, 20))
            < self.stable_threshold
        ):
            self.log.info("Loss looks stable, stopping here.")
            return True
        return False

    def _optimization_loop(
        self,
        cost_fn: Callable,
        by_optimizing: Sequence[Variable | CircuitComponent | Circuit],
        max_steps: int,
        progress_bar: ProgressBar | None = None,
    ) -> Sequence[Variable | CircuitComponent | Circuit]:
        r"""
        The core optimization loop.
        """
        trainable_params = get_trainable_params(by_optimizing)
        model = Objective(trainable_params)

        def loss(params, static):
            model = eqx.combine(params, static)
            return model(cost_fn, by_optimizing)

        optim = math.euclidean_opt(learning_rate=self.learning_rate)
        opt_state = optim.init(eqx.filter(model, eqx.is_array))

        # optimize
        while not self.should_stop(max_steps):
            model, opt_state, loss_value = self.make_step(optim, loss, model, opt_state)
            self.opt_history.append(loss_value)
            if progress_bar is not None:
                progress_bar.step(math.asnumpy(loss_value))

        # update the parameters one last time
        for key, val in zip(model.static, model.dynamic):
            trainable_params[key].value = val

        return by_optimizing
