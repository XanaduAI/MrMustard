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

import equinox as eqx
import jax
from optax import GradientTransformation, OptState, multi_transform

from mrmustard import math, settings
from mrmustard.lab import Circuit, CircuitComponent
from mrmustard.math.parameters import Variable
from mrmustard.training.parameter_update_jax import (
    update_orthogonal,
    update_symplectic,
    update_unitary,
)
from mrmustard.training.progress_bar import ProgressBar
from mrmustard.utils.logger import create_logger

__all__ = ["OptimizerJax"]


class OptimizerJax:
    r"""
    A Jax based optimizer for any parametrized object.

    Args:
        euclidean_lr: The euclidean learning rate of the optimizer.
        symplectic_lr: The symplectic learning rate of the optimizer.
        unitary_lr: The unitary learning rate of the optimizer.
        orthogonal_lr: The orthogonal learning rate of the optimizer.
        stable_threshold: The threshold for the loss to be considered stable.

    Raises:
        ValueError: If the set backend is not "jax".
    """

    def __init__(
        self,
        euclidean_lr: float = 0.001,
        symplectic_lr: float = 0.001,
        unitary_lr: float = 0.001,
        orthogonal_lr: float = 0.1,
        stable_threshold: float = 1e-6,
    ):
        if math.backend_name != "jax":
            raise ValueError(
                "OptimizerJax only supports the Jax backend. Please set the backend to Jax using `math.change_backend('jax')`.",
            )
        self.euclidean_lr = euclidean_lr
        self.symplectic_lr = symplectic_lr
        self.unitary_lr = unitary_lr
        self.orthogonal_lr = orthogonal_lr
        self.opt_history = [0]
        self.log = create_logger(__name__)
        self.stable_threshold = stable_threshold

    @eqx.filter_jit
    def make_step(
        self,
        optim: GradientTransformation,
        cost_fn: Callable,
        by_optimizing: Sequence[Variable | CircuitComponent | Circuit],
        opt_state: OptState,
    ) -> tuple[Sequence[Variable | CircuitComponent | Circuit], OptState, float]:
        r"""
        Make a step of the optimization.

        Args:
            optim: The optimizer to use.
            cost_fn: The cost function to minimize.
            by_optimizing: The items to optimize.
            opt_state: The current state of the optimizer.

        Returns:
            The updated by_optimizing, the updated optimizer state, and the loss value.
        """
        loss_value, grads = jax.value_and_grad(cost_fn, argnums=tuple(range(len(by_optimizing))))(
            *by_optimizing,
        )
        updates, opt_state = optim.update(grads, opt_state, by_optimizing)
        by_optimizing = eqx.apply_updates(by_optimizing, updates)
        return by_optimizing, opt_state, loss_value

    def minimize(
        self,
        cost_fn: Callable,
        by_optimizing: Sequence[Variable | CircuitComponent | Circuit],
        max_steps: int = 1000,
        euclidean_optim: type[GradientTransformation] | None = None,
    ) -> Sequence[Variable | CircuitComponent | Circuit]:
        r"""
        Minimizes the given cost function by optimizing ``Variable``s either on their own or within a ``CircuitComponent`` / ``Circuit``.

        Args:
            cost_fn: A function that will be executed in a differentiable context in
                order to compute gradients as needed.
            by_optimizing: A list of elements that contain the parameters to optimize.
            max_steps: The minimization keeps going until the loss is stable or max_steps are
                reached (if ``max_steps=0`` it will only stop when the loss is stable).
            euclidean_optim: The type of euclidean optimizer to use. If ``None``, the default optimizer used is ``optax.adamw``.

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
                    euclidean_optim=euclidean_optim,
                )
        else:
            by_optimizing = self._optimization_loop(
                cost_fn,
                by_optimizing,
                max_steps=max_steps,
                euclidean_optim=euclidean_optim,
            )
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
        euclidean_optim: type[GradientTransformation] | None = None,
    ) -> Sequence[Variable | CircuitComponent | Circuit]:
        r"""
        The core optimization loop.
        """
        by_optimizing = tuple(by_optimizing)
        euclidean_optim = (
            euclidean_optim(learning_rate=self.euclidean_lr)
            if euclidean_optim is not None
            else math.euclidean_opt(learning_rate=self.euclidean_lr)
        )

        labels_pytree = jax.tree_util.tree_map(
            lambda node: str(node.update_fn.__name__),
            by_optimizing,
            is_leaf=lambda n: isinstance(n, Variable),
        )

        optim = multi_transform(
            {
                "update_euclidean": euclidean_optim,
                "update_unitary": update_unitary(self.unitary_lr),
                "update_symplectic": update_symplectic(self.symplectic_lr),
                "update_orthogonal": update_orthogonal(self.orthogonal_lr),
            },
            labels_pytree,
        )

        opt_state = optim.init(by_optimizing)

        # optimize
        while not self.should_stop(max_steps):
            by_optimizing, opt_state, loss_value = self.make_step(
                optim,
                cost_fn,
                by_optimizing,
                opt_state,
            )
            self.opt_history.append(loss_value)
            if progress_bar is not None:
                progress_bar.step(math.asnumpy(loss_value))

        return by_optimizing
