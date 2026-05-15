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

from mrmustard import math, settings
from mrmustard.lab import CircuitComponent
from mrmustard.parameters import ParameterDict, Variable
from mrmustard.training.progress_bar import ProgressBar
from mrmustard.utils.logger import create_logger

try:
    import equinox as eqx
    import jax
    import optax
    from optax import GradientTransformation, OptState, multi_transform

    from mrmustard.training.parameter_update import (
        update_orthogonal,
        update_symplectic,
        update_unitary,
    )

except ImportError:
    raise ImportError(
        "Optimizer only supports the Jax backend. Please install the `jax_backend` group using `uv pip install -g jax_backend` and set the backend to Jax using `math.change_backend('jax')`."
    ) from None

__all__ = ["Optimizer"]


class Optimizer:
    r"""
    A Jax based optimizer for any parametrized object.

    Args:
        euclidean_lr: The learning rate for euclidean parameters.
        symplectic_lr: The learning rate for symplectic parameters.
        unitary_lr: The learning rate for unitary parameters.
        orthogonal_lr: The learning rate for orthogonal parameters.
        euclidean_optimizer: The optax optimizer class for euclidean updates (default: ``optax.adabelief``).
        symplectic_optimizer: The optax optimizer class for symplectic updates (default: ``optax.adabelief``).
        unitary_optimizer: The optax optimizer class for unitary updates (default: ``optax.adabelief``).
        orthogonal_optimizer: The optax optimizer class for orthogonal updates (default: ``optax.adabelief``).
        stable_threshold: The threshold for the loss to be considered stable.

    Raises:
        ValueError: If the set backend is not "jax".

    Example:
        Using different optimizers for different parameter types:

        >>> from mrmustard.training import Optimizer
        >>> import optax
        >>> opt = Optimizer(
        ...     euclidean_lr=0.01,
        ...     symplectic_lr=0.001,
        ...     euclidean_optimizer=optax.adamw,
        ...     symplectic_optimizer=optax.adam,
        ... )
    """

    def __init__(
        self,
        euclidean_lr: float = 0.001,
        symplectic_lr: float = 0.001,
        unitary_lr: float = 0.001,
        orthogonal_lr: float = 0.1,
        euclidean_optimizer: type[GradientTransformation] | None = None,
        symplectic_optimizer: type[GradientTransformation] | None = None,
        unitary_optimizer: type[GradientTransformation] | None = None,
        orthogonal_optimizer: type[GradientTransformation] | None = None,
        stable_threshold: float = 1e-6,
    ):
        if math.backend_name != "jax":
            raise ValueError(
                "Optimizer only supports the Jax backend. Please set the backend to Jax using `math.change_backend('jax')`.",
            )
        self.euclidean_lr = euclidean_lr
        self.symplectic_lr = symplectic_lr
        self.unitary_lr = unitary_lr
        self.orthogonal_lr = orthogonal_lr
        self.euclidean_optimizer = euclidean_optimizer or optax.adabelief
        self.symplectic_optimizer = symplectic_optimizer or optax.adabelief
        self.unitary_optimizer = unitary_optimizer or optax.adabelief
        self.orthogonal_optimizer = orthogonal_optimizer or optax.adabelief
        self.opt_history = [0]
        self.log = create_logger(__name__)
        self.stable_threshold = stable_threshold

    @eqx.filter_jit
    def make_step(
        self,
        optim: GradientTransformation,
        cost_fn: Callable,
        by_optimizing: Sequence[Variable | CircuitComponent],
        opt_state: OptState,
    ) -> tuple[Sequence[Variable | CircuitComponent], OptState, float]:
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
        conj_grads = jax.tree.map(lambda x: jax.numpy.conj(x), grads)
        updates, opt_state = optim.update(conj_grads, opt_state, by_optimizing)
        by_optimizing = eqx.apply_updates(by_optimizing, updates)
        return by_optimizing, opt_state, loss_value

    def minimize(
        self,
        cost_fn: Callable,
        by_optimizing: Sequence[Variable] | ParameterDict,
        max_steps: int = 1000,
    ) -> Sequence[Variable] | ParameterDict:
        r"""
        Minimizes the given cost function by optimizing ``Variable``\ s.
        If a ``ParameterDict`` is provided, it will return a ``ParameterDict`` with the optimized variables.

        Args:
            cost_fn: A function that will be executed in a differentiable context in
                order to compute gradients as needed.
            by_optimizing: A list of ``Variable``\ s to optimize.
            max_steps: The minimization keeps going until the loss is stable or max_steps are
                reached (if ``max_steps=0``\ , it will only stop when the loss is stable).

        Returns:
            The list of optimized ``Variable``\ s or the ``ParameterDict`` with the optimized variables.
        """
        if isinstance(by_optimizing, ParameterDict):
            return_param_dict = True
            by_optimizing = by_optimizing.variables.values()
        else:
            return_param_dict = False
            by_optimizing = tuple(by_optimizing)

        self.opt_history = [0]
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
            by_optimizing = self._optimization_loop(
                cost_fn,
                by_optimizing,
                max_steps=max_steps,
            )
        return ParameterDict(*by_optimizing) if return_param_dict else by_optimizing

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
        by_optimizing: Sequence[Variable],
        max_steps: int,
        progress_bar: ProgressBar | None = None,
    ) -> Sequence[Variable]:
        r"""
        The core optimization loop.
        """
        by_optimizing = tuple(by_optimizing)

        labels_pytree = jax.tree_util.tree_map(
            lambda node: str(node.update_fn),
            by_optimizing,
            is_leaf=lambda n: isinstance(n, Variable),
        )

        optim = multi_transform(
            {
                "update_euclidean": self.euclidean_optimizer(learning_rate=self.euclidean_lr),
                "update_unitary": update_unitary(self.unitary_lr, self.unitary_optimizer),
                "update_symplectic": update_symplectic(
                    self.symplectic_lr, self.symplectic_optimizer
                ),
                "update_orthogonal": update_orthogonal(
                    self.orthogonal_lr, self.orthogonal_optimizer
                ),
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
