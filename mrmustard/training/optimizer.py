# Copyright 2022 Xanadu Quantum Technologies Inc.

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

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import chain, groupby

from mrmustard import math, settings
from mrmustard.lab import Circuit
from mrmustard.math.parameters import (
    Constant,
    Variable,
    update_euclidean,
    update_orthogonal,
    update_symplectic,
    update_unitary,
)
from mrmustard.training.callbacks import Callback
from mrmustard.training.progress_bar import ProgressBar
from mrmustard.utils.logger import create_logger

__all__ = ["Optimizer"]


class Optimizer:
    r"""An optimizer for any parametrized object: it can optimize euclidean, orthogonal and symplectic parameters.

    .. note::

        In the future it will also include a compiler, so that it will be possible to
        simplify the circuit/detector/gate/etc before the optimization and also
        compile other types of structures like error correcting codes and encoders/decoders.
    """

    def __init__(
        self,
        symplectic_lr: float = 0.1,
        unitary_lr: float = 0.1,
        orthogonal_lr: float = 0.1,
        euclidean_lr: float = 0.001,
    ):
        self.learning_rate = {
            update_euclidean: euclidean_lr,
            update_symplectic: symplectic_lr,
            update_unitary: unitary_lr,
            update_orthogonal: orthogonal_lr,
        }
        self.opt_history: list[float] = [0]
        self.callback_history: dict[str, list] = {}
        self.log = create_logger(__name__)

    def minimize(
        self,
        cost_fn: Callable,
        by_optimizing: Sequence[Constant | Variable | Circuit],
        max_steps: int = 1000,
        callbacks: Callable | Sequence[Callable] | Mapping[str, Callable] | None = None,
    ):
        r"""Minimizes the given cost function by optimizing circuits and/or detectors.

        Args:
            cost_fn (Callable): a function that will be executed in a differentiable context in
                order to compute gradients as needed
            by_optimizing (list of circuits and/or detectors and/or gates): a list of elements that
                contain the parameters to optimize
            max_steps (int): the minimization keeps going until the loss is stable or max_steps are
                reached (if ``max_steps=0`` it will only stop when the loss is stable)
            callbacks (:class:`Callback`, `Callable`, or List/Dict of them): callback functions that
                will be executed at each step of the optimization after backprop but before gradient
                gets applied. It takes as arguments the optimizer itself, training step (int), the
                cost value, the cost function, and the trainable parameters (values & grads) dict.
                The optional returned dict for each step is stored in self.callback_history which
                is a callback-name-keyed dict with each value a list of such callback result dicts.
                Learn more about how to use callbacks to have finer control of the optimization
                process in the :mod:`.callbacks` module.
        """
        math._euclidean_opt = None  # TODO: fix this temporary workaround
        callbacks = self._coerce_callbacks(callbacks)

        try:
            self._minimize(cost_fn, by_optimizing, max_steps, callbacks)
        except KeyboardInterrupt:  # graceful exit
            self.log.info("Optimizer execution halted due to keyboard interruption.")
            raise self.OptimizerInterruptedError from None

    def _minimize(self, cost_fn, by_optimizing, max_steps, callbacks):
        # finding out which parameters are trainable from the ops
        trainable_params = self._get_trainable_params(by_optimizing)
        if settings.PROGRESSBAR:
            bar = ProgressBar(max_steps)
            with bar:
                self._optimization_loop(cost_fn, trainable_params, max_steps, callbacks, bar)
        else:
            self._optimization_loop(cost_fn, trainable_params, max_steps, callbacks)

    def _optimization_loop(
        self,
        cost_fn,
        trainable_params,
        max_steps,
        callbacks,
        progress_bar=None,
    ):
        """Internal method that performs the main optimization loop.

        Args:
            cost_fn (Callable): The cost function to minimize
            trainable_params (dict): Dictionary of trainable parameters
            max_steps (int): Maximum number of optimization steps
            callbacks (dict): Dictionary of callback functions to execute during optimization
            progress_bar (ProgressBar, optional): Progress bar instance for displaying optimization progress.
                If None, no progress will be displayed. Defaults to None.

        Note:
            This method maintains internal state in self.opt_history and self.callback_history,
            tracking the optimization progress and callback results respectively.
        """
        cost_fn_modified = False
        orig_cost_fn = cost_fn

        while not self.should_stop(max_steps):
            cost, grads = self.compute_loss_and_gradients(cost_fn, trainable_params.values())

            trainables = {tag: (x, dx) for (tag, x), dx in zip(trainable_params.items(), grads)}

            if cost_fn_modified:
                self.callback_history["orig_cost"].append(orig_cost_fn())

            new_cost_fn, new_grads = self._run_callbacks(
                callbacks=callbacks,
                cost_fn=cost_fn,
                cost=cost,
                trainables=trainables,
            )

            self.apply_gradients(trainable_params.values(), new_grads or grads)
            self.opt_history.append(cost)
            if progress_bar is not None:
                progress_bar.step(math.asnumpy(cost))

            if callable(new_cost_fn):
                cost_fn = new_cost_fn
                if not cost_fn_modified:
                    cost_fn_modified = True
                    self.callback_history["orig_cost"] = self.opt_history.copy()

    def apply_gradients(self, trainable_params, grads):
        """Apply gradients to variables.

        This method group parameters by variable type (euclidean, symplectic, orthogonal) and
        applies the corresponding update method for each variable type. Update methods are
        registered on :mod:`parameter_update` module.
        """
        grouped_items = sorted(
            zip(grads, trainable_params),
            key=lambda x: hash(getattr(x[1], "update_fn", update_euclidean)),
        )
        grouped_items = {
            key: list(result)
            for key, result in groupby(
                grouped_items,
                key=lambda x: hash(getattr(x[1], "update_fn", update_euclidean)),
            )
        }

        for grads_vars in grouped_items.values():
            update_fn = getattr(grads_vars[0][1], "update_fn", update_euclidean)
            params_lr = self.learning_rate[update_fn]
            # extract value (tensor) from the parameter object and group with grad
            grads_and_vars = [(grad, p.value) for grad, p in grads_vars]
            update_fn(grads_and_vars, params_lr)

    @staticmethod
    def _get_trainable_params(trainable_items, root_tag: str = "optimized"):
        """Traverses all instances of gates, states, detectors, or trainable items that belong to the backend
        and return a dict of trainables of the form `{tags: trainable_parameters}` where the `tags`
        are traversal paths of collecting all parent tags for reaching each parameter.
        """
        trainables = []
        for i, item in enumerate(trainable_items):
            owner_tag = f"{root_tag}[{i}]"
            if isinstance(item, Circuit):
                for j, op in enumerate(item.components):
                    tag = f"{owner_tag}:{item.__class__.__qualname__}/_ops[{j}]"
                    tagged_vars = op.parameters.tagged_variables(tag)
                    trainables.append(tagged_vars.items())
            elif hasattr(item, "parameter_set"):
                tag = f"{owner_tag}:{item.__class__.__qualname__}"
                tagged_vars = item.parameter_set.tagged_variables(tag)
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

    @staticmethod
    def _group_vars_and_grads_by_type(trainable_params, grads):
        """Groups `trainable_params` and `grads` by type into a dict of the form
        `{"euclidean": [...], "orthogonal": [...], "symplectic": [...]}, "unitary": [...]`.
        """
        sorted_grads_and_vars = sorted(
            zip(grads, trainable_params),
            key=lambda grads_vars: grads_vars[1].type,
        )
        return {
            key: list(result)
            for key, result in groupby(
                sorted_grads_and_vars,
                key=lambda grads_vars: grads_vars[1].type,
            )
        }

    @staticmethod
    def compute_loss_and_gradients(cost_fn: Callable, parameters: list[Variable]):
        r"""Uses the backend to compute the loss and gradients of the parameters
        given a cost function.

        This functions is a wrapper around the backend optimizer to extract tensors
        from `parameters` and correctly compute the loss and gradients. Results of
        the calculation are associated back again with the given parameters.

        Args:
            cost_fn (Callable with no args): The cost function.
            parameters (List[Variable]): The variables to optimize.

        Returns:
            tuple(Tensor, List[Tensor]): The loss and the gradients.
        """
        param_tensors = [p.value for p in parameters]
        loss, grads = math.value_and_gradients(cost_fn, param_tensors)

        return loss, grads

    def should_stop(self, max_steps: int) -> bool:
        r"""Returns ``True`` if the optimization should stop (either because
        the loss is stable or because the maximum number of steps is reached)."""
        if max_steps != 0 and len(self.opt_history) > max_steps:
            return True
        # if cost varies less than 10e-6 over 20 steps
        if (
            len(self.opt_history) > 20
            and sum(abs(self.opt_history[-i - 1] - self.opt_history[-i]) for i in range(1, 20))
            < 1e-6
        ):
            self.log.info("Loss looks stable, stopping here.")
            return True
        return False

    @staticmethod
    def _coerce_callbacks(callbacks):
        r"""Coerce callbacks into dict and validate them."""
        if callbacks is None:
            callbacks = {}
        elif callable(callbacks):
            callbacks = {
                (
                    callbacks.tag if isinstance(callbacks, Callback) else callbacks.__name__
                ): callbacks,
            }
        elif isinstance(callbacks, Sequence):
            callbacks = {
                cb.tag if isinstance(cb, Callback) else cb.__name__: cb for cb in callbacks
            }
        elif not isinstance(callbacks, Mapping):
            raise TypeError(
                f"Argument `callbacks` expected to be a callable or a list/dict of callables, got {type(callbacks)}.",
            )

        if any(not callable(cb) for cb in callbacks.values()):
            raise TypeError("Not all provided callbacks is callable.")

        return callbacks

    def _run_callbacks(self, callbacks, cost_fn, cost, trainables):
        """Iteratively calls all callbacks and applies the necessary updates."""
        new_cost_fn, new_grads = None, None

        for cb_tag, cb in callbacks.items():
            if cb_tag not in self.callback_history:
                self.callback_history[cb_tag] = []

            cb_result = cb(
                optimizer=self,
                cost_fn=cost_fn if new_cost_fn is None else new_cost_fn,
                cost=cost,
                trainables=trainables,
            )

            if not isinstance(cb_result, Mapping | type(None)):
                raise TypeError(
                    f"The expected return type of callback functions is dict, got {type(cb_result)}.",
                )

            new_cost_fn = cb_result.pop("cost_fn", None)

            if "grads" in cb_result:
                new_grads = cb_result["grads"]
                trainables = {
                    tag: (x, dx) for (tag, (x, _)), dx in zip(trainables.items(), new_grads)
                }

            if cb_result is not None and cb_result:
                self.callback_history[cb_tag].append(cb_result)

        return new_cost_fn, new_grads

    class OptimizerInterruptedError(Exception):
        """A helper class to quietly stop execution without printing a traceback."""

        def _render_traceback_(self):
            pass
