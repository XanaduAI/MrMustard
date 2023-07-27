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

from itertools import chain, groupby
from typing import List, Callable, Sequence, Union, Mapping, Dict
from mrmustard.training.callbacks import Callback
from mrmustard.utils import graphics
from mrmustard.logger import create_logger
from mrmustard.math import Math
from .parameter import Parameter, Trainable, create_parameter
from .parametrized import Parametrized
from .parameter_update import param_update_method

math = Math()

__all__ = ["Optimizer"]


# pylint: disable=disallowed-name
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
            "euclidean": euclidean_lr,
            "symplectic": symplectic_lr,
            "unitary": unitary_lr,
            "orthogonal": orthogonal_lr,
        }
        self.opt_history: List[float] = [0]
        self.callback_history: Dict[str, List] = {}
        self.log = create_logger(__name__)

    def minimize(
        self,
        cost_fn: Callable,
        by_optimizing: Sequence[Trainable],
        max_steps: int = 1000,
        callbacks: Union[Callable, Sequence[Callable], Mapping[str, Callable]] = None,
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
        callbacks = self._coerce_callbacks(callbacks)

        try:
            self._minimize(cost_fn, by_optimizing, max_steps, callbacks)
        except KeyboardInterrupt:  # graceful exit
            self.log.info("Optimizer execution halted due to keyboard interruption.")
            raise self.OptimizerInterruptedError() from None

    def _minimize(self, cost_fn, by_optimizing, max_steps, callbacks):
        # finding out which parameters are trainable from the ops
        trainable_params = self._get_trainable_params(by_optimizing)
        cost_fn_modified = False
        orig_cost_fn = cost_fn

        bar = graphics.Progressbar(max_steps)
        with bar:
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
                bar.step(math.asnumpy(cost))

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

        # group grads and vars by type (i.e. euclidean, symplectic, orthogonal, unitary)
        grouped_vars_and_grads = self._group_vars_and_grads_by_type(trainable_params, grads)

        for param_type, grads_vars in grouped_vars_and_grads.items():
            param_lr = self.learning_rate[param_type]
            # extract value (tensor) from the parameter object and group with grad
            grads_and_vars = [(grad, p.value) for grad, p in grads_vars]
            update_method = param_update_method.get(param_type)
            update_method(grads_and_vars, param_lr)

    @staticmethod
    def _get_trainable_params(trainable_items, root_tag: str = "optimized"):
        """Traverses all instances of Parametrized or trainable items that belong to the backend
        and return a dict of trainables of the form `{tags: trainable_parameters}` where the `tags`
        are traversal paths of collecting all parent tags for reaching each parameter.
        """
        trainables = []
        for i, item in enumerate(trainable_items):
            owner_tag = f"{root_tag}[{i}]"
            if isinstance(item, Parametrized):
                tag = f"{owner_tag}:{item.__class__.__qualname__}"
                trainables.append(item.traverse_trainables(owner_tag=tag).items())
            elif math.from_backend(item) and math.is_trainable(item):
                # the created parameter is wrapped into a list because the case above
                # returns a list, hence ensuring we have a list of lists
                tag = f"{owner_tag}:{math.__class__.__name__}/{getattr(item, 'name', item.__class__.__name__)}"
                trainables.append(
                    [
                        (
                            tag,
                            create_parameter(
                                item, name="from_backend", is_trainable=True, owner=tag
                            ),
                        )
                    ]
                )

        return dict(chain(*trainables))

    @staticmethod
    def _group_vars_and_grads_by_type(trainable_params, grads):
        """Groups `trainable_params` and `grads` by type into a dict of the form
        `{"euclidean": [...], "orthogonal": [...], "symplectic": [...]}, "unitary": [...]`."""
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

    @staticmethod
    def compute_loss_and_gradients(cost_fn: Callable, parameters: List[Parameter]):
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

    def should_stop(self, max_steps: int) -> bool:
        r"""Returns ``True`` if the optimization should stop (either because
        the loss is stable or because the maximum number of steps is reached)."""
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

    @staticmethod
    def _coerce_callbacks(callbacks):
        r"""Coerce callbacks into dict and validate them."""
        if callbacks is None:
            callbacks = {}
        elif callable(callbacks):
            callbacks = {
                callbacks.tag if isinstance(callbacks, Callback) else callbacks.__name__: callbacks
            }
        elif isinstance(callbacks, Sequence):
            callbacks = {
                cb.tag if isinstance(cb, Callback) else cb.__name__: cb for cb in callbacks
            }
        elif not isinstance(callbacks, Mapping):
            raise TypeError(
                f"Argument `callbacks` expected to be a callable or a list/dict of callables, got {type(callbacks)}."
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

            if not isinstance(cb_result, (Mapping, type(None))):
                raise TypeError(
                    f"The expected return type of callback functions is dict, got {type(cb_result)}."
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
