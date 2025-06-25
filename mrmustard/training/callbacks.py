# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains the implementation of callback functionalities for optimizations.

Callbacks allow users to have finer control over the optimization process by executing
predefined routines as optimization progresses. Even though the :meth:`Optimizer.minimize` accepts
`Callable` functions directly, the :class:`Callback` class modularizes the logic and makes it
easier for users to inherit from it and come up with their own custom callbacks.

Things you can do with callbacks:

* Logging custom metrics.
* Tracking parameters and costs with Tensorboard.
* Scheduling learning rates.
* Modifying the gradient update that gets applied.
* Updating cost_fn to alter the optimization landscape in our favour.
* Adding some RL into the optimizer.
* ...

Builtin callbacks:

* :class:`Callback`: The base class, to be used for building custom callbacks.
* :class:`TensorboardCallback`: Tracks costs, parameter values and gradients in Tensorboard.

Examples:
=========

.. code-block::

    import numpy as np
    from mrmustard.training import Optimizer, TensorboardCallback

    def cost_fn():
        ...

    def as_dB(cost):
        delta = np.sqrt(np.log(1 / (abs(cost) ** 2)) / (2 * np.pi))
        cost_dB = -10 * np.log10(delta**2)
        return cost_dB

    tb_cb = TensorboardCallback(cost_converter=as_dB, track_grads=True)

    def rolling_cost_cb(optimizer, cost, **kwargs):
        return {
            'rolling_cost': np.mean(optimizer.opt_history[-10:] + [cost]),
        }

    opt = Optimizer(euclidean_lr = 0.001);
    opt.minimize(cost_fn, max_steps=200, by_optimizing=[...], callbacks=[tb_cb, rolling_cost_cb])

    # VScode can be used to open the Tensorboard frontend for live monitoring.

    opt.callback_history['TensorboardCallback']
    opt.callback_history['rolling_cost_cb']

"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"


@dataclass
class Callback:
    """Base callback class for optimizers. Users can inherit from this class and define the
    following custom logic:

    * `.trigger`:
        Custom triggering condition, other than the regular schedule set by `step_per_call`.
    * `.call`:
        The main routine to be customized.
    * `.update_cost_fn`:
        The custom cost_fn updater, which is expected to return a new cost_fn callable to
        replace the original one passed to the optimizer.
    * `.update_grads`:
        The custom grads modifyer, which is expected to return a list of parameter gradients
        after modification, to be applied to the parameters.
    * `.update_optimizer`:
        The custom optimizer updater, which is expected to modify the optimizer inplace for
        things like scheduling learning rates.

    """

    #: Custom tag for a callback instance to be used as keys in `Optimizer.callback_history`.
    #: Defaults to the class name.
    tag: str = None

    #: Sets calling frequency of this callback. Defaults to once per optimization step.
    #: Use higher values to reduce overhead.
    steps_per_call: int = 1

    def __post_init__(self):
        self.tag = self.tag or self.__class__.__name__
        self.optimizer_step: int = 0
        self.callback_step: int = 0

    def get_opt_step(self, optimizer, **kwargs):
        """Gets current step from optimizer."""
        self.optimizer_step = len(optimizer.opt_history)
        return self.optimizer_step

    def _should_call(self, **kwargs) -> bool:
        return (self.get_opt_step(**kwargs) % self.steps_per_call == 0) or self.trigger(**kwargs)

    def trigger(self, **kwargs) -> bool:
        """User implemented custom trigger conditions."""

    def call(self, **kwargs) -> Mapping | None:
        """User implemented main callback logic."""

    def update_cost_fn(self, **kwargs) -> Callable | None:
        """User implemented cost_fn modifier."""

    def update_grads(self, **kwargs) -> Sequence | None:
        """User implemented gradient modifier."""

    def update_optimizer(self, optimizer, **kwargs):
        """User implemented optimizer update scheduler."""

    def __call__(
        self,
        **kwargs,
    ):
        if self._should_call(**kwargs):
            self.callback_step += 1
            callback_result = {
                "optimizer_step": self.optimizer_step,
                "callback_step": self.callback_step,
            }

            callback_result.update(self.call(**kwargs) or {})

            new_cost_fn = self.update_cost_fn(callback_result=callback_result, **kwargs)
            if callable(new_cost_fn):
                callback_result["cost_fn"] = new_cost_fn

            new_grads = self.update_grads(callback_result=callback_result, **kwargs)
            if new_grads is not None:
                callback_result["grads"] = new_grads

            # Modifies the optimizer inplace, e.g. its learning rates.
            self.update_optimizer(callback_result=callback_result, **kwargs)

            return callback_result
        return {}


@dataclass
class TensorboardCallback(Callback):
    """Callback for enabling Tensorboard tracking of optimization progresses.

    Things tracked:

    * the cost
    * the transformed cost, if a `cost_converter` is provided
    * trainable parameter values
    * trainable parameter gradients (if `track_grads` is `True`)

    To start the Tensorboard frontend, either:

    * use VSCode: F1 -> Tensorboard -> select your `root_logdir/experiment_tag`.
    * use command line: `tensorboard --logdir=root_logdir/experiment_tag` and open link in browser.


    """

    #: The root logdir for tensorboard logging.
    root_logdir: str | Path = "./tb_logdir"

    #: The tag for experiment subfolder to group similar optimizations together for easy comparisons.
    #: Defaults to the hash of all trainable variables' names.
    experiment_tag: str | None = None

    #: Extra prefix to name the optimization experiment.
    prefix: str | None = None

    #: Transformation on cost for the purpose of better interpretation.
    cost_converter: Callable | None = None

    #: Whether to track gradients as well as the values for trainable parameters.
    track_grads: bool = False

    #: Whether to return objectives in the callback results to be stored.
    log_objectives: bool = True

    #: Whether to return parameter values in the callback results to be stored.
    log_trainables: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.root_logdir = Path(self.root_logdir)

        # Initialize only when first called to use optimization time rather than init time:
        self.logdir = None
        self.writter_logdir = None
        self.tb_writer = None

    def init_writer(self, trainables):
        """Initializes tb logdir folders and writer."""
        if (self.writter_logdir is None) or (self.optimizer_step <= self.steps_per_call):
            trainable_key_hash = hashlib.sha256(
                ",".join(trainables.keys()).encode("utf-8"),
            ).hexdigest()
            self.experiment_tag = self.experiment_tag or f"experiment-{trainable_key_hash[:7]}"
            self.logdir = self.root_logdir / self.experiment_tag
            self.prefix = self.prefix or "optim"
            existing_exp = [path for path in self.logdir.glob("*") if path.is_dir()]
            optim_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

            self.writter_logdir = self.logdir / (
                f"{self.prefix}-{len(existing_exp):03d}-{optim_timestamp}"
            )
            self.tb_writer = tf.summary.create_file_writer(str(self.writter_logdir))
            self.tb_writer.set_as_default()

    def call(
        self,
        optimizer,
        cost,
        trainables,
        **kwargs,
    ):
        """Logs costs and parameters to Tensorboard."""
        self.init_writer(trainables=trainables)
        obj_tag = "objectives"

        cost = np.array(cost).item()

        obj_scalars = {
            f"{obj_tag}/cost": cost,
        }
        if self.cost_converter is not None:
            obj_scalars[f"{obj_tag}/{self.cost_converter.__name__}(cost)"] = self.cost_converter(
                cost,
            )

        if "orig_cost" in optimizer.callback_history:
            orig_cost = np.array(optimizer.callback_history["orig_cost"][-1]).item()
            obj_scalars[f"{obj_tag}/orig_cost"] = orig_cost
            if self.cost_converter is not None:
                obj_scalars[f"{obj_tag}/{self.cost_converter.__name__}(orig_cost)"] = (
                    self.cost_converter(orig_cost)
                )

        for k, v in obj_scalars.items():
            tf.summary.scalar(k, data=v, step=self.optimizer_step)

        for k, (x, dx) in trainables.items():
            x_val = np.array(x.value)
            if self.track_grads:
                dx = np.array(dx)  # noqa: PLW2901

            tag = k if np.size(x_val) <= 1 else None
            for ind, val in np.ndenumerate(x_val):
                tag = tag or k + str(list(ind)).replace(" ", "")
                tf.summary.scalar(tag + ":value", data=val, step=self.optimizer_step)
                if self.track_grads:
                    tf.summary.scalar(tag + ":grad", data=dx[ind], step=self.optimizer_step)
                tag = None

        result = obj_scalars if self.log_objectives else {}

        if self.log_trainables:
            result.update(trainables)

        return result
