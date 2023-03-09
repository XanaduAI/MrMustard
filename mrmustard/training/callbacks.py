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

"""This module contains the implementation of common callbacks for optimizations.
"""

from dataclasses import dataclass
from datetime import datetime
import hashlib
from pathlib import Path
from typing import Callable, Optional, Mapping, Sequence, Union
import numpy as np
import tensorflow as tf
from mrmustard.math import Math

math = Math()


@dataclass
class Callback:
    """Base callback class for optimizers."""

    tag: str = None
    steps_per_call: int = 1

    def __post_init__(self):
        self.tag = self.tag or self.__class__.__name__
        self.optimizer_step: int = 0
        self.callback_step: int = 0

    def get_opt_step(self, optimizer, **kwargs):  # pylint: disable=unused-argument
        """Gets current step from optimizer."""
        self.optimizer_step = len(optimizer.opt_history)
        return self.optimizer_step

    def _should_call(self, **kwargs) -> bool:
        return (self.get_opt_step(**kwargs) % self.steps_per_call == 0) or self.trigger(**kwargs)

    def trigger(self, **kwargs) -> bool:  # pylint: disable=unused-argument
        """User implemented custom trigger conditions."""

    def call(self, **kwargs) -> Optional[Mapping]:  # pylint: disable=unused-argument
        """User implemented main callback logic."""

    def update_cost_fn(self, **kwargs) -> Optional[Callable]:  # pylint: disable=unused-argument
        """User implemented cost_fn modifier."""

    def update_grads(self, **kwargs) -> Optional[Sequence]:  # pylint: disable=unused-argument
        """User implemented gradient modifier."""

    def update_optimizer(self, optimizer, **kwargs):  # pylint: disable=unused-argument
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
class TensorboardCallback(Callback):  # pylint: disable=too-many-instance-attributes
    """Callback for enabling Tensorboard tracking of optimizations."""

    root_logdir: Union[str, Path] = "./tb_logdir"
    experiment_tag: Optional[str] = None
    prefix: Optional[str] = None
    cost_converter: Optional[Callable] = None
    track_grads: bool = False
    log_objectives: bool = True
    log_trainables: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.root_logdir = Path(self.root_logdir)

        # Initialize only when first called to use optimization time rather than init time:
        self.writter_logdir = None
        self.tb_writer = None

    def init_writer(self, trainables):
        """Initializes tb logdir folders and writer."""
        if (self.writter_logdir is None) or (self.optimizer_step <= self.steps_per_call):
            trainable_key_hash = hashlib.sha256(
                ",".join(trainables.keys()).encode("utf-8")
            ).hexdigest()
            self.experiment_tag = self.experiment_tag or f"experiment-{trainable_key_hash[:7]}"
            self.logdir = self.root_logdir / self.experiment_tag
            self.prefix = self.prefix or f"optim_{len(trainables)}_params"
            optim_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

            self.writter_logdir = self.logdir / (f"{self.prefix}-{optim_timestamp}")
            self.tb_writer = tf.summary.create_file_writer(str(self.writter_logdir))
            self.tb_writer.set_as_default()

    def call(
        self,
        optimizer,
        cost,
        trainables,
        **kwargs,
    ):  # pylint: disable=unused-argument,arguments-differ
        """Logs costs and parameters to Tensorboard."""

        self.init_writer(trainables=trainables)
        obj_tag = "objectives"

        cost = np.array(cost).item()

        obj_scalars = {
            f"{obj_tag}/cost": cost,
        }
        if self.cost_converter is not None:
            obj_scalars[f"{obj_tag}/{self.cost_converter.__name__}(cost)"] = self.cost_converter(
                cost
            )

        if "orig_cost" in optimizer.callback_history:
            orig_cost = np.array(optimizer.callback_history["orig_cost"][-1]).item()
            obj_scalars[f"{obj_tag}/orig_cost"] = orig_cost
            if self.cost_converter is not None:
                obj_scalars[
                    f"{obj_tag}/{self.cost_converter.__name__}(orig_cost)"
                ] = self.cost_converter(orig_cost)

        for k, v in obj_scalars.items():
            tf.summary.scalar(k, data=v, step=self.optimizer_step)

        for k, (x, dx) in trainables.items():
            x = np.array(x.value)
            if self.track_grads:
                dx = np.array(dx)

            tag = k if np.size(x) <= 1 else None
            for ind, val in np.ndenumerate(x):
                tag = tag or k + str(list(ind)).replace(" ", "")
                tf.summary.scalar(tag + ":value", data=val, step=self.optimizer_step)
                if self.track_grads:
                    tf.summary.scalar(tag + ":grad", data=dx[ind], step=self.optimizer_step)
                tag = None

        result = obj_scalars if self.log_objectives else {}

        if self.log_trainables:
            result.update(trainables)

        return result
