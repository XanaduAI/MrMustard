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
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np
import tensorflow as tf

from mrmustard import Optimizer
from mrmustard.utils import graphics
from mrmustard.training.parameter_update import param_update_method


@dataclass
class TensorboardCallback:
    logdir: str = "./tb_logdir"
    tag: str = None
    with_dB: bool = True

    def __post_init__(self):
        self.tag = "" if not self.tag else f"-{self.tag}"
        self.writter_logdir = Path(self.logdir) / (
            datetime.now().strftime("%Y%m%d-%H%M%S") + self.tag
        )
        self.tb_writer = tf.summary.create_file_writer(str(self.writter_logdir))
        self.tb_writer.set_as_default()

    def __call__(
        self,
        optimizer,
        cost,
        trainables,
        **kwargs,
    ):

        step = len(optimizer.opt_history)
        cost = np.array(cost).item()
        tf.summary.scalar("cost", data=cost, step=step)
        if self.with_dB:
            delta = np.sqrt(np.log(1 / (abs(cost) ** 2)) / (2 * np.pi))
            cost_dB = -10 * np.log10(delta**2)
            tf.summary.scalar("dB", data=cost_dB, step=step)

        if "orig_cost" in optimizer.callback_history:
            orig_cost = np.array(optimizer.callback_history["orig_cost"][-1]).item()
            tf.summary.scalar("orig_cost", data=orig_cost, step=step)
            if self.with_dB:
                orig_delta = np.sqrt(np.log(1 / (abs(orig_cost) ** 2)) / (2 * np.pi))
                orig_cost_dB = -10 * np.log10(orig_delta**2)
                tf.summary.scalar("orig_dB", data=orig_cost_dB, step=step)

        for k, v in trainables.items():
            tf.summary.scalar(k, data=v.value.numpy(), step=step)

        return {
            "step": step,
            "cost": cost,
            "dB": cost_dB,
            **trainables,
        }
