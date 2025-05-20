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

"""This module contains the implementation of optimization classes and functions
used within Mr Mustard.
"""
from __future__ import annotations

from typing import Callable, Sequence

import jax

import equinox as eqx

from mrmustard import math, settings
from mrmustard.lab import CircuitComponent
from mrmustard.training.progress_bar import ProgressBar
from mrmustard.utils.logger import create_logger
from mrmustard.math.parameters import (
    update_euclidean,
    update_orthogonal,
    update_symplectic,
    update_unitary,
)

__all__ = ["OptimizerJax"]


class Objective(eqx.Module):
    vars: list[dict[str, jax.Array]]

    def __init__(self, by_optimizing: Sequence[CircuitComponent]):
        self.vars = [
            {key: val.value for key, val in comp.parameters.variables.items()}
            for comp in by_optimizing
        ]

    def __call__(self, cost_fn: Callable, by_optimizing: Sequence[CircuitComponent]):
        for vars, comp in zip(self.vars, by_optimizing):
            for key, val in vars.items():
                comp.parameters.variables[key].value = val
        return cost_fn(*by_optimizing)


class OptimizerJax:
    def __init__(
        self,
        symplectic_lr: float = 0.1,
        unitary_lr: float = 0.1,
        orthogonal_lr: float = 0.1,
        euclidean_lr: float = 0.001,
        stable_threshold=1e-6,
    ):
        self.learning_rate = {
            update_euclidean: euclidean_lr,
            update_symplectic: symplectic_lr,
            update_unitary: unitary_lr,
            update_orthogonal: orthogonal_lr,
        }
        self.opt_history = [0]
        self.log = create_logger(__name__)
        self.stable_threshold = stable_threshold

    @eqx.filter_jit
    def make_step(
        self,
        optim,
        loss,
        model,
        opt_state,
    ):
        params, static = eqx.partition(model, eqx.is_array)
        loss_value, grads = jax.value_and_grad(loss)(params, static)
        updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    def minimize(self, cost_fn, by_optimizing, max_steps=1000):
        if settings.PROGRESSBAR:
            progress_bar = ProgressBar(max_steps)
            with progress_bar:
                model = self._optimization_loop(
                    cost_fn, by_optimizing, max_steps=max_steps, progress_bar=progress_bar
                )
        else:
            model = self._optimization_loop(cost_fn, by_optimizing, max_steps=max_steps)

        # update vals
        for vars, comp in zip(model.vars, by_optimizing):
            for key, val in vars.items():
                comp.parameters.variables[key].value = val

    def _optimization_loop(self, cost_fn, by_optimizing, max_steps, progress_bar=None):
        model = Objective(by_optimizing)

        def loss(params, static):
            model = eqx.combine(params, static)
            return model(cost_fn, by_optimizing)

        optim = math.euclidean_opt(learning_rate=self.learning_rate[update_euclidean])
        opt_state = optim.init(eqx.filter(model, eqx.is_array))

        # optimize
        while not self.should_stop(max_steps):
            model, opt_state, loss_value = self.make_step(optim, loss, model, opt_state)
            self.opt_history.append(loss_value)
            if progress_bar is not None:
                progress_bar.step(math.asnumpy(loss_value))

        return model

    def should_stop(self, max_steps: int) -> bool:
        if max_steps != 0 and len(self.opt_history) > max_steps:
            return True
        if len(self.opt_history) > 20:  # if cost varies less than threshold over 20 steps
            if (
                sum(abs(self.opt_history[-i - 1] - self.opt_history[-i]) for i in range(1, 20))
                < self.stable_threshold
            ):
                self.log.info("Loss looks stable, stopping here.")
                return True
        return False
