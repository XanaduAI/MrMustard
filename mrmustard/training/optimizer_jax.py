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

from typing import Callable, Sequence

import jax

import equinox as eqx

from mrmustard import math
from mrmustard.lab import CircuitComponent

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

    def minimize(self, cost_fn, by_optimizing, max_steps=10):
        # loss function that accepts parameters and updates the circuit and returns the cost_fn
        def loss(params, static):
            model = eqx.combine(params, static)
            return model(cost_fn, by_optimizing)

        optim = math.euclidean_opt
        model = Objective(by_optimizing)
        opt_state = optim.init(eqx.filter(model, eqx.is_array))

        # optimize
        for epoch in range(max_steps):
            model, opt_state, loss_value = self.make_step(optim, loss, model, opt_state)
            print(f"epoch: {epoch}, loss: {loss_value}")

        # update vals
        for vars, comp in zip(model.vars, by_optimizing):
            for key, val in vars.items():
                comp.parameters.variables[key].value = val
