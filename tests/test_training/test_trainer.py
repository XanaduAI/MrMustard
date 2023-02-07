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

"""Tests for the ray-based trainer."""

import pytest

import numpy as np
from mrmustard.lab import Vacuum, Dgate, Ggate, Gaussian
from mrmustard.physics import fidelity
from mrmustard.training import Optimizer
from mrmustard.training.trainer import map_trainer


@pytest.fixture
def wrappers():
    def make_circ(x=0.0, return_list=False):
        circ = Ggate(num_modes=1, symplectic_trainable=True) >> Dgate(
            x=x, x_trainable=True, y_trainable=True
        )
        return circ if not return_list else [circ]

    def cost_fn(circ=make_circ(0.1), y_targ=0.0):
        target = Gaussian(1) >> Dgate(-1.5, y_targ)
        s = Vacuum(1) >> circ
        return -fidelity(s, target)

    return make_circ, cost_fn


@pytest.mark.parametrize(
    "tasks", [5, [{"y_targ": 0.1}, {"y_targ": -0.2}], {"c0": {}, "c1": {"y_targ": -0.7}}]
)
@pytest.mark.parametrize("seed", [None, 42])
def test_circ_cost(wrappers, tasks, seed):
    """Test distributed cost calculations."""
    has_seed = isinstance(seed, int)
    _, cost_fn = wrappers
    results = map_trainer(
        cost_fn=cost_fn,
        tasks=tasks,
        **({"SEED": seed} if has_seed else {}),
    )

    if isinstance(tasks, dict):
        set(results.keys()) == set(tasks.keys())
        results = list(results.values())
    assert all(r["optimizer"] is None for r in results)
    assert all(r["device"] == [] for r in results)
    if has_seed and isinstance(tasks, int):
        assert len(set(r["cost"] for r in results)) == 1
    else:
        assert (
            len(set(r["cost"] for r in results))
            >= (tasks if isinstance(tasks, int) else len(tasks)) - 1
        )


@pytest.mark.parametrize(
    "tasks", [[{"x": 0.1}, {"y_targ": 0.2}], {"c0": {}, "c1": {"euclidean_lr": 0.02, "HBAR": 1.0}}]
)
def test_circ_optimize(wrappers, tasks):
    """Test distributed optimizations."""
    max_steps = 10
    make_circ, cost_fn = wrappers
    results = map_trainer(
        cost_fn=cost_fn,
        device_factory=make_circ,
        tasks=tasks,
        max_steps=max_steps,
        symplectic_lr=0.05,
    )

    if isinstance(tasks, dict):
        set(results.keys()) == set(tasks.keys())
        results = list(results.values())
    assert (
        len(set(r["cost"] for r in results))
        >= (tasks if isinstance(tasks, int) else len(tasks)) - 1
    )
    assert all(isinstance(r["optimizer"], Optimizer) for r in results)
    assert all((r["optimizer"].opt_history) for r in results)

    # Check if optimization history is actually decreasing.
    opt_history = np.array(results[0]["optimizer"].opt_history)
    assert len(opt_history) == max_steps + 1
    assert opt_history[0] - opt_history[-1] > 1e-6
    assert (np.diff(opt_history) < 0).sum() > max_steps // 2


@pytest.mark.parametrize(
    "metric_fns",
    [
        {"is_gaussian": lambda c: c.is_gaussian, "foo": lambda _: 17.0},
        [
            lambda c: c.modes,
            lambda c: len(c),
        ],
        lambda c: (Vacuum(1) >> c >> c >> c).fock_probabilities([5]),
    ],
)
def test_circ_optimize_metrics(wrappers, metric_fns):
    """Tests custom metric functions on final circuits."""
    make_circ, cost_fn = wrappers

    tasks = {
        "my-job": {"x": 0.1, "euclidean_lr": 0.005, "max_steps": 20},
        "my-other-job": {"x": -0.7, "euclidean_lr": 0.1, "max_steps": 12},
    }

    results = map_trainer(
        cost_fn=cost_fn,
        device_factory=make_circ,
        tasks=tasks,
        y_targ=0.35,
        symplectic_lr=0.05,
        metric_fns=metric_fns,
        return_list=True,
    )

    set(results.keys()) == set(tasks.keys())
    results = list(results.values())
    assert all(("metrics" in r or set(metric_fns.keys()).issubset(set(r.keys()))) for r in results)
    assert (
        len(set(r["cost"] for r in results))
        >= (tasks if isinstance(tasks, int) else len(tasks)) - 1
    )
    assert all(isinstance(r["optimizer"], Optimizer) for r in results)
    assert all((r["optimizer"].opt_history) for r in results)

    # Check if optimization history is actually decreasing.
    opt_history = np.array(results[0]["optimizer"].opt_history)
    assert opt_history[0] - opt_history[-1] > 1e-6
