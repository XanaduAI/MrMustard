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

# pylint: disable=import-outside-toplevel

"""Tests for the ray-based trainer."""

import sys
from time import sleep

import numpy as np
import pytest

try:
    import ray

    ray_available = True

    NUM_CPUS = 1
    ray.init(num_cpus=NUM_CPUS)
except ImportError:
    ray_available = False

from mrmustard.lab import Dgate, Gaussian, Ggate, Vacuum
from mrmustard.physics import fidelity
from mrmustard.training import Optimizer
from mrmustard.training.trainer import map_trainer, train_device, update_pop

from ..conftest import skip_np


def wrappers():
    """Dummy wrappers tested."""

    def make_circ(x=0.0, return_type=None):
        from mrmustard import math

        math.change_backend("tensorflow")

        circ = Ggate(num_modes=1, symplectic_trainable=True) >> Dgate(
            x=x, x_trainable=True, y_trainable=True
        )
        return (
            [circ]
            if return_type == "list"
            else {"circ": circ}
            if return_type == "dict"
            else circ
        )

    def cost_fn(circ=make_circ(0.1), y_targ=0.0):
        from mrmustard import math

        math.change_backend("tensorflow")

        target = Gaussian(1) >> Dgate(-0.1, y_targ)
        s = Vacuum(1) >> circ
        return -fidelity(s, target)

    return make_circ, cost_fn


@pytest.mark.skipif(not ray_available, reason="ray is not available")
class TestTrainer:
    """Class containinf ray-related tests."""

    @pytest.mark.parametrize(
        "tasks",
        [5, [{"y_targ": 0.1}, {"y_targ": -0.2}], {"c0": {}, "c1": {"y_targ": 0.07}}],
    )
    @pytest.mark.parametrize("seed", [None, 42])
    def test_circ_cost(self, tasks, seed):  # pylint: disable=redefined-outer-name
        """Test distributed cost calculations."""
        skip_np()

        has_seed = isinstance(seed, int)
        _, cost_fn = wrappers()
        results = map_trainer(
            cost_fn=cost_fn,
            tasks=tasks,
            num_cpus=NUM_CPUS,
            **({"SEED": seed} if has_seed else {}),
        )

        if isinstance(tasks, dict):
            assert set(results.keys()) == set(tasks.keys())
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
        "tasks",
        [[{"x": 0.1}, {"y_targ": 0.2}], {"c0": {}, "c1": {"euclidean_lr": 0.02}}],
    )
    @pytest.mark.parametrize(
        "return_type",
        [None, "dict"],
    )
    def test_circ_optimize(
        self, tasks, return_type
    ):  # pylint: disable=redefined-outer-name
        """Test distributed optimizations."""
        skip_np()

        max_steps = 15
        make_circ, cost_fn = wrappers()
        results = map_trainer(
            cost_fn=cost_fn,
            device_factory=make_circ,
            tasks=tasks,
            max_steps=max_steps,
            symplectic_lr=0.05,
            return_type=return_type,
            num_cpus=NUM_CPUS,
        )

        if isinstance(tasks, dict):
            assert set(results.keys()) == set(tasks.keys())
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
        assert (np.diff(opt_history) < 0).sum() >= 3

    @pytest.mark.parametrize(
        "metric_fns",
        [
            {"is_gaussian": lambda c: c.is_gaussian, "foo": lambda _: 17.0},
            [
                lambda c: c.modes,
                len,
            ],
            lambda c: (Vacuum(1) >> c >> c >> c).fock_probabilities([5]),
        ],
    )
    def test_circ_optimize_metrics(
        self, metric_fns
    ):  # pylint: disable=redefined-outer-name
        """Tests custom metric functions on final circuits."""
        skip_np()

        make_circ, cost_fn = wrappers()

        tasks = {
            "my-job": {"x": 0.1, "euclidean_lr": 0.01, "max_steps": 100},
            "my-other-job": {"x": -0.7, "euclidean_lr": 0.1, "max_steps": 20},
        }

        results = map_trainer(
            cost_fn=cost_fn,
            device_factory=make_circ,
            tasks=tasks,
            y_targ=0.35,
            symplectic_lr=0.05,
            metric_fns=metric_fns,
            return_list=True,
            num_cpus=NUM_CPUS,
        )

        assert set(results.keys()) == set(tasks.keys())
        results = list(results.values())
        assert all(
            ("metrics" in r or set(metric_fns.keys()).issubset(set(r.keys())))
            for r in results
        )
        assert (
            len(set(r["cost"] for r in results))
            >= (tasks if isinstance(tasks, int) else len(tasks)) - 1
        )
        assert all(isinstance(r["optimizer"], Optimizer) for r in results)
        assert all((r["optimizer"].opt_history) for r in results)

        # Check if optimization history is actually decreasing.
        opt_history = np.array(results[0]["optimizer"].opt_history)
        assert opt_history[1] - opt_history[-1] > 1e-6

    def test_update_pop(self):
        """Test for coverage."""
        skip_np()

        d = {"a": 3, "b": "foo"}
        kwargs = {"b": "bar", "c": 22}
        d1, kwargs = update_pop(d, **kwargs)
        assert d1["b"] == "bar"
        assert len(kwargs) == 1

    def test_no_ray(self, monkeypatch):
        """Tests ray import error"""
        skip_np()

        monkeypatch.setitem(sys.modules, "ray", None)
        with pytest.raises(ImportError, match="Failed to import `ray`"):
            _ = map_trainer(
                tasks=2,
                num_cpus=NUM_CPUS,
            )

    def test_invalid_tasks(self):
        """Tests unexpected tasks arg"""
        skip_np()

        with pytest.raises(
            ValueError, match="`tasks` is expected to be of type int, list, or dict."
        ):
            _ = map_trainer(
                tasks=2.3,
                num_cpus=NUM_CPUS,
            )

    def test_warn_unused_kwargs(self):  # pylint: disable=redefined-outer-name
        """Test warning of unused kwargs"""
        skip_np()

        _, cost_fn = wrappers()
        with pytest.warns(UserWarning, match="Unused kwargs:"):
            results = train_device(
                cost_fn=cost_fn,
                foo="bar",
            )
        assert len(results) >= 4
        assert isinstance(results["cost"], float)

    def test_no_pbar(self):  # pylint: disable=redefined-outer-name
        """Test turning off pregress bar"""
        skip_np()

        _, cost_fn = wrappers()
        results = map_trainer(
            cost_fn=cost_fn,
            tasks=2,
            pbar=False,
            num_cpus=NUM_CPUS,
        )
        assert len(results) == 2

    @pytest.mark.parametrize("tasks", [2, {"c0": {}, "c1": {"y_targ": -0.7}}])
    def test_unblock(self, tasks):  # pylint: disable=redefined-outer-name
        """Test unblock async mode"""
        skip_np()

        _, cost_fn = wrappers()
        result_getter = map_trainer(
            cost_fn=cost_fn,
            tasks=tasks,
            unblock=True,
            num_cpus=NUM_CPUS,
        )
        assert callable(result_getter)

        sleep(0.2)
        results = result_getter()
        if len(results) <= (tasks if isinstance(tasks, int) else len(tasks)):
            # safer on slower machines
            sleep(1)
            results = result_getter()

        assert len(results) == (tasks if isinstance(tasks, int) else len(tasks))
