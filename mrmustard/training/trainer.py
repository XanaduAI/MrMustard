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

"""This module contains the implementation of distributed training utilities for parallelized
optimization of MrMustard circuits/devices through the function :meth:`map_trainer`.

This module requires extra dependencies, to install:

.. code-block:: bash

    git clone https://github.com/XanaduAI/MrMustard
    cd MrMustard
    pip install -e .[ray]


User-provided Wrapper Functions
===============================
To distribute your optimization workflow, two user-defined functions are needed for wrapping up user logic:

* A `device_factory` that wraps around the logic for making your circuits/states to be optimized; it is expected to return a single, or list of, :class:`Circuit`(s).
* A `cost_fn` that takes the circuits made and additional keyword arguments and returns a backprop-able scalar cost.

Separating the circuit-making logic from the cost calculation logic has the benefit of returning the optimized circuit in the result dict for further inspection. One can also pass extra `metric_fns` to directly extract info from the circuit.


Examples:
=========

.. code-block::

    from mrmustard.lab import Vacuum, Dgate, Ggate, Gaussian
    from mrmustard.physics import fidelity
    from mrmustard.training.trainer import map_trainer

    def make_circ(x=0.):
        return Ggate(num_modes=1, symplectic_trainable=True) >> Dgate(x=x, x_trainable=True, y_trainable=True)

    def cost_fn(circ=make_circ(0.1), y_targ=0.):
        target = Gaussian(1) >> Dgate(-1.5, y_targ)
        s = Vacuum(1) >> circ
        return -fidelity(s, target)

    # Use case 0: Calculate the cost of a randomly initialized circuit 5 times without optimizing it.
    results_0 = map_trainer(
        cost_fn=cost_fn,
        tasks=5,
    )

    # Use case 1: Run circuit optimization 5 times on randomly initialized circuits.
    results_1 = map_trainer(
        cost_fn=cost_fn,
        device_factory=make_circ,
        tasks=5,
        max_steps=50,
        symplectic_lr=0.05,
    )

    # Use case 2: Run 2 sets of circuit optimization with custom parameters passed as list.
    results_2 = map_trainer(
        cost_fn=cost_fn,
        device_factory=make_circ,
        tasks=[
            {'x': 0.1, 'euclidean_lr': 0.005, 'max_steps': 50},
            {'x': -0.7, 'euclidean_lr': 0.1, 'max_steps': 2},
        ],
        y_targ=0.35,
        symplectic_lr=0.05,
        AUTOCUTOFF_MAX_CUTOFF=7,
    )

    # Use case 3: Run 2 sets of circuit optimization with custom parameters passed as dict with extra metric functions for evaluating the final optimized circuit.
    results_3 = map_trainer(
    cost_fn=cost_fn,
    device_factory=make_circ,
    tasks={
        'my-job': {'x': 0.1, 'euclidean_lr': 0.005, 'max_steps': 50},
        'my-other-job': {'x': -0.7, 'euclidean_lr': 0.1, 'max_steps': 2},
    },
    y_targ=0.35,
    symplectic_lr=0.05,
    metric_fns={
        'is_gaussian': lambda c: c.is_gaussian,
        'foo': lambda _: 17.
    },
)


"""

import warnings
from functools import partial
from inspect import Parameter, signature
from typing import Mapping, Sequence

import numpy as np
from rich.progress import track

from mrmustard import settings
from .optimizer import Optimizer


def _apply_partial_cost(device, cost_fn, **kwargs):
    """Helper partial cost fn maker."""
    if isinstance(device, Sequence):
        cost_fn, kwargs = partial_pop(cost_fn, *device, **kwargs)
        optimized = device
    elif isinstance(device, Mapping):  # pragma: no cover
        cost_fn, kwargs = partial_pop(cost_fn, **device, **kwargs)
        optimized = list(device.values())
    return cost_fn, kwargs, optimized


def train_device(
    cost_fn,
    device_factory=None,
    metric_fns=None,
    return_kwargs=True,
    skip_opt=False,
    tag=None,
    **kwargs,
):
    """A general and flexible training loop for circuit optimizations with configurations adjustable through kwargs.

    Args:
        cost_fn (callable): The optimized cost function to be distributed. It's expected to accept the
            output of `device_factory` as *args as well as user-defined **kwargs, and returns a scalar cost.
            Its user-defined **kwargs will be passed from this function's **kwargs which must include all its
            required arguments.
        device_factory (callable): Function that (partially) takes `kwargs` and returns a device, or
            list/dict of devices. If None, `cost_fn` will be assumed to take no positional argument (for
            example, when device-making is contained in `cost_fn`). Defaults to None.
        metric_fns (Union[Sequence[callable], Mapping[callable], callable]): Optional collection of functions that takes the
            output of `device_factory` after optimization and returns arbitrary evaluation/information.
        return_kwargs (bool): Whether to include input config `kwargs` in the output dict. Defualts to True.
        skip_opt (bool): Whether to skip the optimization and directly calculate cost.
        tag (str): Optional label of the training task associated with the `kwargs` to be included in the output dict.
        kwargs:
            Dict containing all arguments to any of the functions below:
                - `cost_fn`: exluding the output of `device_factory`.
                - `device_factory`: e.g. `x`, `r`, `theta`, etc.
                - `Optimizer`: e.g. `euclidean_lr`.
                - `Optimizer.minimize`: excluding `cost_fn` and `by_optimizing`, e.g. `max_steps`.

    Returns:
        dict: A result dict summarizing the optimized circuit, cost, metrics and/or input configs.

    """

    setting_updates, kwargs = update_pop(settings, **kwargs)

    input_kwargs = kwargs.copy() if return_kwargs else {}

    device, kwargs = (
        curry_pop(device_factory, **kwargs)
        if callable(device_factory)
        else ([], kwargs)
    )
    device = [device] if not isinstance(device, (Sequence, Mapping)) else device

    cost_fn, kwargs, optimized = _apply_partial_cost(device, cost_fn, **kwargs)

    opt = None
    if optimized and not skip_opt:  # pragma: no cover
        opt, kwargs = curry_pop(Optimizer, **kwargs)
        _, kwargs = curry_pop(
            opt.minimize, **{"cost_fn": cost_fn, "by_optimizing": optimized}, **kwargs
        )

    if kwargs:
        warnings.warn(f"Unused kwargs: {kwargs}")

    final_cost = cost_fn()

    results = {
        "cost": np.array(final_cost).item(),
        "device": device,
        "optimizer": opt,
    }

    if callable(metric_fns):
        results["metrics"] = metric_fns(*device)  # pragma: no cover
    elif isinstance(metric_fns, Sequence):
        results["metrics"] = [
            f(*device) for f in metric_fns if callable(f)
        ]  # pragma: no cover
    elif isinstance(metric_fns, Mapping):  # pragma: no cover
        results = {
            **results,
            **{k: f(*device) for k, f in metric_fns.items() if callable(f)},
        }

    return {
        **({"tag": tag} if tag is not None else {}),
        **results,
        **input_kwargs,
        **setting_updates,
    }


def _iter_futures(futures):
    """Make ray futures iterable for easy passing to a progress bar.
    Hacky: https://github.com/ray-project/ray/issues/5554
    """
    import ray  # pylint: disable=import-outside-toplevel

    while futures:
        done, futures = ray.wait(futures)
        yield ray.get(done[0])


def map_trainer(
    trainer=train_device, tasks=1, pbar=True, unblock=False, num_cpus=None, **kwargs
):
    """Maps multiple training tasks across multiple workers using `ray`.

    In practice, the most common use case is to ignore the keywords `trainer` (as it defaults to
    :meth:`train_device`), `pbar`, `unblock`, etc. and just concentrate on `tasks` and `**kwargs`
    which passes arguments to the wrapper functions that contain the task execution logic, as well
    as the :class:`Optimizer` and its :meth:`Optimizer.minimize`.

    For example, with the default `trainer` :meth:`train_device`, two user-defined functions are used for wrapping up user logic:

    * A `device_factory` (optional) that wraps around the logic for making circuits/states to be optimized; it is expected to return a single, or list of, :class:`Circuit`(s).

    * A `cost_fn` (required) that takes the circuits made and additional keyword arguments and returns a backprop-able scalar cost.

    Refer to the `kwargs` section below for more available options.

    Args:
        trainer (callable): The function containing the training loop to be distributed, whose
            fixed arguments are to be passed by `**kwargs` and task-specific arguments iterated
            through `tasks`. Provide only when custom evaluation/training logic is needed.
            Defaults to :meth:`train_device`.
        tasks (Union[int, Sequence, Mapping]): Number of repeats or collection of task-specific training
            config arguments feeding into :meth:`train_device`.
            Refer to `kwargs` below for the available options.
            Defaults to 1 which runs `trainer` exactly once.
        pbar (bool): Whether to show a progress bar, available only in blocking mode (i.e. `unblock==False`). Defaults to True.
        unblock (bool): Whether to unblock the process and returns a getter function returning the available results.
            Defaults to False.
        num_cpus (int): Number of cpu workers to initialize ray. Defaults to the number of virtual cores.
        kwargs: Additional arguments containing fixed training config kwargs feeding into `trainer`.
            For the default `trainer` :meth:`train_device`, available options are:
                - cost_fn (callable):
                    The optimized cost function to be distributed. It's expected to accept the
                    output of `device_factory` as *args as well as user-defined **kwargs, and returns a scalar cost.
                    Its user-defined **kwargs will be passed from this function's **kwargs which must include all its
                    required arguments.
                - device_factory (callable):
                    Function that (partially) takes `kwargs` and returns a device, or
                    list/dict of devices. If None, `cost_fn` will be assumed to take no positional argument (for
                    example, when device-making is contained in `cost_fn`). Defaults to None.
                - metric_fns (Union[Sequence[callable], Mapping[callable], callable]):
                    Optional collection of functions that takes the
                    output of `device_factory` after optimization and returns arbitrary evaluation/information.
                - return_kwargs (bool):
                    Whether to include input config `kwargs` in the output dict. Defualts to True.
                - skip_opt (bool):
                    Whether to skip the optimization and directly calculate cost.
                - tag (str):
                    Optional label of the training task associated with the `kwargs` to be included in the output dict.
                - any kwargs to `cost_fn`: exluding the output of `device_factory`.
                - any kwargs to `device_factory`: e.g. `x`, `r`, `theta`, etc.
                - any kwargs to `Optimizer`: e.g. `euclidean_lr`.
                - any kwargs to `Optimizer.minimize`: excluding `cost_fn` and `by_optimizing`, e.g. `max_steps`.

    Returns
        Union[List, Dict]: The collection of results from each training task. Returns
            - a list if `tasks` is provided as an int or a list; or
            - a dict with the same keys if `tasks` is provided as a dict.


    Examples:
    =========

    .. code-block::

        from mrmustard.lab import Vacuum, Dgate, Ggate, Gaussian
        from mrmustard.physics import fidelity
        from mrmustard.training.trainer import map_trainer

        def make_circ(x=0.):
            return Ggate(num_modes=1, symplectic_trainable=True) >> Dgate(x=x, x_trainable=True, y_trainable=True)

        def cost_fn(circ=make_circ(0.1), y_targ=0.):
            target = Gaussian(1) >> Dgate(-1.5, y_targ)
            s = Vacuum(1) >> circ
            return -fidelity(s, target)

        # Use case 0: Calculate the cost of a randomly initialized circuit 5 times without optimizing it.
        results_0 = map_trainer(
            cost_fn=cost_fn,
            tasks=5,
        )

        # Use case 1: Run circuit optimization 5 times on randomly initialized circuits.
        results_1 = map_trainer(
            cost_fn=cost_fn,
            device_factory=make_circ,
            tasks=5,
            max_steps=50,
            symplectic_lr=0.05,
        )

        # Use case 2: Run 2 sets of circuit optimization with custom parameters passed as list.
        results_2 = map_trainer(
            cost_fn=cost_fn,
            device_factory=make_circ,
            tasks=[
                {'x': 0.1, 'euclidean_lr': 0.005, 'max_steps': 50},
                {'x': -0.7, 'euclidean_lr': 0.1, 'max_steps': 2},
            ],
            y_targ=0.35,
            symplectic_lr=0.05,
            AUTOCUTOFF_MAX_CUTOFF=7,
        )

        # Use case 3: Run 2 sets of circuit optimization with custom parameters passed as dict with extra metric functions for evaluating the final optimized circuit.
        results_3 = map_trainer(
        cost_fn=cost_fn,
        device_factory=make_circ,
        tasks={
            'my-job': {'x': 0.1, 'euclidean_lr': 0.005, 'max_steps': 50},
            'my-other-job': {'x': -0.7, 'euclidean_lr': 0.1, 'max_steps': 2},
        },
        y_targ=0.35,
        symplectic_lr=0.05,
        metric_fns={
            'is_gaussian': lambda c: c.is_gaussian,
            'foo': lambda _: 17.
        },
    )
    """
    try:
        import ray  # pylint: disable=import-outside-toplevel
    except ImportError as e:
        raise ImportError(
            "Failed to import `ray` which is an extra dependency. Please install with `pip install -e .[ray]`."
        ) from e

    if not ray.is_initialized():  # pragma: no cover
        ray.init(num_cpus=num_cpus)

    return_dict = False
    if isinstance(tasks, int):
        tasks = [{} for _ in range(tasks)]
    elif isinstance(tasks, Mapping):
        return_dict = True
        tasks = [{"tag": tag, **task} for tag, task in tasks.items()]

    remote_trainer, kwargs = partial_pop(
        ray.remote(trainer).remote,
        **kwargs,
    )

    if isinstance(tasks, Sequence):
        promises = [
            curry_pop(
                remote_trainer,
                **task_kwargs,
                **kwargs.copy(),
            )[0]
            for task_kwargs in tasks
            if isinstance(task_kwargs, Mapping)
        ]
    else:
        raise ValueError(
            f"`tasks` is expected to be of type int, list, or dict. got {type(tasks)}: {tasks}"
        )

    if not unblock:
        # blocks and wait till all tasks complete to return the end results.
        if pbar:
            results = list(
                track(
                    _iter_futures(promises),
                    description=f"{len(promises)} tasks running...",
                    total=len(promises),
                )
            )
        else:
            results = ray.get(promises)

        if return_dict:
            return {r["tag"]: r for r in results}
        else:
            return results

    else:
        # does not block and returns a getter function that returns the available results so far.
        def get_avail_results():
            results, running_tasks = ray.wait(  # pylint: disable=unused-variable
                promises, num_returns=len(promises)
            )
            if return_dict:
                return {r["tag"]: r for r in ray.get(results)}
            else:
                return ray.get(results)

        return get_avail_results


def kwargs_of(fn):
    """Gets the kwarg signature of a callable."""
    params = signature(fn).parameters
    kwarg_kinds = [Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY]

    keywords = [k for k, p in params.items() if p.kind in kwarg_kinds]
    has_var_keyword = any(p.kind is Parameter.VAR_KEYWORD for p in params.values())

    return keywords, has_var_keyword


def partial_pop(fn, *args, **kwargs):
    """Partially applies known kwargs to fn and returns the rest."""
    keywords, has_var_keyword = kwargs_of(fn)
    known_kwargs = {k: kwargs.pop(k) for k in set(kwargs).intersection(keywords)}
    partial_fn = partial(
        fn, *args, **known_kwargs, **(kwargs if has_var_keyword else {})
    )
    return partial_fn, kwargs


def curry_pop(fn, *args, **kwargs):
    """A poor man's reader monad bind function."""
    partial_fn, kwargs = partial_pop(fn, *args, **kwargs)
    return partial_fn(), kwargs


def update_pop(obj, **kwargs):
    """Updates an object/dict while popping keys out and returns the updated dict and remaining kwargs."""
    updated = {}
    if isinstance(obj, Mapping):
        for k in set(kwargs).intersection(obj):
            obj[k] = kwargs.pop(k)
            updated[k] = obj[k]
    else:
        for k in set(kwargs).intersection(dir(obj)):  # pragma: no cover
            setattr(obj, k, kwargs.pop(k))
            updated[k] = getattr(obj, k)
    return updated, kwargs
