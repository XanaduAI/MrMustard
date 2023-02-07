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

"""This module contains the implementation of distributed training utilities for optimizing
MrMustard circuits/devices.
"""

from inspect import signature, Parameter
from functools import partial
from typing import Sequence, Mapping
import warnings
import numpy as np
from rich.progress import track
import mrmustard as mm
from .optimizer import Optimizer

try:
    import ray
except ImportError as e:
    raise ImportError(
        "Failed to import `ray` which is an extra dependency. Please install with `pip install -e .[ray]`."
    ) from e


def train_device(
    cost_fn, device_factory=None, metric_fns=None, return_kwargs=True, skip_opt=False, tag=None, **kwargs
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
        kwargs: Dict containing all arguments to any of the functions below:
            - `cost_fn`: exluding the output of `device_factory`.
            - `device_factory`: e.g. `x`, `r`, `theta`, etc.
            - `Optimizer`: e.g. `euclidean_lr`.
            - `Optimizer.minimize`: excluding `cost_fn` and `by_optimizing`, e.g. `max_steps`.

    Returns:
        dict: A result dict summarizing the optimized circuit, cost, metrics and/or input configs.

    """

    setting_updates, kwargs = update_pop(mm.settings, **kwargs)

    input_kwargs = kwargs.copy() if return_kwargs else {}

    device = None
    if callable(device_factory):
        device, kwargs = curry_pop(device_factory, **kwargs)

    if isinstance(device, Sequence):
        optimized = device
        cost_fn, kwargs = partial_pop(cost_fn, *optimized, **kwargs)
    elif isinstance(device, Mapping):
        optimized = list(device.values())
        cost_fn, kwargs = partial_pop(cost_fn, **device, **kwargs)
    else:
        optimized = [device] if device is not None else []
        cost_fn, kwargs = partial_pop(cost_fn, *optimized, **kwargs)

    opt = None
    if optimized and not skip_opt:
        opt, kwargs = curry_pop(Optimizer, **kwargs)
        _, kwargs = curry_pop(opt.minimize, **{"cost_fn": cost_fn, "by_optimizing": optimized}, **kwargs)

    if kwargs:
        warnings.warn(f"Unused kwargs: {kwargs}")

    final_cost = cost_fn()

    results = {
        "cost": np.array(final_cost).item(),
        "device": device,
        "optimizer": opt,
    }

    if callable(metric_fns):
        results["metrics"] = metric_fns(*device)
    elif isinstance(metric_fns, Sequence):
        results["metrics"] = [f(*device) for f in metric_fns if callable(f)]
    elif isinstance(metric_fns, Mapping):
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
    while futures:
        done, futures = ray.wait(futures)
        yield ray.get(done[0])


def map_trainer(trainer=train_device, tasks=1, pbar=True, unblock=False, num_cpus=None, **kwargs):
    """Maps multiple training tasks across multiple workers using `ray`.

    Args:
        trainer (callable): The function containing the training loop to be distributed, whose
            fixed arguments are to be passed by `**kwargs` and task-specific arguments iterated through `tasks`.
            Defaults to the `train_device` function.
        tasks (Union[int, Sequence, Mapping]): Number of repeats or collection of task-specific training
            config arguments feeding into `train_device`.
            Refer to `kwargs` below for the available options.
            Defaults to 1 which runs `trainer` exactly once.
        pbar (bool): Whether to show a progress bar, available only in blocking mode (i.e. `unblock==False`). Defaults to True.
        unblock (bool): Whether to unblock the process and returns a getter function returning the available results.
            Defaults to False.
        num_cpus (int): Number of cpu workers to initialize ray. Defaults to the number of virtual cores.
        kwargs: Additional arguments containing fixed training config kwargs feeding into `trainer`.
        For the default `trainer` `train_device`, available options are:
            - cost_fn (callable): The optimized cost function to be distributed. It's expected to accept the
                output of `device_factory` as *args as well as user-defined **kwargs, and returns a scalar cost.
                Its user-defined **kwargs will be passed from this function's **kwargs which must include all its
                required arguments.
            - device_factory (callable): Function that (partially) takes `kwargs` and returns a device, or
                list/dict of devices. If None, `cost_fn` will be assumed to take no positional argument (for
                example, when device-making is contained in `cost_fn`). Defaults to None.
            - metric_fns (Union[Sequence[callable], Mapping[callable], callable]): Optional collection of functions that takes the
                output of `device_factory` after optimization and returns arbitrary evaluation/information.
            - return_kwargs (bool): Whether to include input config `kwargs` in the output dict. Defualts to True.
            - skip_opt (bool): Whether to skip the optimization and directly calculate cost.
            - tag (str): Optional label of the training task associated with the `kwargs` to be included in the output dict.
            - any kwargs to `cost_fn`: exluding the output of `device_factory`.
            - any kwargs to `device_factory`: e.g. `x`, `r`, `theta`, etc.
            - any kwargs to `Optimizer`: e.g. `euclidean_lr`.
            - any kwargs to `Optimizer.minimize`: excluding `cost_fn` and `by_optimizing`, e.g. `max_steps`.

    Returns
        Union[List, Dict]: The collection of results from each training task. Returns
            - a list if `tasks` is provided as an int or a list; or
            - a dict with the same keys if `tasks` is provided as a dict.
    """

    if not ray.is_initialized():
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
            )[0]
            for task_kwargs in tasks
            if isinstance(task_kwargs, Mapping)
        ]
    else:
        raise ValueError(f"`tasks` is expected to be of type int, list, or dict. got {type(tasks)}: {tasks}")

    if not unblock:
        # blocks and wait till all tasks complete to return the end results.
        if pbar:
            # results = [
            #     result
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
    partial_fn = partial(fn, *args, **known_kwargs, **(kwargs if has_var_keyword else {}))
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
        for k in set(kwargs).intersection(dir(obj)):
            setattr(obj, k, kwargs.pop(k))
            updated[k] = getattr(obj, k)
    return updated, kwargs
