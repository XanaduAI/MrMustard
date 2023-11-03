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

# pylint: disable=no-member

"""
This module contains the utility functions used by the classes in ``mrmustard.lab``.
"""
from typing import Callable, Optional, Tuple

from mrmustard.math.parameters import update_euclidean, Constant, Variable


def make_parameter(
    is_trainable: bool,
    value: any,
    name: str,
    bounds: Tuple[Optional[float], Optional[float]],
    update_fn: Callable = update_euclidean,
):
    r"""
    Returns a constant or variable parameter with given name, value, bounds, and update function.

    Args:
        is_trainable: Whether to return a variable (``True``) or constant (``False``) parameter.
        value: The value of the returned parameter.
        name: The name of the returned parameter.
        bounds: The bounds of the returned parameter (ignored if ``is_trainable`` is ``False``).
        update_fn: The update_fn of the returned parameter (ignored if ``is_trainable`` is ``False``).
    """
    if isinstance(value, (Constant, Variable)):
        return value
    if not is_trainable:
        return Constant(value=value, name=name)
    return Variable(value=value, name=name, bounds=bounds, update_fn=update_fn)
