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

"""This module contains the logic for cachin tensor functions in Mr Mustard."""

from functools import lru_cache, wraps

import numpy as np

from mrmustard.math.backend_manager import BackendManager

math = BackendManager()


def tensor_int_cache(fn):
    """Decorator function to cache functions with a 1D Tensor (Vector) and int as arguments,
    that is, functions with signature ``func(x: Vector, n: int)``.

    To do so the input vector (non-hashable) is converted into a numpy array (non-hashable)
    and then into a tuple (hashable). This tuple is used by ``functools.lru_cache`` to cache
    the result."""

    @lru_cache
    def cached_wrapper(hashable_array, cutoff):
        array = np.array(hashable_array, dtype=np.float64)
        return fn(array, cutoff)

    @wraps(fn)
    def wrapper(tensor, cutoff):
        return cached_wrapper(tuple(math.asnumpy(tensor)), cutoff)

    # copy lru_cache attributes over too
    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper
