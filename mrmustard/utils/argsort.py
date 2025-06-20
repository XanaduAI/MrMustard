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

"""Sorting functions"""

from collections import defaultdict
from collections.abc import Generator

import numpy as np


def argsort_gen(generators: list[Generator[float, None, None]]) -> list[int]:
    r"""
    Sorts a list of generator objects based on their yielded values.

    This function takes a list of generator objects, each yielding a sequence of numbers.
    It sorts the generators based on their first yielded values. If multiple generators
    yield the same first value, the function recursively sorts them based on their next
    yielded values. The result is a list of indices that would sort the original list of
    generators.

    Parameters:
    gen_list (list): A list of generator objects, each yielding a sequence of numbers.

    Returns:
    list: A list of indices that would sort the original list of generators.
    """
    vals = []
    for gen in generators:
        try:
            vals.append(next(gen))
        except StopIteration:
            vals.append(np.inf)

    if np.allclose(vals, np.inf):
        return list(range(len(generators)))

    # dict with values and indices where they occur
    vals_dict = defaultdict(list)
    for i, val in enumerate(vals):
        vals_dict[val].append(i)

    # sort dict by keys (vals)
    vals_dict = {key: vals_dict[key] for key in sorted(vals_dict)}

    # if there are multiple values in the same pool, sort them with argsort_gen
    for val, index_pool in vals_dict.items():
        if len(index_pool) > 1:
            sub_gen_list = [generators[i] for i in index_pool]
            sub_order = argsort_gen(sub_gen_list)
            pool_sorted = [index_pool[i] for i in sub_order]
            vals_dict[val] = pool_sorted

    return [i for pool in vals_dict.values() for i in pool]
