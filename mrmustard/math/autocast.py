# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from mrmustard.types import *
from functools import lru_cache
from scipy.special import binom
from scipy.stats import unitary_group
from itertools import product
from functools import lru_cache, wraps


class Autocast:
    r"""A decorator that casts all castable arguments of a method to the dtype with highest precision."""

    def __init__(self):
        self.dtype_order = ("float16", "float32", "float64", "complex64", "complex128")
        self.no_cast = ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64")

    def can_cast(self, arg):
        return hasattr(arg, "dtype") and arg.dtype.name not in self.no_cast

    def should_cast(self, arg, proposed_dtype):
        if not self.can_cast(arg):
            return False
        return self.dtype_order.index(proposed_dtype) > self.dtype_order.index(arg.dtype.name)

    def get_dtypes(self, *args, **kwargs) -> List:
        r"""Returns the dtypes of the arguments."""
        args_dtypes = [arg.dtype.name for arg in args if self.can_cast(arg)]
        kwargs_dtypes = [v.dtype.name for v in kwargs.values() if self.can_cast(v)]
        return args_dtypes + kwargs_dtypes

    def max_dtype(self, dtypes: List):
        r"""Returns the dtype with the highest precision."""
        if dtypes == []:
            return None
        return max(dtypes, key=lambda dtype: self.dtype_order.index(dtype))

    def cast_all(self, backend, *args, **kwargs):
        r"""Casts all arguments to the highest precision when possible and needed."""
        max_dtype = self.max_dtype(self.get_dtypes(*args, **kwargs))
        args = [
            backend.cast(arg, max_dtype) if self.should_cast(arg, max_dtype) else arg
            for arg in args
        ]
        kwargs = {
            k: backend.cast(v, max_dtype) if self.should_cast(v, max_dtype) else v
            for k, v in kwargs.items()
        }
        return args, kwargs

    def __call__(self, func):
        @wraps(func)
        def wrapper(backend, *args, **kwargs):
            args, kwargs = self.cast_all(backend, *args, **kwargs)
            return func(backend, *args, **kwargs)

        return wrapper
