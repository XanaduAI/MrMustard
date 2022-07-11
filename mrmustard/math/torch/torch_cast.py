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

from functools import wraps
import torch
import numpy as np
from .torch import TorchMath

class TorchCast:
    """Casts function input arrays to torch tensors"""

    def __init__(self) -> None:
        self._torchmath = TorchMath()

    def is_nparray(self, arg):
        r"""Returns `True` if the `arg` can (and should) be casted to `proposed_dtype`."""
        return isinstance(arg, (np.ndarray, np.generic, int, float, complex))

    def cast_args(self, *args, **kwargs):
        r"""Casts array arguments to torch Tensor."""
        args = [torch.tensor(arg) if self.is_nparray(arg) else arg for arg in args]
        kwargs = {k: torch.tensor(v) if self.is_nparray(v) else v for k, v in kwargs.items()}
        return args, kwargs

    def method(self, name):
        func = object.__getattribute__(self._torchmath, name)

        @wraps(func)
        def wrapper(*args, **kwargs):
            args, kwargs = self.cast_args(*args, **kwargs)
            return func(*args, **kwargs)

        return wrapper