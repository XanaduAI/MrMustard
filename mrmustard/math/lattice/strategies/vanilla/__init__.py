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

"""Vanilla strategies for Fock representation calculation, including core, batched, and gradient functions."""

from .batch import vanilla_batch_numba
from .core import stable_numba, vanilla_numba
from .gradients import vanilla_batch_vjp_numba, vanilla_vjp_numba

__all__ = [
    "stable_numba",
    "vanilla_batch_numba",
    "vanilla_batch_vjp_numba",
    "vanilla_numba",
    "vanilla_vjp_numba",
]
