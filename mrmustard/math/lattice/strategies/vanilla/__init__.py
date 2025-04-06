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

from .batch import vanilla_b_batch, vanilla_full_batch, vanilla_stable_b_batch
from .core import vanilla, vanilla_stable
from .gradients import (
    vanilla_full_batch_vjp,
    vanilla_stable_vjp,
    vanilla_vjp,
)

__all__ = [
    "vanilla",
    "vanilla_stable",
    "vanilla_b_batch",
    "vanilla_stable_b_batch",
    "vanilla_full_batch",
    "vanilla_vjp",
    "vanilla_stable_vjp",
    "vanilla_full_batch_vjp",
]
