# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .strategies import (
    beamsplitter,
    beamsplitter_batched,
    displacement,
    displacement_batched,
    homodyne_projector,
    homodyne_projector_batched,
    jacobian_displacement,
    squeezed,
    squeezed_batched,
    squeezer,
    squeezer_batched,
)
from .utils import np_inv_sqrt as INV_SQRT
from .utils import np_sqrt as SQRT
from .vanilla import vanilla, vanilla_batched, vanilla_vjp, vanilla_vjp_batched

__all__ = [
    "INV_SQRT",
    "SQRT",
    "beamsplitter",
    "beamsplitter_batched",
    "displacement",
    "displacement_batched",
    "homodyne_projector",
    "homodyne_projector_batched",
    "jacobian_displacement",
    "squeezed",
    "squeezed_batched",
    "squeezer",
    "squeezer_batched",
    "vanilla",
    "vanilla_batched",
    "vanilla_vjp",
    "vanilla_vjp_batched",
]
