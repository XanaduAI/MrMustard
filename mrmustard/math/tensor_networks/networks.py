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

""" Functions and classes for tensor networks."""

from __future__ import annotations

from opt_einsum import contract as opt_contract
from typing import Optional

from .tensors import Wire, Tensor


def connect(wire1: Wire, wire2: Wire, dim: Optional[int] = None):
    r"""Connects two wires in a tensor network.

    Args:
        wire1: The first wire.
        wire2: The second wire.
        dim: The dimension of the contraction.

    Raises:
        ValueError: If one or both of the wires are already connected.
    """
    if wire1.is_connected or wire2.is_connected:
        msg = "Tried to connect wires that are already connected."
        raise ValueError(msg)

    if dim:
        wire1.dim = dim
        wire2.dim = dim

    wire1.is_connected = True
    wire2.is_connected = True

    wire1.contraction_id = wire2.contraction_id


def contract(tensors: list[Tensor], default_dim: int):
    r"""Contract a list of tensors.

    Args:
        tensors: the tensors to contract.
        dim: the default dimension of the contractions.

    Returns:
        (tensor) the contracted tensor
    """
    opt_einsum_args = []
    for t in tensors:
        shape = t.shape(default_dim=default_dim, swap=True)
        opt_einsum_args.append(t.value(shape=shape))
        opt_einsum_args.append([w.contraction_id for w in t.wires])
    return opt_contract(*opt_einsum_args)
