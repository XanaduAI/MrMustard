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

"""
This module contains functions for handling tensor networks in MrMustard
"""

from typing import Optional

import numpy as np

from mrmustard import settings
from mrmustard.lab.abstract import State
from mrmustard.lab.abstract.circuitpart import Wire
from mrmustard.math import Math
from mrmustard.physics.fock import autocutoffs

math = Math()


def connect(wire1: Wire, wire2: Wire, dim: Optional[int] = None):
    r"""Connects a wire of this CircuitPart to another Wire (of the same or another CircuitPart).
    Arguments:
        wire1: the first wire
        wire2: the second wire
        dim (int): set the dimension of the contraction (if None, it's automatically computed
                    for States or propagated for other CircuitParts)
    """
    wire1.connected_to = wire2
    wire2.connected_to = wire1
    # set dimension to max of the two wires if exists
    if not dim:
        if isinstance(wire1.owner, State):
            wire1.dimension = get_dimension(wire1)
        if isinstance(wire2.owner, State):
            wire2.dimension = get_dimension(wire2)
    dim = dim or max(wire1.dimension, wire2.dimension, key=lambda x: x or 0)
    wire1.dimension = wire2.dimension = dim
    # set the same contraction_id for both wires
    wire2.contraction_id = wire1.contraction_id


# TODO: revisit when we have Bargmann by default
def get_dimension(wire: Wire, probability: Optional[float] = None) -> Optional[int]:
    r"""Returns the dimension of a wire (fock cutoff)
    Arguments:
        wire (Wire): the wire
    Returns:
        (int) the dimension of the wire if it is a State, None otherwisee
    """
    if isinstance(wire.owner, State):
        i = wire.owner.modes.index(wire.mode)
        j = i + len(wire.owner.modes)
        cov = wire.owner.cov
        sub_cov = np.array([[cov[i, i], cov[i, j]], [cov[j, i], cov[j, j]]])
        means = wire.owner.means
        sub_means = np.array([means[i], means[j]])
        return autocutoffs(sub_cov, sub_means, probability or settings.AUTOCUTOFF_PROBABILITY)[0]


def contract(tensors: list, default_dim: Optional[int] = None):
    r"""Contract a list of tensors.
    Arguments:
        tensors: the tensors to contract
        default_dim (optional int): the dimension to use for the wires that don't have one
    Returns:
        (tensor) the contracted tensor
    """
    opt_einsum_args = []
    for t in tensors:
        for w in t.wires:
            w.dimension = w.dimension or default_dim or settings.TN_BOND_DIM
        opt_einsum_args.append(t.fock)
        opt_einsum_args.append([w.contraction_id for w in t.wires])
    return math.einsum(*opt_einsum_args)
