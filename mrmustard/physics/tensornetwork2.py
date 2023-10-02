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

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from mrmustard import settings
from mrmustard.lab.abstract import State
from mrmustard.physics.tensornetwork import CircuitPart, Wire
from mrmustard.math import Math
from mrmustard.physics.fock import autocutoffs

math = Math()


def connect(wire1: Wire, wire2: Wire, dim: Optional[int] = None):
    r"""Connects a wire of this CircuitPart to another Wire (of the same or another CircuitPart).
    Arguments:
        wire1: the first wire
        wire2: the second wire
        dim (optional int): set the dimension of the contraction (if None, we use autocutoff
                    for States or we use the largest of the dimension of the two wires)
    """
    wire1.connected_to = wire2
    wire2.connected_to = wire1
    # set dimension to max of the two wires if exists
    if not dim:
        dim1 = get_dimension(wire1) if isinstance(wire1.owner, State) else None
        dim2 = get_dimension(wire2) if isinstance(wire2.owner, State) else None
    dim = dim or max(dim1, dim2, key=lambda x: x or 0)
    if dim is None:
        raise ValueError("Dimension cannot be inferred from wires. Please set it manually.")
    wire1.dimension = wire2.dimension = dim
    wire2.contraction_id = wire1.contraction_id


# TODO: revisit when we have Bargmann by default
def get_dimension(wire: Wire, probability: Optional[float] = None) -> Optional[int]:
    r"""Returns the dimension of a wire (fock cutoff)
    Arguments:
        wire (Wire): the wire
    Returns:
        (int) the dimension of the wire if it is a State, None otherwise
    """
    if isinstance(wire.owner, State):
        i = wire.owner.modes.index(wire.mode)
        j = i + len(wire.owner.modes)
        cov = wire.owner.cov
        sub_cov = np.array([[cov[i, i], cov[i, j]], [cov[j, i], cov[j, j]]])
        means = wire.owner.means
        sub_means = np.array([means[i], means[j]])
        return autocutoffs(sub_cov, sub_means, probability or settings.AUTOCUTOFF_PROBABILITY)[0]


def contract(tensors: list[CircuitPart], default_dim):
    r"""Contract a list of tensors.
    Arguments:
        tensors: the tensors to contract
        default_dim: the default dimension to use for the contraction
    Returns:
        (tensor) the contracted tensor
    """
    opt_einsum_args = []
    for t in tensors:
        for w in t.wires:
            w.dimension = w.dimension or default_dim
        opt_einsum_args.append(t.fock)
        opt_einsum_args.append([w.contraction_id for w in t.wires])
    return math.einsum(*opt_einsum_args)


### DRAWING ###


def draw(
    tensors: list[CircuitPart],
    show_direction=True,
    show_modes=True,
    show_types=False,
    figsize=(6, 6),
):
    r"""
    Draws a tensor network using networkx.
    Arguments:
        tensors (sequence of CircuitPart): the tensors to draw
        show_direction (bool): whether to show the direction in the node labels (default: True)
        show_modes (bool): whether to show the modes in the node labels (default: True)
        show_types (bool): whether to show the types in the node labels (default: False)
    """
    G = nx.Graph()

    ### NODES AND EDGES ###
    # Add fake edges between the wires of the same tensor
    for tensor in tensors:
        for i, wire1 in enumerate(tensor.wires):
            for wire2 in tensor.wires[i + 1 :]:
                G.add_edge(wire1.id, wire2.id, color="grey", style="dashed")
    # Add real edges between contracted wires
    done = []
    for tensor in tensors:
        for wire in tensor.wires:
            if wire.connected_to and wire.id not in done and wire.connected_to not in done:
                G.add_edge(wire.id, wire.connected_to.id, color="red", style="solid")
                done.append(wire.id)
                done.append(wire.connected_to.id)

    ### LABELS ###
    node_labels = {}
    edge_labels = {}
    for tensor in tensors:
        for wire in tensor.wires:
            label_parts = (
                [wire.direction] * show_direction
                + [wire.type] * show_types
                + [str(wire.mode)] * show_modes
            )
            node_labels[wire.id] = "_".join(label_parts)
            if wire.connected_to:
                edge_labels[(wire.id, wire.connected_to.id)] = str(wire.dimension)

    # positions for all nodes
    pos = nx.kamada_kawai_layout(G)

    # positions for all nodes
    edge_label_pos = {k: [v[0], v[1]] for k, v in pos.items()}

    plt.figure(figsize=figsize)
    edges = G.edges(data=True)
    colors = [d["color"] for u, v, d in edges]

    # Draw the nodes and edges
    nx.draw(G, pos, edge_color=colors, labels=node_labels)
    nx.draw_networkx_edge_labels(G, edge_label_pos, edge_labels=edge_labels)

    # Draw bounding boxes around the tensors
    for tensor in tensors:
        x_min, y_min, x_max, y_max = _get_bounding_box(pos, [w.id for w in tensor.wires])
        _draw_bounding_box(x_min, y_min, x_max, y_max)
        # add tensor name on top of the bounding box
        plt.text(
            x_min + 0.03,
            y_max + 0.0,
            tensor.short_name,
            horizontalalignment="center",
            verticalalignment="bottom",
        )

    plt.title("Tensor Network")
    plt.show()


def _get_bounding_box(pos, node_ids, offset=0.03):
    node_positions = [pos[node_id] for node_id in node_ids]

    x_coordinates = [pos[0] for pos in node_positions]
    y_coordinates = [pos[1] for pos in node_positions]

    x_min, x_max = min(x_coordinates), max(x_coordinates)
    y_min, y_max = min(y_coordinates), max(y_coordinates)
    return (
        x_min - 1.6 * offset,
        y_min - offset,
        x_max + 1.6 * offset,
        y_max + offset,
    )  # NOTE: 1.6 otherwise boxes aren't square 😵‍💫??


def _draw_bounding_box(x_min, y_min, x_max, y_max):
    box = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=None, color="blue")
    plt.gca().add_patch(box)
