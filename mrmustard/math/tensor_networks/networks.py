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

from typing import Optional
from opt_einsum import contract as opt_contract

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from .tensors import Wire, Tensor


def connect(wire1: Wire, wire2: Wire, dim: Optional[int] = None):
    r"""
    Connects two wires in a tensor network.

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
    r"""
    Contract a list of tensors.

    Args:
        tensors: The tensors to contract.
        dim: The default dimension of the contractions.

    Returns:
        The contracted tensor.
    """
    opt_einsum_args = []
    for t in tensors:
        shape = t.shape(default_dim=default_dim, out_in=True)
        opt_einsum_args.append(t.value(shape=shape))
        opt_einsum_args.append([w.contraction_id for w in t.wires])
    return opt_contract(*opt_einsum_args)


def draw(
    tensors: list[Tensor],
    layout: str = "spring_layout",
    figsize: tuple[int, int] = (10, 6),
    block: bool = True,
):
    r"""
    Draws a tensor network.

    Args:
        tensors: The tensors to draw.
        layout: The layout method. Must be one of the methods in ``nx.drawing.layout``.
        figsize: The size of the returned figure.
        block: Whether to have the figure block execution.

    Returns:
        A figure showing the tensor network.
    """
    try:
        fn_layout = getattr(nx.drawing.layout, layout)
    except AttributeError:
        msg = f"Invalid layout {layout}."
        # pylint: disable=raise-missing-from
        raise ValueError(msg)

    # initialize empty lists and dictionaries used to store metadata
    tensor_labels = {}
    mode_labels = {}
    node_size = []
    node_color = []

    # initialize three graphs--one to store nodes and edges, two to keep track of arrows
    graph = nx.Graph()
    arrows_in = nx.Graph()
    arrows_out = nx.Graph()

    for idx, tensor in enumerate(tensors):
        tensor_id = tensor.name + str(idx)
        graph.add_node(tensor_id)
        tensor_labels[tensor_id] = tensor.name
        mode_labels[tensor_id] = ""
        node_size.append(150)
        node_color.append("red")

        for wire in tensor.wires:
            wire_id = wire.contraction_id
            if wire_id not in graph.nodes:
                node_size.append(0)
                node_color.append("white")
                tensor_labels[wire_id] = ""
                mode_labels[wire_id] = wire.mode

            graph.add_node(wire_id)
            graph.add_edge(wire_id, tensor_id)
            if wire.is_input:
                arrows_in.add_edge(tensor_id, wire_id)
            else:
                arrows_out.add_edge(tensor_id, wire_id)

    pos = fn_layout(graph)
    pos_labels = {k: v + np.array([0.0, 0.05]) for k, v in pos.items()}

    fig = plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(
        graph,
        pos,
        edgecolors="gray",
        alpha=0.9,
        node_size=node_size,
        node_color=node_color,
    )
    nx.draw_networkx_edges(graph, pos, edge_color="lightgreen", width=4, alpha=0.6)
    nx.draw_networkx_edges(
        arrows_in,
        pos,
        edge_color="darkgreen",
        width=0.5,
        arrows=True,
        arrowsize=10,
        arrowstyle="<|-",
    )
    nx.draw_networkx_edges(
        arrows_out,
        pos,
        edge_color="darkgreen",
        width=0.5,
        arrows=True,
        arrowsize=10,
        arrowstyle="-|>",
    )
    nx.draw_networkx_labels(
        graph,
        pos=pos_labels,
        labels=tensor_labels,
        font_size=12,
        font_color="black",
        font_family="serif",
    )
    nx.draw_networkx_labels(
        graph,
        pos=pos_labels,
        labels=mode_labels,
        font_size=12,
        font_color="black",
        font_family="FreeMono",
    )

    plt.title("Mr Mustard Network")
    plt.show(block=block)
    return fig
