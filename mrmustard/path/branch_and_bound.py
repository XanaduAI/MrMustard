# Copyright 2024 Xanadu Quantum Technologies Inc.

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
Branch and bound algorithm for optimal contraction of a tensor network.
"""

from __future__ import annotations

import functools
import operator
import random
from collections.abc import Generator
from copy import deepcopy
from math import factorial
from queue import PriorityQueue

import networkx as nx
import numpy as np

from mrmustard.lab.circuit_components import CircuitComponent
from mrmustard.physics.wires import Wires

Edge = tuple[int, int]


# =====================
# ====== Classes ======
# =====================


class GraphComponent:
    r"""
    A lightweight "CircuitComponent" without the actual ansatz.
    Basically a wrapper around Wires, so that it can emulate components in
    a circuit. It exposes the ansatz, wires, shape, name and cost of obtaining
    the component from previous contractions.

    Args:
        ansatz: The name of the ansatz of the component.
        wires: The wires of the component.
        shape: The fock shape of the component.
        name: The name of the component.
        cost: The cost of obtaining this component.
    """

    def __init__(self, ansatz: str, wires: Wires, shape: list[int], name: str = "", cost: int = 0):
        if None in shape:
            raise ValueError("Detected `None`s in shape. Please provide a full shape.")
        self.ansatz = ansatz
        self.wires = wires
        self.shape = list(shape)
        self.name = name
        self.cost = cost

    @classmethod
    def from_circuitcomponent(cls, c: CircuitComponent):
        r"""
        Creates a GraphComponent from a CircuitComponent.

        Args:
            c: A CircuitComponent.
        """
        return GraphComponent(
            ansatz=str(c.ansatz.__class__.__name__),
            wires=Wires(*c.wires.args),
            shape=c.auto_shape(),
            name=c.__class__.__name__,
        )

    def contraction_cost(self, other: GraphComponent) -> int:
        r"""
        Returns the computational cost in approx FLOPS for contracting this component with another
        one. Three cases are possible:

        1. If both components are in Fock ansatz the cost is the product of the values along
        both shapes, counting only once the shape of the contracted indices. E.g. a tensor with shape
        (20,30,40) contracts its 1,2 indices with the 0,1 indices of a tensor with shape (30,40,50,60).
        The cost is 20 x 30 x 40 x 50 x 60 = 72_000_000 (note 30,40 were counted only once).

        2. If the ansatze are a mix of Bargmann and Fock, we add the cost of converting the
        Bargmann to Fock, which is the product of the shape of the Bargmann component.

        3. If both are in Bargmann ansatz, the contraction can be a simple a Gaussian integral
        which scales like the cube of the number of contracted indices, i.e. ~ just 8 in the example above.

        Arguments:
            other: GraphComponent

        Returns:
            int: contraction cost in approx FLOPS

        """
        idxA, idxB = self.wires.contracted_indices(other.wires)
        m = len(idxA)  # same as len(idxB)
        nA, nB = len(self.shape) - m, len(other.shape) - m

        if self.ansatz == "PolyExpAnsatz" and other.ansatz == "PolyExpAnsatz":
            cost = (  # +1s to include vector part)
                m * m * m  # M inverse
                + (m + 1) * m * nA  # left matmul
                + (m + 1) * m * nB  # right matmul
                + (m + 1) * m  # addition
                + m * m * m  # determinant of M
            )
        else:  # otherwise need to use fock cost
            prod_A = np.prod([s for i, s in enumerate(self.shape) if i not in idxA])
            prod_B = np.prod([s for i, s in enumerate(other.shape) if i not in idxB])
            prod_contracted = np.prod(
                [min(self.shape[i], other.shape[j]) for i, j in zip(idxA, idxB)],
            )
            cost = (
                prod_A * prod_B * prod_contracted  # matmul
                + np.prod(self.shape) * (self.ansatz == "PolyExpAnsatz")  # conversion
                + np.prod(other.shape) * (other.ansatz == "PolyExpAnsatz")  # conversion
            )
        return int(cost)

    def __matmul__(self, other) -> GraphComponent:
        r"""
        Returns the contracted GraphComponent.

        Args:
            other: Another GraphComponent
        """
        new_wires, perm = self.wires @ other.wires
        idxA, idxB = self.wires.contracted_indices(other.wires)
        shape_A = [n for i, n in enumerate(self.shape) if i not in idxA]
        shape_B = [n for i, n in enumerate(other.shape) if i not in idxB]
        shape = shape_A + shape_B
        new_shape = [shape[p] for p in perm]
        return GraphComponent(
            "PolyExpAnsatz" if self.ansatz == other.ansatz == "PolyExpAnsatz" else "ArrayAnsatz",
            new_wires,
            new_shape,
            f"({self.name}@{other.name})",
            self.contraction_cost(other) + self.cost + other.cost,
        )

    def __repr__(self):
        return f"{self.name}({self.shape}, {self.wires})"


class Graph(nx.DiGraph):
    r"""
    Power pack for nx.DiGraph with additional attributes and methods.

    Args:
        solution: The sequence of edges contracted to obtain this graph.
        costs: The costs of contracting each edge in the current solution.
    """

    def __init__(self, solution: tuple[Edge, ...] = (), costs: tuple[int, ...] = ()):
        super().__init__()
        self.solution = solution
        self.costs = costs

    @property
    def cost(self) -> int:
        r"""
        Returns the total cost of the graph.
        """
        return sum(self.costs)

    def component(self, n) -> GraphComponent:
        r"""
        Returns the ``GraphComponent`` associated with a node.

        Args:
            n: The node index.
        """
        return self.nodes[n]["component"]

    def components(self) -> Generator[GraphComponent, None, None]:
        r"""
        Yields the ``GraphComponents`` associated with the nodes.
        """
        for n in self.nodes:
            yield self.component(n)

    def optimize_fock_shapes(self, components: list[CircuitComponent], verbose: bool = False):
        r"""
        Optimizes the Fock shapes of the components in this circuit.
        It starts by matching the existing connected wires and keeps the smaller shape,
        then it enforces the BSgate symmetry (conservation of photon number) to further
        reduce the shapes across the circuit.
        This operation acts in place.
        """
        if verbose:
            print("===== Optimizing Fock shapes =====")
        optimize_fock_shapes(self, 0, verbose=verbose)
        for i, c in enumerate(components):
            c.manual_shape = self.component(i).shape

    def __lt__(self, other: Graph) -> bool:
        r"""
        Compares two graphs by their cost. Used for sorting in the priority queue.

        Args:
            other: Another graph.
        """
        return self.cost < other.cost

    def __hash__(self) -> int:
        r"""
        Returns a hash of the graph.
        """
        return hash(
            tuple(self.nodes)
            + tuple(self.edges)
            + tuple(self.solution)
            + tuple(functools.reduce(operator.iadd, (c.shape for c in self.components()), [])),
        )


# =======================
# ====== Functions ======
# =======================


def optimize_fock_shapes(graph: Graph, iteration: int, verbose: bool) -> Graph:  # noqa: C901
    r"""
    Iteratively optimizes the Fock shapes of the components in the graph.

    Args:
        graph: The graph to optimize.
        iteration: The iteration number.
        verbose: Whether to print the progress.
    """
    hash_before = hash(graph)
    for A, B in graph.edges:
        wires_A = graph.nodes[A]["component"].wires
        wires_B = graph.nodes[B]["component"].wires
        idx_A, idx_B = wires_A.contracted_indices(wires_B)
        # ensure at idx_i and idx_j the shapes are the minimum
        for i, j in zip(idx_A, idx_B):
            value = min(
                graph.nodes[A]["component"].shape[i],
                graph.nodes[B]["component"].shape[j],
            )
            graph.nodes[A]["component"].shape[i] = value
            graph.nodes[B]["component"].shape[j] = value

    for component in graph.components():
        if component.name == "BSgate":
            a, b, c, d = component.shape
            if c and d:
                if not a or a > c + d:
                    a = c + d
                if not b or b > c + d:
                    b = c + d
            if a and b:
                if not c or c > a + b:
                    c = a + b
                if not d or d > a + b:
                    d = a + b
            component.shape = [a, b, c, d]

    if hash(graph) == hash_before:
        return graph

    if verbose:
        print(f"Iteration {iteration}: graph updated")

    return optimize_fock_shapes(graph, iteration + 1, verbose)


def parse_components(components: list[CircuitComponent]) -> Graph:
    r"""
    Parses a list of CircuitComponents into a Graph.

    Each node in the graph corresponds to a GraphComponent and an edge between two nodes indicates that
    the GraphComponents are connected in the circuit. Whether they are connected by one wire
    or by many, in the graph they will have a single edge between them.

    Args:
        components: A list of CircuitComponents.
    """
    validate_components(components)
    graph = Graph()
    for i, A in enumerate(components):
        comp = GraphComponent.from_circuitcomponent(A)
        wires = A.wires.copy()
        comp.wires = wires
        for j, B in enumerate(components[i + 1 :]):
            ovlp_bra, ovlp_ket = wires.overlap(B.wires)
            if ovlp_ket or ovlp_bra:
                graph.add_edge(i, i + j + 1)
                wires = Wires(
                    wires.args[0] - ovlp_bra,
                    wires.args[1],
                    wires.args[2] - ovlp_ket,
                    wires.args[3],
                )
            if not wires.output:
                break
        graph.add_node(i, component=comp)
    return graph


def validate_components(components: list[CircuitComponent]) -> None:
    r"""
    Raises an error if the components will not contract correctly.

    Args:
        components: A list of CircuitComponents.
    """
    if len(components) == 0:
        return
    w = components[0].wires
    for comp in components[1:]:
        w = (w @ comp.wires)[0]


def contract(graph: Graph, edge: Edge, debug: int = 0) -> Graph:
    r"""
    Contracts an edge in a graph and returns the contracted graph.
    Makes a copy of the original graph.

    Args:
        graph (Graph): A graph.
        edge (tuple[int, int]): An edge to contract.
        debug (int): Whether to print debug information.

    Returns:
        Graph: A new graph with the contracted edge.
    """
    new_graph = nx.contracted_edge(graph, edge, self_loops=False, copy=True)
    A = graph.nodes[edge[0]]["component"]
    B = graph.nodes[edge[1]]["component"]
    if debug > 0:
        print(f"A wires: {A.wires}, B wires: {B.wires}")
    new_graph.nodes[edge[0]]["component"] = A @ B
    new_graph.costs = (*graph.costs, graph.edges[edge]["cost"])
    new_graph.solution = (*graph.solution, edge)
    assign_costs(new_graph)
    return new_graph


def children(graph: Graph, cost_bound: int) -> set[Graph]:
    r"""
    Returns a set of graphs obtained by contracting each edge.
    Only graphs with a cost below ``cost_bound`` are returned.
    Two nodes are contracted by removing the edge between them and merging
    the two nodes into a single node. The shape of the new node
    is the union of the shapes of the two nodes without the wires that were
    contracted (this is all handled by the wires).

    Args:
        graph (Graph): A graph.
        cost_bound (int): The maximum cost of the children.

    Returns:
        set[Graph]: The set of graphs obtained by contracting each edge.
    """
    children_set = set()
    for edge in sorted(graph.out_edges, key=lambda e: graph.out_edges[e]["cost"]):
        if graph.cost + graph.edges[edge]["cost"] < cost_bound:
            children_set.add(contract(graph, edge))
    return children_set


def grandchildren(graph: Graph, cost_bound: int) -> set[Graph]:
    r"""
    A set of grandchildren constructed from each child's children.
    Only grandchildren with a cost below ``cost_bound`` are returned.
    Note that children without further children are included, so with
    a single call to this function we get all the descendants up to
    the grandchildren level including leaf nodes whether they are
    children or grandchildren.

    Args:
        graph (Graph): A graph.
        cost_bound (int): The maximum cost of the grandchildren

    Returns:
        set[Graph]: The set of grandchildren below the cost bound.
    """
    grandchildren_set = set()
    for child in children(graph, cost_bound):
        if child.number_of_edges() == 0:
            grandchildren_set.add(child)
            continue
        for grandchild in children(child, cost_bound):
            grandchildren_set.add(grandchild)
    return grandchildren_set


def assign_costs(graph: Graph, debug: int = 0) -> None:
    r"""
    Assigns to each edge in the graph the cost of contracting the two nodes it connects.

    Args:
        graph (Graph): A graph.
        debug (int): Whether to print debug information
    """
    for edge in graph.edges:
        A = graph.nodes[edge[0]]["component"]
        B = graph.nodes[edge[1]]["component"]
        graph.edges[edge]["cost"] = A.contraction_cost(B)
        if debug > 0:
            print(
                f"cost of edge {edge}: {A.ansatz}|{A.shape} x {B.ansatz}|{B.shape} = {graph.edges[edge]['cost']}",
            )


def random_solution(graph: Graph) -> Graph:
    r"""
    Returns a random solution to contract a given graph.

    Args:
        graph (Graph): The initial graph.

    Returns:
        Graph: The contracted graph
    """
    while graph.number_of_edges() > 0:
        edge = random.choice(list(graph.edges))
        graph = contract(graph, edge)
    return graph


def reduce_first(graph: Graph, code: str) -> tuple[Graph, Edge | bool]:
    r"""
    Reduces the first pair of nodes that match the pattern in the code.
    The first number and letter describe a node with that number of
    edges and that ansatz (B for Bargmann, F for Fock), and the last letter
    describes the ansatz of the second node.
    For example 1BB means we will contract the first occurrence of a node
    that has one edge (a leaf) connected to a node of ansatz B with an arbitrary
    number of edges.
    We typically use codes like 1BB, 2BB, 1FF, 2FF by default because they are
    safe, and codes like 1BF, 1FB optionally as they are not always the best choice.

    Args:
        graph: A graph.
        code: A pattern indicating the type of nodes to contract.
    """
    n, tA, tB = code
    for node in graph.nodes:
        if int(n) == graph.degree(node):
            for edge in list(graph.out_edges(node)) + list(graph.in_edges(node)):
                A = graph.nodes[edge[0]]["component"]
                B = graph.nodes[edge[1]]["component"]
                if A.ansatz[0] == tA and B.ansatz[0] == tB:
                    graph = contract(graph, edge)
                    return graph, edge
    return graph, False


def heuristic(graph: Graph, code: str, verbose: bool) -> Graph:
    r"""
    Simplifies the graph by contracting all pairs of nodes that match the given pattern.

    Args:
        graph: A graph.
        code: A pattern indicating the type of nodes to contract.
        verbose: Whether to print the progress.
    """
    edge = True
    while edge:
        graph, edge = reduce_first(graph, code)
        if edge and verbose:
            print(f"{code} edge: {edge} | tot cost: {graph.cost}")
    return graph


def optimal_contraction(  # noqa: C901
    graph: Graph,
    n_init: int,
    heuristics: tuple[str, ...],
    verbose: bool,
) -> Graph:
    r"""
    Finds the optimal path to contract a graph.

    Args:
        graph: The graph to contract.
        n_init: The number of random contractions to find an initial cost upper bound.
        heuristics: A sequence of patterns to reduce in order.
        verbose: Whether to print the progress.

    Returns:
        The optimally contracted graph with associated cost and solution
    """
    assign_costs(graph)
    if verbose:
        print("\n===== Simplify graph via heuristics =====")
    for code in heuristics:
        graph = heuristic(graph, code, verbose)
    if graph.number_of_edges() == 0:
        return graph

    if verbose:
        print(f"\n===== Branch and bound ({factorial(len(graph.nodes)):_d} paths) =====")
    best = Graph(costs=(np.inf,))  # will be replaced by first random contraction
    for _ in range(n_init):
        rand = random_solution(deepcopy(graph))
        best = rand if rand.cost < best.cost else best
    if verbose:
        print(
            f"Best cost from {n_init} random contractions: {best.cost}. Solution: {best.solution}\n",
        )

    queue = PriorityQueue()
    queue.put(graph)
    while not queue.empty():
        candidate = queue.get()
        if verbose:
            print(
                f"Queue: {queue.qsize()}/{queue.unfinished_tasks} | cost: {candidate.cost} | solution: {candidate.solution}",
                end="\x1b[1K\r",
            )

        if candidate.cost >= best.cost:
            if verbose:
                print("warning: early stop")
            return candidate  # early stopping because first in queue is already worse
        if candidate.number_of_edges() == 0:  # better solution! 🥳
            best = candidate
            queue.queue = [g for g in queue.queue if g.cost < best.cost]  # prune
        else:
            for g in grandchildren(candidate, cost_bound=best.cost):
                if g not in queue.queue:
                    queue.put(g)
    if verbose:
        print(f"\n\nFinal path: best cost = {best.cost}. Solution is {best.solution}")
    return best
