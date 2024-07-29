# Copyright 2024 Xanadu Quantum Technologies Inc.
from __future__ import annotations
from mrmustard.lab_dev.wires import Wires
from mrmustard.lab_dev.circuit_components import CircuitComponent
import networkx as nx
import numpy as np
from queue import PriorityQueue
import random
from math import factorial
from typing import Generator

Edge = tuple[int, int]


class Component:
    r"""A lightweight "CircuitComponent" without the actual representation.
    Basically a wrapper around Wires, so that it can emulate components in
    a circuit. It exposes the repr, wires, shape, name and order (in the circuit).

    TODO: this could be replaced by a simple CircuitComponent without the representation.
    """

    def __init__(self, repr: str, wires: Wires, shape: list[int], name: str = ""):
        if None in shape:
            raise ValueError("Detected `None`s in shape. Please provide a full shape.")
        self.repr = repr
        self.wires = wires
        self.shape = list(shape)
        self.name = name

    @classmethod
    def from_circuitcomponent(cls, c: CircuitComponent):
        return Component(
            repr=str(c.representation.__class__.__name__),
            wires=Wires(*c.wires.args),
            shape=c.auto_shape(),
            name=c.__class__.__name__,
        )

    def copy(self) -> Component:
        return Component(self.repr, self.wires, self.shape, self.name)

    def contraction_cost(self, other: Component) -> int:
        r"""
        Returns the computational cost in approx FLOPS for contracting this component with another
        one. Three cases are possible:

        1. If both components are in Fock representation the cost is the product of the values along
        both shapes, counting only once the shape of the contracted indices. E.g. a tensor with shape
        (20,30,40) contracts its 1,2 indices with the 0,1 indices of a tensor with shape (30,40,50,60).
        The cost is 20 x 30 x 40 x 50 x 60 = 72_000_000 (note 30,40 were counted only once).

        2. If the representations are a mix of Bargmann and Fock, we add the cost of converting the
        Bargmann to Fock, which is the product of the shape of the Bargmann component.

        3. If both are in Bargmann representation, the contraction can be a simple a Gaussian integral
        which scales like the cube of the number of contracted indices, i.e. ~ just 8 in the example above.

        Arguments:
            other: Component

        Returns:
            int: contraction cost in FLOPS

        TODO: this will need to be updated once we have the poly x exp ansatz.
        TODO: be more precise on costs (profile properly? use wall time?)
        """
        idxA, idxB = self.wires.contracted_indices(other.wires)
        m = len(idxA)  # same as len(idxB)
        nA, nB = len(self.shape) - m, len(other.shape) - m

        if self.repr == "Bargmann" and other.repr == "Bargmann":
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
                [min(self.shape[i], other.shape[j]) for i, j in zip(idxA, idxB)]
            )
            cost = (
                prod_A * prod_B * prod_contracted  # matmul
                + np.prod(self.shape) * (self.repr == "Bargmann")  # conversion
                + np.prod(other.shape) * (other.repr == "Bargmann")  # conversion
            )
        return int(cost)

    def __matmul__(self, other) -> Component:
        """Returns the contracted Component."""
        new_wires, perm = self.wires @ other.wires
        idxA, idxB = self.wires.contracted_indices(other.wires)
        shape_A = [n for i, n in enumerate(self.shape) if i not in idxA]
        shape_B = [n for i, n in enumerate(other.shape) if i not in idxB]
        shape = shape_A + shape_B
        new_shape = [shape[p] for p in perm]
        new_component = Component(
            "Bargmann" if self.repr == other.repr == "Bargmann" else "Fock",
            new_wires,
            new_shape,
            f"({self.name}@{other.name})",
        )
        return new_component

    def __repr__(self):
        return f"{self.name}({self.shape}, {self.wires})"


class Graph(nx.DiGraph):
    """Power pack for nx.DiGraph with additional attributes and methods."""

    def __init__(self, solution: tuple[Edge, ...] = (), costs: tuple[int, ...] = ()):
        super().__init__()
        self.solution = solution
        self.costs = costs

    @property
    def cost(self) -> int:
        return sum(self.costs)

    def __lt__(self, other: Graph) -> bool:
        return self.cost < other.cost

    def __hash__(self) -> int:
        return hash(
            tuple(self.nodes)
            + tuple(self.edges)
            + tuple(self.solution)  # do we want this?
            + tuple(sum((c.shape for c in self.components()), start=[]))
        )

    def component(self, n) -> Component:
        return self.nodes[n]["component"]

    def components(self) -> Generator[Component, None, None]:
        for n in self.nodes:
            yield self.component(n)


def optimize_fock_shapes(graph: Graph, iter: int) -> Graph:
    h = hash(graph)
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

    if h != hash(graph):
        print(f"Iteration {iter}: graph updated")
        graph = optimize_fock_shapes(graph, iter + 1)
    return graph


def parse(components: list[CircuitComponent]) -> Graph:
    """Parses a list of CircuitComponents into a Graph.

    Each node in the graph corresponds to a Component and an edge between two nodes indicates that
    the Components are connected in the circuit level. Whether they are connected by one wire
    or by many, in the graph they will have a single edge between them.

    Args:
        components: A list of CircuitComponents.
    """
    graph = Graph()
    for i, A in enumerate(components):
        comp = Component.from_circuitcomponent(A)
        graph.add_node(i, component=comp.copy())
        for j, B in enumerate(components[i + 1 :]):
            overlap_ket = comp.wires.output.ket.modes & B.wires.input.ket.modes
            overlap_bra = comp.wires.output.bra.modes & B.wires.input.bra.modes
            if overlap_ket or overlap_bra:
                graph.add_edge(i, i + j + 1)
                comp.wires = Wires(
                    comp.wires.args[0] - overlap_bra,  # output bra
                    comp.wires.args[1],
                    comp.wires.args[2] - overlap_ket,  # output ket
                    comp.wires.args[3],
                )
                if not comp.wires.output:
                    break
    return graph


def contract(graph: Graph, edge: Edge, debug: int = 0) -> Graph:
    """Contracts an edge in a graph and returns the contracted graph.
    Makes a copy of the original graph.

    Args:
        graph (Graph): A graph.
        edge (tuple[int, int]): An edge to contract.

    Returns:
        Graph: A new graph with the contracted edge.
    """
    new_graph = nx.contracted_edge(graph, edge, self_loops=False, copy=True)
    A = graph.nodes[edge[0]]["component"]
    B = graph.nodes[edge[1]]["component"]
    if debug > 0:
        print(f"A wires: {A.wires}, B wires: {B.wires}")
    new_graph.nodes[edge[0]]["component"] = A @ B
    new_graph.costs = graph.costs + (graph.edges[edge]["cost"],)
    new_graph.solution = graph.solution + (edge,)
    assign_costs(new_graph)
    return new_graph


def children(graph: Graph, cost_bound: int) -> set[Graph]:
    """Returns a set of graphs obtained by contracting each edge.
    Only graphs with a cost below ``cost_bound`` are returned.
    Two nodes are contracted by removing the edge between them and merging
    the two nodes into a single node. The shape of the new node
    is the union of the shapes of the two nodes without the wires that were
    contracted (this is all handled by the wires).

    Args:
        graph (Graph): A graph.
        cost_bound (int): The maximum cost of the children.

    Returns:
        set[Graph]: The set of graphs.obtained by contracting each edge.
    """
    children = set()
    for edge in sorted(graph.out_edges, key=lambda e: graph.out_edges[e]["cost"]):
        if graph.cost + graph.edges[edge]["cost"] < cost_bound:
            children.add(contract(graph, edge))
    return children


def grandchildren(graph: Graph, cost_bound: int) -> set[Graph]:
    """
    A set of grandchildren constructed from each child's children.
    Only grandchildren with a cost below ``cost_bound`` are returned.
    Note that children without further children are included, so with
    a single call to this function we get all the descendants up to
    the grandchildren level.

    Args:
        graph (Graph): A graph.
        cost_bound (int): The maximum cost of the grandchildren

    Returns:
        set[Graph]: The set of grandchildren below the cost bound.
    """
    grandchildren = set()
    for child in children(graph, cost_bound):
        if child.number_of_edges() == 0:
            grandchildren.add(child)
            continue
        for grandchild in children(child, cost_bound):
            grandchildren.add(grandchild)
    return grandchildren


def assign_costs(graph: Graph, debug: int = 0) -> None:
    """Assigns to each edge in the graph the cost of contracting the two nodes it connects."""
    for edge in graph.edges:
        A = graph.nodes[edge[0]]["component"]
        B = graph.nodes[edge[1]]["component"]
        graph.edges[edge]["cost"] = A.contraction_cost(B)
        if debug > 0:
            print(
                f"cost of edge {edge}: {A.repr}|{A.shape} x {B.repr}|{B.shape} = {graph.edges[edge]['cost']}"
            )


def random_solution(graph: Graph) -> Graph:
    r"""Returns the cost and solution of a random contraction."""
    while graph.number_of_edges() > 0:
        edge = random.choice(list(graph.edges))
        graph = contract(graph, edge)
    return graph


def reduce_first(graph: Graph, code: str) -> tuple[Graph, Edge | bool]:
    r"""Reduces the first pair of nodes that match the pattern in the code.
    The first number and letter describe a node with that number of
    edges and that repr (B for Bargmann, F for Fock), and the last letter
    describes the repr of the second node.
    For example 1BB means we will contract the first occurrence of a node
    that has one edge (a leaf) connected to a node of repr B with an arbitrary
    number of edges.
    We typically use codes like 1BB, 2BB, 1FF, 2FF by default because they are
    safe, and codes like 1BF, 1FB optionally as they are not always the best choice.
    """
    n, tA, tB = code
    for node in graph.nodes:
        if int(n) == graph.degree(node):
            for edge in list(graph.edges(node)) + list(graph.in_edges(node)):
                A = graph.nodes[edge[0]]["component"]
                B = graph.nodes[edge[1]]["component"]
                if A.repr[0] == tA and B.repr[0] == tB:
                    graph = contract(graph, edge)
                    return graph, edge
    return graph, False


def heuristic(graph: Graph, code: str) -> Graph:
    r"""Simplifies the graph by contracting all pairs of nodes that match the given pattern."""
    edge = True
    while edge:
        graph, edge = reduce_first(graph, code)
        if edge:
            print(f"{code} edge: {edge} | tot cost: {graph.cost}")
    return graph


def optimal_contraction(
    graph: Graph,
    n_init: int,
    heuristics: tuple[str, ...],
) -> Graph:
    r"""Finds the optimal path to contract a graph.

    Args:
        graph: The graph to contract.
        n_init: The number of random contractions to find an initial cost upper bound.
        heuristics: A sequence of patterns to reduce in order.
        debug: Whether to print debug information.

    Returns:
        The optimally contracted graph with associated cost and solution
    """
    assign_costs(graph)
    print("\n===== Simplify graph via heuristics =====")
    for code in heuristics:
        graph = heuristic(graph, code)
    if graph.number_of_edges() == 0:
        return graph

    print(f"\n===== Branch and bound ({factorial(len(graph.nodes)):_d} paths) =====")
    best = Graph(costs=(np.inf,))  # will be replaced by first random contraction
    for _ in range(n_init):
        rand = random_solution(graph.copy())
        best = rand if rand.cost < best.cost else best
    print(f"Best cost from {n_init} random contractions: {best.cost}\n")

    queue = PriorityQueue()
    queue.put(graph)
    while not queue.empty():
        candidate = queue.get()
        print(
            f"Queue: {queue.qsize()}/{queue.unfinished_tasks} | cost: {candidate.cost} | solution: {candidate.solution}",
            end="\x1b[1K\r",
        )

        if candidate.cost >= best.cost:
            print("warning: early stop")
            return  # early stopping because first in queue is worse
        elif (
            candidate.number_of_edges() == 0
        ):  # better solution! 🥳r_of_edges() == 0:  # better solution! 🥳
            best = candidate
            queue.queue = [g for g in queue.queue if g.cost < best.cost]  # prune
        else:
            for g in grandchildren(candidate, cost_bound=best.cost):
                if g not in queue.queue:  # superfluous check?
                    queue.put(g)
    return best
