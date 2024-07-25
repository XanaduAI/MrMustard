# Copyright 2024 Xanadu Quantum Technologies Inc.
from __future__ import annotations
from mrmustard.lab_dev import CircuitComponent, Wires
import networkx as nx
import numpy as np
from queue import PriorityQueue
import random

Edge = tuple[int, int]


class Component:
    r"""A lightweight counterpart of a CircuitComponent without the actual representation.
    It's used in the computation of the optimal path. Each node in the graph has a Component
    of the corresponding CircuitComponent. The Component has wires and shape, but no representation.

    TODO: if CircuitComponent could function without a representation we might not need Component.
    """

    def __init__(self, type: str, wires: Wires, shape: list[int], name: str = ""):
        if None in shape:
            raise ValueError("Detected `None`s in shape. Please provide a full shape.")
        self.type = type
        self.wires = wires
        self.shape = shape
        self.name = name

    @classmethod
    def from_component(cls, c):
        return Component(
            str(c.representation.__class__.__name__),
            Wires(*c.wires.args),
            c.manual_shape,
            c.__class__.__name__,
        )

    def contraction_cost(self, other: Component) -> int:
        r"""
        Returns the computational cost of contracting this component with another one.
        The cost depends on the representation. If both components are in Fock representation,
        the cost is the product of the values along both shapes, counting only once the values
        corresponding to the contracted indices. E.g. consider a tensor with shape (2,3,4) where
        indices 1,2 are contracted with indices 0,1 of a tensor of shape (3,4,5,6).
        This contraction has complexity 2 x 3 x 4 x 5 x 6 = 720 (note 3,4 were counted only once).
        If one of the two is in Bargmann, we add the cost of converting it to Fock.
        If both are in Bargmann representation, the contraction is much cheaper because it's
        the cost of a gaussian integral, which scales like the cube of the number of contracted wires.

        Arguments:
            other: Component

        Returns:
            int: contraction cost

        TODO: this will need to be updated once we have the poly x exp ansatz.
        TODO: be more precise on costs (profile properly)
        """
        idxA, idxB = self.wires.contracted_indices(other.wires)
        m = len(idxA)  # same as len(idxB)
        nA, nB = len(self.shape) - m, len(other.shape) - m

        if self.type == "Bargmann" and other.type == "Bargmann":
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
                prod_A * prod_B * prod_contracted
                + np.prod(self.shape) * (self.type == "Bargmann")  # conversion
                + np.prod(other.shape) * (other.type == "Bargmann")  # conversion
            )
        return int(cost)

    def __matmul__(self, other) -> Component:
        """computes the contracted Component"""
        new_wires, perm = self.wires @ other.wires
        idxA, idxB = self.wires.contracted_indices(other.wires)
        shape_A = [n for i, n in enumerate(self.shape) if i not in idxA]
        shape_B = [n for i, n in enumerate(other.shape) if i not in idxB]
        shape = shape_A + shape_B
        new_shape = [shape[p] for p in perm]
        new_component = Component(
            "Bargmann" if self.type == other.type == "Bargmann" else "Fock",
            new_wires,
            new_shape,
        )
        return new_component

    def __repr__(self):
        return f"Component({self.name}, {self.shape}, {self.wires})"


class Graph(nx.DiGraph):
    """Giving the nx.Graph a cost and a solution attribute and making it sortable."""

    def __init__(self, solution: tuple[Edge, ...] = (), costs: tuple[int, ...] = ()):
        super().__init__()
        self.solution = solution
        self.costs = costs

    # def custom_copy(self) -> Graph:
    #     g = Graph(self.solution, self.costs)
    #     g.add_nodes_from(self.nodes(data=True))
    #     g.add_edges_from(self.edges(data=True))
    #     return g

    @property
    def cost(self) -> int:
        return sum(self.costs)

    def __lt__(self, other: Graph) -> bool:
        return self.cost < other.cost

    def __hash__(self) -> int:
        return hash(tuple(self.nodes) + tuple(self.edges) + (self.cost,) + tuple(self.solution))


def parse(components: list[CircuitComponent], debug: int = 0) -> Graph:
    """Parses a list of CircuitComponents into a graph.
    Warning: It assumes that there are no missing adjoints and such.
    This should be ensured before calling this function.

    Each node in the graph corresponds to a CircuitComponent and
    an edge between two nodes indicates that the two CircuitComponents
    are directly connected at the circuit level. Whether they are connected by one wire
    or by many, in the graph they will have a single edge between them.

    Each node has a "component" attribute that is a Component.
    Component is a lightweight version of CircuitComponent that has wires
    and can simulate the contraction of two components, but without the actual representation.

    Args:
        components (list[CircuitComponent]): A list of CircuitComponents.
    """
    graph = Graph()
    for i, A in enumerate(components):
        GC = Component.from_component(A)
        if debug > 0:
            print(f"\nAdding node {i}: {GC}")
        graph.add_node(i, component=Component.from_component(A))
        for j, B in enumerate(components[i + 1 :]):
            overlap_ket = GC.wires.output.ket.modes & B.wires.input.ket.modes
            overlap_bra = GC.wires.output.bra.modes & B.wires.input.bra.modes
            if overlap_ket or overlap_bra:
                graph.add_edge(i, i + j + 1)  # edge data goes in here
                if debug > 0:
                    print(f"Adding edge {i} -> {i+j+1} between {GC} and {B}")
                if debug > 1:
                    print(f"Overlap bra: {overlap_bra}, overlap ket: {overlap_ket}")
                GC.wires = Wires(
                    GC.wires.args[0] - overlap_bra,
                    GC.wires.args[1],  # input bra
                    GC.wires.args[2] - overlap_ket,
                    GC.wires.args[3],  # input ket
                )
                if debug > 1:
                    print(f"New wires: {GC.wires}")
                    print("GC output: ", GC.wires.output)
                if not GC.wires.output:
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
            children.add(contract(graph, edge))  # duplicates are automatically removed
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
        set[Graph]: The set of grandchildren
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
                f"cost of edge {edge}: {A.type}|{A.shape} x {B.type}|{B.shape} = {graph.edges[edge]['cost']}"
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
    edges and that type (B for Bargmann, F for Fock), and the last letter
    describes the type of the second node.
    For example 1BB means we will contract the first occurrence of a node
    that has one edge (a leaf) connected to a node of type B with an arbitrary
    number of edges.
    We typically use codes like 1BB, 2BB, 1FF, 2FF by default because they are
    safe, and codes like 1BF, 1FB optionally as they are not always the best choice.
    """
    n, tA, tB = code
    for node in graph.nodes:
        if int(n) == graph.degree(node):
            for edge in graph.out_edges(node):
                A = graph.nodes[edge[0]]["component"]
                B = graph.nodes[edge[1]]["component"]
                if A.type[0] == tA and B.type[0] == tB:
                    graph = contract(graph, edge)
                    return graph, edge
    return graph, False


def heuristic(graph: Graph, code: str, debug: int = 0) -> Graph:
    r"""Simplifies the graph by contracting all pairs of nodes that match the given pattern."""
    edge = True
    while edge:
        graph, edge = reduce_first(graph, code)
        if debug > 0 and edge:
            print(
                f"{code} edge found: {edge} | tot cost: {graph.cost} | solution: {graph.solution}"
            )
    return graph


def optimal_contraction(
    graph: Graph,
    n_init: int = 1,
    heuristics: tuple[str, ...] = ("1BB",),
    debug: int = 0,
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
    if debug > 0:
        print("===== Simplify graph via heuristics =====")
    for code in heuristics:
        graph = heuristic(graph, code, debug)

    if debug > 0:
        print(f"\n===== {n_init} Random contractions =====")
    best = Graph(costs=(np.inf,))  # will be replaced by first random contraction
    for _ in range(n_init):
        rand = random_solution(graph.copy())
        best = rand if rand.cost < best.cost else best
    if debug > 0:
        print(f"Best cost found: {best.cost}\n")

    print("===== Branch and bound =====")
    queue = PriorityQueue()
    queue.put(graph)
    while not queue.empty():
        candidate = queue.get()
        print(
            f"Queue: {queue.qsize()}/{queue.unfinished_tasks} | cost: {candidate.cost} | solution: {candidate.solution}",
            end="\x1b\r",
        )
        if candidate.cost >= best.cost:
            return best
        elif candidate.number_of_edges() == 0:  # better solution! 🥳
            best = candidate
            queue.queue = [g for g in queue.queue if g.cost <= best.cost]
        else:
            for g in grandchildren(candidate, best.cost):
                if g not in queue.queue:
                    queue.put(g)
    print("\n")
    return best
