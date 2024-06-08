from __future__ import annotations
from queue import PriorityQueue
import networkx as nx
import random
from typing import Optional, Sequence
from dataclasses import dataclass
import numpy as np

Node = int
Edge = tuple[Node, Node]


@dataclass
class NodeData:
    type: str
    shape: list[int]
    indices: dict[Node, dict[int, int]]


class CircuitGraph:
    r"""A graph representing a circuit.
    It can be initialized directly from a list of CircuitComponents or by starting with
    Circuit() and then using the rightshift operator to add components.

    The cost argument is the cost of obtaining this graph from previous contractions.
    The solution argument is a list of contractions that lead to this graph.
    Therefore, a brand new unoptimized graph has cost 0 and no list of contractions.

    Args:
        data: A dict of nodes (ints) to NodeData objects.
        cost: The cost of obtaining this graph from previous contractions.
        solution: The contractions that lead to this graph.
    """

    def __init__(
        self,
        data: dict[Node, NodeData],
        cost: int = 0,
        solution: Sequence[tuple[Node, Node]] = (),
    ) -> None:
        solution = list(solution)
        self.G = nx.DiGraph()
        for i, nd in data.items():
            self.G.add_node(i, type=nd.type, shape=nd.shape)  # , indices=nd.indices)
            for j in nd.indices:
                self.G.add_edge(i, j, indices=nd.indices[j])
        self.cost = cost
        self.solution = solution

    def nodes(self, i: Node) -> dict:
        r"""Returns the data of node i."""
        return self.G.nodes[i]

    def edges(self, i: Node, j: Node) -> dict:
        r"""Returns the data of the edge between i and j."""
        return self.G.edges[i, j]

    def __lt__(self, other: CircuitGraph) -> bool:
        "this is to make the priority queue work"
        return self.cost < other.cost

    def is_leaf(self) -> bool:
        r"""Returns whether the graph has no edges left to contract"""
        return self.G.number_of_edges() == 0

    def children(self, cost_bound: int, all_edges: bool = False) -> list[CircuitGraph]:
        r"""Returns all the graphs obtained by contracting a single outgoing edge.
        Only children with a cost below ``cost_bound`` are returned. If ``all_edges=True``,
        all edges are contracted, including incoming edges.
        Note that incoming edges should never be needed though.

        Args:
            cost_bound: The maximum cost of the children.
            all_edges: Whether to consider incoming edges as well.
        """
        children = []
        for i, j in self.G.out_edges:
            child = self.contract((i, j))
            if child.cost < cost_bound:
                children.append(child)
        if all_edges:
            for i, j in self.G.in_edges:
                child = self.contract((i, j))
                if child.cost < cost_bound:
                    children.append(child)
        return children

    def __hash__(self) -> int:
        return hash(self.G)

    def grandchildren(self, cost_bound: int) -> list[CircuitGraph]:
        r"""Returns all the graphs obtained by contracting two outgoing edges.
        Only grandchildren with a cost below ``cost_bound`` are returned."""

        children = self.children(cost_bound)
        grandchildren = []
        for c in children:
            if c.is_leaf():
                grandchildren.append(c)
            else:
                grandchildren += c.children(cost_bound)
        return grandchildren

    @classmethod
    def from_attributes(cls, G: nx.DiGraph, cost, solution) -> CircuitGraph:
        r"""Creates a new CircuitGraph from the given attributes."""
        ret = CircuitGraph({}, cost, solution)
        ret.G = G
        return ret

    def contract(self, edge: tuple[int, int]) -> CircuitGraph:
        r"""Returns a copy of self with the given edge = (i,j) contracted.
        The new graph does not contain j, all edges to j are now to i
        and we update cost and solution. If i and j are nodes of different types,
        the result is of F type, otherwise the same as the types of i and j."""
        newG = nx.contracted_edge(self.G, edge, self_loops=False, copy=True)
        if self.node_type(edge[0]) != self.node_type(edge[1]):
            newG.nodes[edge[0]]["type"] = "F"  # edge[0] is the new node now
        delta_cost = self.edge_cost(edge)
        # remove i,j from the indices of all the remaining nodes: (now not needed because indices are in the edges)
        # for i in newG.nodes:
        #     newG.nodes[i]["indices"] = {
        #         j: {k: v for k, v in idx.items() if j not in §§edge[0], edge[1])}
        #         for j, idx in newG.nodes[i]["indices"].items()
        #     }

        cost = self.cost + delta_cost
        solution = self.solution + [(delta_cost, edge)]
        return self.from_attributes(newG, cost, solution)

    def node_type(self, i: int) -> str:
        return self.G.nodes[i]["type"]

    def node_shape(self, i: int) -> list[int]:
        return self.G.nodes[i]["shape"]

    def edge_cost(self, edge: tuple[Node, Node]) -> int:
        r"""Returns the cost of contracting all the wires between two nodes."""
        indices = self.edges(*edge)["indices"]
        i, j = edge
        shape_a = [s for m, s in enumerate(self.node_shape(i)) if m not in indices.keys()]
        shape_b = [s for n, s in enumerate(self.node_shape(j)) if n not in indices.values()]
        shape_k = [s for k, s in enumerate(self.node_shape(i)) if i in indices.keys()]
        print(self.node_shape(i), self.node_shape(j))
        print(shape_a, shape_k, shape_b)
        print([s for k, s in enumerate(self.node_shape(j)) if k in indices.values()])
        assert shape_k == [
            s for k, s in enumerate(self.node_shape(j)) if k in indices.values()
        ]  # shape check
        t0 = self.node_type(edge[0])
        t1 = self.node_type(edge[1])
        k = len(shape_k)
        a = len(shape_a)
        b = len(shape_b)
        if (t0, t1) == ("B", "B"):
            cost = (2 * k) ** 3  # matrix inversion
            cost += (a + b) * (a + b) * (2 * k) * (2 * k)  # matrix multiplication
            cost += (a + b) * (a + b)  # matrix addition
            print(f"BB({i},{j})", cost)
        elif (t0, t1) == ("B", "F"):
            cost = np.product(shape_a + shape_k)  # B->F
            cost += np.product(shape_a + shape_k + shape_b)  # inner product
            print(f"BF({i},{j})", cost)
        elif (t0, t1) == ("F", "B"):
            cost = np.product(shape_b + shape_k)  # B->F
            cost += np.product(shape_a + shape_k + shape_b)  # inner product
            print(f"FB({i},{j})", cost)
        elif (t0, t1) == ("F", "F"):
            cost = np.product(shape_a + shape_k + shape_b)  # inner product
            print(f"FF({i},{j})", cost)
        return cost


def random_cost(g: CircuitGraph) -> tuple[int, list]:
    r"""Returns the cost of a random contraction."""
    while not g.is_leaf():
        i = random.choice(list(g.G.nodes))
        if len(list(g.G.neighbors(i))) == 0:
            continue
        j = random.choice(list(g.G.neighbors(i)))
        g = g.contract((i, j))
    return g.cost, g.solution


def optimal_path(
    graph: CircuitGraph, n_init: int, heuristics: list[str], debug: bool
) -> tuple[int, list]:
    r"""Optimizes the contraction path for a given graph.

    Args:
        graph: The graph to contract.
        n_init: The number of random contractions to find an initial cost upper bound.

    Returns:
        The optimal contraction path given as an ordered list of edges.
    """
    for h in heuristics:
        graph = reduce_pattern(graph, h, debug=debug)
    best_cost = 1e42
    for i in range(n_init):
        cost, sol = random_cost(graph)
        if cost < best_cost:
            best_cost = cost
            best_solution = sol

    graph_queue = PriorityQueue()
    graph_queue.put(graph)
    max_qsize = 1
    while not graph_queue.empty():
        graph = graph_queue.get()
        qsize = graph_queue.qsize()
        if qsize > max_qsize:
            max_qsize = qsize
        if graph.cost > best_cost:
            continue
        if graph.is_leaf() and graph.cost < best_cost:
            best_solution = graph.solution
            best_cost = graph.cost
            graph_queue.queue = [g for g in graph_queue.queue if g.cost <= best_cost]
        else:
            for child in graph.grandchildren(cost_bound=best_cost):
                if child not in graph_queue.queue:
                    graph_queue.put(child)
                # else:
                #     i = graph_queue.queue.index(child)
                #     if child < graph_queue.queue[i]:
                #         graph_queue.queue[i] = child
    if debug:
        print(f"Max queue size: {max_qsize}, best cost: {best_cost}")
    return best_cost, best_solution


def reduce_first(graph, pattern: str) -> tuple[CircuitGraph, Optional[Edge]]:
    r"""Reduces the first pair of nodes that match the given pattern.
    We use patterns: 1BB, 2BB, 1FF, 1BF, 1FB, 2FF."""
    for node in graph.G.nodes:
        out_edges = list(graph.G.out_edges(node))
        in_edges = list(graph.G.in_edges(node))
        if (
            len(out_edges) + len(in_edges) == int(pattern[0])
            and graph.G.nodes[node]["type"] == pattern[1]
        ):
            for e in out_edges:
                if graph.G.nodes[e[1]]["type"] == pattern[2]:
                    return graph.contract((node, e[1])), (node, e[1])
            for e in in_edges:
                if graph.G.nodes[e[0]]["type"] == pattern[2]:
                    return graph.contract((e[0], node)), (e[0], node)
    return graph, None


def reduce_pattern(graph, pattern: str, debug=False) -> CircuitGraph:
    r"""Reduces all pairs of nodes that match the given pattern."""
    graph, info = reduce_first(graph, pattern)
    if debug and info:
        print(f"Reduced {pattern}", info)
    while info:
        graph, info = reduce_first(graph, pattern)
        if debug and info:
            print(f"Reduced {pattern}", info)
    return graph
