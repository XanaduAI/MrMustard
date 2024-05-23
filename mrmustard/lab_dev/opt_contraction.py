from __future__ import annotations
from queue import PriorityQueue
import networkx as nx
import random
from typing import Optional
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
    r"""A graph representing a circuit. If it is the first graph in a contraction path,
    initialize it with ``cost=0`` and ``solution=[]``, as we get this graph for free without
    any contractions needed.

    Args:
        data: A dict of nodes to types and neighbors.
        cost: The cost of obtaining this graph from previous contractions.
        solution: The contractions that lead to this graph.
    """

    def __init__(
        self,
        data: dict[Node, NodeData],
        cost: int,
        solution: list[tuple[Node, Node]],
    ) -> None:
        self.G = nx.DiGraph()
        for i, nd in data.items():
            self.G.add_node(i, type=nd.type, shape=nd.shape, indices=nd.indices)
            for j in nd.indices:
                self.G.add_edge(i, j)
        self.cost = cost
        self.solution = solution

    def nodes(self, i: Node) -> NodeData:
        r"""Returns the data of node i."""
        return self.G.nodes[i]

    def __lt__(self, other: CircuitGraph) -> bool:
        "this is to make the priority queue work"
        return self.cost < other.cost

    def is_leaf(self) -> bool:
        r"""Returns whether the graph has no edges left to contract"""
        return self.G.number_of_edges() == 0

    def children(self, cost_bound: int, only_out: bool = True) -> list[CircuitGraph]:
        r"""Returns all the graphs obtained by contracting a single outgoing edge.
        Only children with a cost below ``cost_bound`` are returned. If ``only_out=False``,
        all edges are contracted.

        Args:
            cost_bound: The maximum cost of the children.
            only_out: Whether to contract only outgoing edges (default) or all edges.
        """
        children = []
        for i, j in self.G.out_edges:
            child = self.contract((i, j))
            if child.cost < cost_bound:
                children.append(child)
        if not only_out:
            for i, j in self.G.in_edges:
                child = self.contract((i, j))
                if child.cost < cost_bound:
                    children.append(child)
        return children

    def __hash__(self) -> int:
        return hash(self.G)

    def grandchildren(self, cost_bound: int) -> list[CircuitGraph]:
        children = self.children(cost_bound, only_out=True)
        grandchildren = []
        for c in children:
            if c.is_leaf():
                grandchildren.append(c)
            else:
                grandchildren += c.children(cost_bound, only_out=False)
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
        if self.G.nodes[edge[0]]["type"] != self.G.nodes[edge[1]]["type"]:
            newG.nodes[edge[0]]["type"] = "F"  # edge[0] is the new node now
        delta_cost = self.edge_cost(edge)
        # remove i,j from the indices of all the nodes:
        for i in newG.nodes:
            newG.nodes[i]["indices"] = {
                j: {k: v for k, v in idx.items() if j not in (edge[0], edge[1])}
                for j, idx in newG.nodes[i]["indices"].items()
            }

        cost = self.cost + delta_cost
        solution = self.solution + [(delta_cost, edge)]
        return self.from_attributes(newG, cost, solution)

    def node_type(self, i: int) -> str:
        return self.G.nodes[i]["type"]

    def edge_cost(self, edge: tuple[Node, Node]) -> int:
        r"""Returns the cost of contracting the edges between two nodes."""
        # no, we need to compute shapes lazily.
        idx = self.nodes(edge[0])["indices"]
        i, j = edge
        iA = [k for k, v in idx[i + j + 1].items()] if i + j + 1 in idx else []
        iB = [v for k, v in idx[i + j + 1].items()] if i + j + 1 in idx else []
        shape_k = [s for i, s in enumerate(self.nodes(i)["shape"]) if i in iA]
        shape_a = [s for i, s in enumerate(self.nodes(i)["shape"]) if i not in iA]
        shape_b = [s for i, s in enumerate(self.nodes(j)["shape"]) if i not in iB]
        t0 = self.node_type(edge[0])
        t1 = self.node_type(edge[1])
        k = len(shape_k)
        a = len(shape_a)
        b = len(shape_b)
        if (t0, t1) == ("B", "B"):
            cost = (2 * k) ** 3  # matrix inversion
            cost += (a + b) * (a + b) * (2 * k) * (2 * k)  # matrix multiplication
            cost += (a + b) * (a + b)  # matrix addition
            # print(f"BB({i},{j})", cost)
        elif (t0, t1) == ("B", "F"):
            cost = np.product(shape_a + shape_k)  # B->F
            cost += np.product(shape_a + shape_k + shape_b)  # inner product
            # print(f"BF({i},{j})", cost)
        elif (t0, t1) == ("F", "B"):
            cost = np.product(shape_b + shape_k)  # B->F
            cost += np.product(shape_a + shape_k + shape_b)  # inner product
            # print(f"FB({i},{j})", cost)
        elif (t0, t1) == ("F", "F"):
            cost = np.product(shape_a + shape_k + shape_b)  # inner product
            # print(f"FF({i},{j})", cost)
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


def optimal_path(graph: CircuitGraph, n_init: int = 100) -> tuple[int, list]:
    r"""Optimizes the contraction path for a given graph.

    Args:
        graph: The graph to contract.
        n_init: The number of random contractions to find an initial cost upper bound.

    Returns:
        The optimal contraction path given as an ordered list of edges.
    """
    print("-" * 50)
    print("Optimal contraction path algorithm")
    print("-" * 50)
    print()

    debug = True
    print("1. Applying heuristics to simplify the graph...")
    graph = reduce_pattern(graph, "1BB", debug=debug)
    graph = reduce_pattern(graph, "2BB", debug=debug)
    graph = reduce_pattern(graph, "1BF", debug=debug)
    graph = reduce_pattern(graph, "1FF", debug=debug)
    graph = reduce_pattern(
        graph, "1FB", debug=debug
    )  # not always right. Good for staircase.
    # graph = reduce_pattern(graph, "2FF", debug=debug)  # not always right.
    print(f"Edges remaining: {graph.G.number_of_edges()}")
    print(f"\n2. Getting cost upper bound by {n_init} random contractions...")
    best_cost = 1e42
    for i in range(n_init):
        cost, sol = random_cost(graph)
        if cost < best_cost:
            print(f"contraction {i}: new upper bound = {cost}")
            best_cost = cost
            best_solution = sol

    print("\n3. Init Branch and Bound")
    graph_queue = PriorityQueue()
    graph_queue.put(graph)
    max_qsize = 1
    while not graph_queue.empty():
        graph = graph_queue.get()
        qsize = graph_queue.qsize()
        if qsize > max_qsize:
            max_qsize = qsize
        print(" " * 80, end="\r")
        print(
            f"Queue size: {qsize}/{max_qsize}",
            "Current cost:",
            graph.cost,
            "Contractions left:",
            graph.G.number_of_edges(),
            end="\r",
        )
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
    print()
    print(" " * 80, end="\r")
    print(f"Max queue size: {max_qsize}, best cost: {best_cost}", end="\r")
    return best_cost, best_solution


#  TODO: move this function to the tests files
def staircase(m: int):
    assert m > 1, "at least 2 mode staircase"
    data = {
        0: Info("B", (2,)),
        1: Info("B", (2,)),
        2: Info("B", (3,) + ((5,) if m > 2 else ())),
        3: Info("F", ()),
    }
    for i in range(m - 2):
        data[3 * i + 4] = Info("B", (3 * i + 5,))
        data[3 * i + 5] = Info("B", (3 * i + 6, 3 * i + 8))
        data[3 * i + 6] = Info("F", ())
    data[3 * (m - 1) + 2] = Info("F", ())
    return data


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
