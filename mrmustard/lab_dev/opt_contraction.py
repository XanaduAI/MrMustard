from __future__ import annotations
from queue import PriorityQueue
import networkx as nx
import random
from typing import Sequence, Optional
from dataclasses import dataclass


Edge = tuple[int, int]
Node = int


@dataclass
class Info:
    type: str
    neighbors: list[int]
    shape: tuple[Optional[int], ...]


class CircuitGraph:
    r"""A graph representing a circuit. If it is the first graph in a contraction path,
    initialize it with ``cost=0`` and ``solution=[]``, as we get this graph for free without
    any contractions needed.

    Args:
        data: A dict of nodes to types and neighbors.
        cost: The cost of obtaining this graph from previous contractions.
        solution: The contractions that lead to this graph.
    """

    def __init__(self, data: dict[Node, Info], cost: int, solution: list[Edge]) -> None:
        self.G = nx.MultiDiGraph()
        for node, info in data.items():
            assert info.type in ("B", "F")
            self.G.add_node(node, type=info.type, shape=info.shape)
            for neighbor in info.neighbors:
                self.G.add_edge(node, neighbor)
        self.cost = cost
        self.solution = solution

    def __lt__(self, other: CircuitGraph) -> bool:
        "this is to make the priority queue work"
        return self.cost < other.cost

    def is_leaf(self) -> bool:
        r"""Returns whether the graph has no edges left to contract"""
        return self.G.number_of_edges() == 0

    def children(self, cost_bound: int) -> list[CircuitGraph]:
        r"""Returns all the graphs obtained by contracting a single edge, for all edges.
        Only children with a cost below ``cost_bound`` are returned."""
        children = []
        for i, j, _ in self.G.edges:
            child = self.contract((i, j))
            if child.cost < cost_bound:
                children.append(child)
        return children

    def grandchildren(self, cost_bound: int) -> list[CircuitGraph]:
        grandchildren = []
        for i, j, _ in self.G.edges:
            # i, j = sorted([i, j])
            g = self.contract((i, j))
            assert j not in g.G.neighbors(i)
            if g.cost < cost_bound:
                # neighbors = [n for n in g.G.neighbors(i)]
                edges = list(g.G.edges(i))  # I want only outgoing edges
                for k in edges:
                    g2 = g.contract(k)
                    assert k[1] not in g2.G.neighbors(k[0])
                    if g2.cost < cost_bound:
                        grandchildren.append(g2)
                if not edges:
                    grandchildren.append(g)
        return grandchildren

    def contract(self, edge: tuple[int, int]) -> CircuitGraph:
        r"""Returns a copy of self with the given edge = (i,j) contracted.
        The new graph does not contain j, all edges to j are now to i
        and we update cost and solution. If i and j are nodes of different types,
        the result is of F type, otherwise the same as the types of i and j."""
        G = nx.contracted_edge(self.G, edge, self_loops=False, copy=True)
        if self.G.nodes[edge[0]]["type"] != self.G.nodes[edge[1]]["type"]:
            G.nodes[edge[0]]["type"] = "F"  # edge[0] is the new node now
        delta_cost = self.edge_cost(edge)
        cost = self.cost + delta_cost
        solution = self.solution + [(delta_cost, edge)]
        ret = CircuitGraph({}, cost, solution)
        ret.G = G
        return ret

    def edge_cost(self, edge: tuple[int, int]) -> int:
        r"""Returns the cost of contracting an edge."""
        t0 = self.G.nodes[edge[0]]["type"]
        t1 = self.G.nodes[edge[1]]["type"]
        k = self.G.number_of_edges(edge[0], edge[1])
        m = self.G.degree[edge[0]] - k
        n = self.G.degree[edge[1]] - k
        if t0 == "B" and t1 == "B":
            cost = (2 * k) ** 3  # matrix inversion
            cost += (m + n) * (m + n) * (2 * k) * (2 * k)  # matrix multiplication
            cost += (m + n) * (m + n)  # matrix addition
        if t0 == "B" and t1 == "F":
            cost = 50**m  # B -> F assuming cutoff of 50 per wire
            cost += 50 ** (m + n + k)  # inner product
        if t0 == "F" and t1 == "B":
            cost = 50**n  # B -> F assuming cutoff of 50 per wire
            cost += 50 ** (m + n + k)  # inner product
        if t0 == "F" and t1 == "F":
            cost = 50 ** (m + n + k)  # inner product
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

    debug = False
    print("1. Applying heuristics to simplify the graph...")
    graph = reduce_1BB(graph, debug=debug)
    graph = reduce_2BB(graph, debug=debug)
    graph = reduce_1FF(graph, debug=debug)
    # 1FB is not always the best thing to do but it is for the staircase GBS
    graph = reduce_1FB(graph, debug=debug)
    graph = reduce_2FF(graph, debug=debug)

    print(f"2. Getting cost upper bound by {n_init} random contractions...", end=" ")
    best_cost = 1e42
    for _ in range(n_init):
        cost, sol = random_cost(graph)
        if cost < best_cost:
            print("upper bound =", cost)
            best_cost = cost
            best_solution = sol

    print("3 Init Branch and Bound")
    graph_queue = PriorityQueue()
    graph_queue.put(graph)
    max_qsize = 1
    while not graph_queue.empty():
        graph = graph_queue.get()
        print("\033[K", end="\r")
        qsize = graph_queue.qsize()
        if qsize > max_qsize:
            max_qsize = qsize
        print(f"Queue size: {qsize}/{max_qsize}", "Current cost:", graph.cost, end="\r")
        if graph.cost > best_cost:
            continue
        if graph.is_leaf() and graph.cost < best_cost:
            best_solution = graph.solution
            best_cost = graph.cost
            graph_queue.queue = [g for g in graph_queue.queue if g.cost < best_cost]
        else:
            for child in graph.grandchildren(cost_bound=best_cost):
                if child not in graph_queue.queue:
                    graph_queue.put(child)
                else:
                    i = graph_queue.queue.index(child)
                    if child < graph_queue.queue[i]:
                        graph_queue.queue[i] = child
    print("", end="\r")
    print(f"Max queue size: {max_qsize}, best_cost: {best_cost}")
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


# Strategies for reducing the graph


def reduce_first_1BB(graph) -> tuple[CircuitGraph, Optional[Edge]]:
    r"""Reduces the first BB pair it finds where one of the nodes has degree 1 (leaf)."""
    for node in graph.G.nodes:
        neighbors = list(graph.G.neighbors(node))
        if len(neighbors) == 1 and graph.G.nodes[node]["type"] == "B":
            neighbor = neighbors[0]  # there's only one
            if graph.G.nodes[neighbor]["type"] == "B":
                return graph.contract((node, neighbor)), (node, neighbor)
    return graph, None


def reduce_1BB(graph, debug=False) -> CircuitGraph:
    r"""Reduces all BB pairs where one of the nodes has degree 1 (leaf)."""
    graph, info = reduce_first_1BB(graph)
    if debug and info:
        print("Reduced 1BB", info)
    while info:
        graph, info = reduce_first_1BB(graph)
        if debug and info:
            print("Reduced 1BB", info)
    return graph


def reduce_first_2BB(graph):
    r"""Reduces the first BB pair it finds where both nodes have degree 2."""
    for node in graph.G.nodes:
        neighbors = list(graph.G.neighbors(node))
        if len(neighbors) == 2 and graph.G.nodes[node]["type"] == "B":
            if graph.G.nodes[neighbors[0]]["type"] == "B":
                return graph.contract((node, neighbors[0])), (node, neighbors[0])
            if graph.G.nodes[neighbors[1]]["type"] == "B":
                return graph.contract((node, neighbors[1])), (node, neighbors[1])
    return graph, None


def reduce_2BB(graph, debug=False):
    r"""Reduces all BB pairs where both nodes have degree 2."""
    graph, info = reduce_first_2BB(graph)
    if debug and info:
        print("Reduced 2BB", info)
    while info:
        graph, info = reduce_first_2BB(graph)
        if debug and info:
            print("Reduced 2BB", info)
    return graph


def reduce_first_1FF(graph) -> tuple[CircuitGraph, Optional[Edge]]:
    r"""Reduces the first FF pair it finds where one of the nodes has degree 1 (leaf)."""
    for node in graph.G.nodes:
        nbrs = list(graph.G.neighbors(node))
        if len(nbrs) == 1 and graph.G.nodes[node]["type"] == "F":
            neighbor = nbrs[0]
            if graph.G.nodes[neighbor]["type"] == "F":
                return graph.contract((node, neighbor)), (node, neighbor)
    return graph, None


def reduce_1FF(graph, debug=False) -> CircuitGraph:
    r"""Reduces all FF pairs where one of the nodes has degree 1 (leaf)."""
    graph, info = reduce_first_1FF(graph)
    if debug and info:
        print("Reduced 1FF", info)
    while info:
        graph, info = reduce_first_1FF(graph)
        if debug and info:
            print("Reduced 1FF", info)
    return graph


def reduce_first_1FB(graph) -> tuple[CircuitGraph, Optional[Edge]]:
    r"""Reduces the first FB pair it finds where one of the nodes has degree 1 (leaf)."""
    for node in graph.G.nodes:
        nbrs = list(graph.G.neighbors(node))
        if len(nbrs) == 1 and graph.G.nodes[node]["type"] == "F":
            neighbor = nbrs[0]
            if graph.G.nodes[neighbor]["type"] == "B":
                return graph.contract((node, neighbor)), (node, neighbor)
    return graph, None


def reduce_1FB(graph, debug=False) -> CircuitGraph:
    r"""Reduces all FB pairs where one of the nodes has degree 1 (leaf)."""
    graph, info = reduce_first_1FB(graph)
    if debug and info:
        print("Reduced 1FB", info)
    while info:
        graph, info = reduce_first_1FB(graph)
        if debug and info:
            print("Reduced 1FB", info)
    return graph


def reduce_first_2FF(graph) -> tuple[CircuitGraph, Optional[Edge]]:
    r"""Reduces the first FF pair it finds where both nodes have degree 2."""
    for node in graph.G.nodes:
        nbrs = list(graph.G.neighbors(node))
        if len(nbrs) == 2 and graph.G.nodes[node]["type"] == "F":
            if graph.G.nodes[nbrs[0]]["type"] == "F":
                return graph.contract((node, nbrs[0])), (node, nbrs[0])
            if graph.G.nodes[nbrs[1]]["type"] == "F":
                return graph.contract((node, nbrs[1])), (node, nbrs[1])
    return graph, None


def reduce_2FF(graph, debug=False) -> CircuitGraph:
    r"""Reduces all FF pairs where both nodes have degree 2."""
    graph, info = reduce_first_2FF(graph)
    if debug and info:
        print("Reduced 2FF", info)
    while info:
        graph, info = reduce_first_2FF(graph)
        if debug and info:
            print("Reduced 2FF", info)
    return graph


def propagate_shapes(graph: CircuitGraph) -> CircuitGraph:
    r"""Propagates the known shapes where edges connect."""
