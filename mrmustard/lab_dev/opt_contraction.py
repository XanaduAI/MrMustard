from __future__ import annotations
from queue import PriorityQueue
import networkx as nx
import random
from typing import Optional
from dataclasses import dataclass


Index = int
Edge = tuple[Index, Index]
Node = int


@dataclass
class Info:
    type: type
    neighbors: dict[Node, list[Edge]]
    shape: tuple[int]


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
        components: dict[Node, Info],
        cost: int,
        solution: list[tuple[Node, Node]],
    ) -> None:
        self.G = nx.MultiDiGraph()
        for i, info in components.items():
            self.G.add_node(i, type=info.type, shape=info.shape)
            for j, edges in info.neighbors.items():
                for edge in edges:
                    self.G.add_edge(i, j, indices=edge, shape=info.shape[edge[0]])
        self.cost = cost
        self.solution = solution

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
        for i, j, _ in self.G.out_edges:
            child = self.contract((i, j))
            if child.cost < cost_bound:
                children.append(child)
        if not only_out:
            for i, j, _ in self.G.in_edges:
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
    print()
    print(f"2. Getting cost upper bound by {n_init} random contractions...")
    best_cost = 1e42
    for i in range(n_init):
        cost, sol = random_cost(graph)
        if cost < best_cost:
            print(f"contraction {i}: new upper bound = {cost}")
            best_cost = cost
            best_solution = sol

    print("3 Init Branch and Bound")
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
