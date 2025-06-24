# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Library for optimizing paths in a quantum circuit."""

from mrmustard.lab.circuit_components import CircuitComponent
from mrmustard.path.branch_and_bound import (
    assign_costs,
    optimal_contraction,
    parse_components,
    random_solution,
)


def optimal_path(
    components: list[CircuitComponent],
    n_init: int = 100,
    with_BF_heuristic: bool = True,
    verbose: bool = True,
) -> list[tuple[int, int]]:
    """Find the optimal path for a given set of components.

    Args:
        components (list[CircuitComponent]): A list of CircuitComponent objects.
        n_init: The number of random contractions to find an initial cost upper bound.
        with_BF_heuristic: If True (default), the 1BF/1FB heuristics are included in the optimization process.
        verbose: If True (default), the progress of the optimization is shown.

    Returns:
        list[tuple[int, int]]: A list of tuples representing the optimal path.
    """
    graph = parse_components(components)
    assign_costs(graph)

    G = random_solution(graph)
    if len(G.nodes) > 1:
        raise ValueError("Circuit has disconnected components.")

    i = next(iter(G.nodes))
    if len(G.nodes[i]["component"].wires) > 0:
        raise NotImplementedError("Cannot optimize a circuit with dangling wires yet.")

    graph.optimize_fock_shapes(components, verbose=verbose)

    heuristics = (
        ("1BB", "2BB", "1BF", "1FB", "1FF", "2FF")
        if with_BF_heuristic
        else ("1BB", "2BB", "1FF", "2FF")
    )
    optimized_graph = optimal_contraction(graph, n_init, heuristics, verbose)
    return list(optimized_graph.solution)
