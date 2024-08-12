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
Create the circuit graph.
"""

from mrmustard.lab_dev.wires import Wires


def build_graph(wires: Wires):
    graph = {}
    # a dictionary to store the ``ids`` of the dangling wires
    ids_dangling_wires = {m: {"ket": None, "bra": None} for w in wires for m in w.modes}

    # populate the graph
    for w in wires:
        # if there is a dangling wire, add a contraction
        for m in w.input.ket.modes:  # ket side
            if ids_dangling_wires[m]["ket"]:
                graph[ids_dangling_wires[m]["ket"]] = w.input.ket[m].ids[0]
                ids_dangling_wires[m]["ket"] = None
        for m in w.input.bra.modes:  # bra side
            if ids_dangling_wires[m]["bra"]:
                graph[ids_dangling_wires[m]["bra"]] = w.input.bra[m].ids[0]
                ids_dangling_wires[m]["bra"] = None

        # update the dangling wires
        for m in w.output.ket.modes:  # ket side
            if w.output.ket[m].ids:
                if ids_dangling_wires[m]["ket"]:
                    raise ValueError("Dangling wires cannot be overwritten.")
                ids_dangling_wires[m]["ket"] = w.output.ket[m].ids[0]
        for m in w.output.bra.modes:  # bra side
            if w.output.bra[m].ids:
                if ids_dangling_wires[m]["bra"]:
                    raise ValueError("Dangling wires cannot be overwritten.")
                ids_dangling_wires[m]["bra"] = w.output.bra[m].ids[0]
    return graph
