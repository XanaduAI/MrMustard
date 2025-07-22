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

"""Tests for the ``branch_and_bound`` library."""

import pytest

from mrmustard import settings
from mrmustard.lab import BSgate, Coherent, Dgate, Number, Sgate, SqueezedVacuum, Vacuum
from mrmustard.path import branch_and_bound as bb


def test_graph_children_and_grandchildren():
    """Test that the children function returns the correct graphs"""

    graph = bb.parse_components([Number(0, n=15), Sgate(0, r=1.0), Dgate(0, x=1.0)])
    bb.assign_costs(graph)
    children_set = bb.children(graph, int(1e20))
    for child in children_set:
        assert isinstance(child, bb.Graph)
        assert len(child.nodes) == 2

    grandchildren_set = bb.grandchildren(graph, int(1e20))
    for grandchild in grandchildren_set:
        assert isinstance(grandchild, bb.Graph)
        assert len(grandchild.nodes) == 1


def test_propagate_shapes():
    """Test that the shapes are propagated correctly."""
    MAX = settings.AUTOSHAPE_MAX
    with settings(AUTOSHAPE_PROBABILITY=0.999):
        circ = [Coherent(0, x=1.0), Dgate(0, 0.1)]
        graph = bb.parse_components(circ)
        assert [op.auto_shape() for op in circ] == [(5,), (MAX, MAX)]
        graph.optimize_fock_shapes(circ, verbose=False)
        assert [op.auto_shape() for op in circ] == [(5,), (MAX, 5)]

        circ = [SqueezedVacuum(0, r=0.5), SqueezedVacuum(1, r=-0.5), BSgate((0, 1), 0.9)]
        graph = bb.parse_components(circ)
        assert [op.auto_shape() for op in circ] == [(6,), (6,), (MAX, MAX, MAX, MAX)]
        graph.optimize_fock_shapes(circ, verbose=True)
        assert [op.auto_shape() for op in circ] == [(6,), (6,), (12, 12, 6, 6)]


def test_path_errors():
    """Test that parse_components raises errors for invalid paths."""
    vac12 = Vacuum((1, 2))

    with pytest.raises(ValueError, match="Overlapping"):
        bb.parse_components([vac12, vac12])

    with pytest.raises(ValueError, match="Overlapping"):
        bb.parse_components([vac12.adjoint, vac12.adjoint])


@pytest.mark.parametrize(
    "cc, name",
    [
        (Number(0, n=15), "ArrayAnsatz"),
        (Sgate(0, r=1.0), "PolyExpAnsatz"),
        (Dgate(0, x=1.0), "PolyExpAnsatz"),
    ],
)
def test_from_circuitcomponent(cc, name):
    """Test that the from_circuitcomponent parses the ansatz name correctly."""
    comp = bb.GraphComponent.from_circuitcomponent(cc)
    assert comp.ansatz == name
