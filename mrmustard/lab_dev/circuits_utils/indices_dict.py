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
Helper to compute the indices_dict for a circuit.
"""


def indices_dict(components):
    res = {}
    for i, opA in enumerate(components):
        out_idx = set(opA.wires.output.indices)
        indices: dict[int, dict[int, int]] = {}
        for j, opB in enumerate(components[i + 1 :]):
            ovlp_bra = opA.wires.output.bra.modes & opB.wires.input.bra.modes
            ovlp_ket = opA.wires.output.ket.modes & opB.wires.input.ket.modes
            if not (ovlp_bra or ovlp_ket):
                continue
            iA = opA.wires.output.bra[ovlp_bra].indices + opA.wires.output.ket[ovlp_ket].indices
            iB = opB.wires.input.bra[ovlp_bra].indices + opB.wires.input.ket[ovlp_ket].indices
            if not out_idx.intersection(iA):
                continue
            indices[i + j + 1] = dict(zip(iA, iB))
            out_idx -= set(iA)
            if not out_idx:
                break
        res[i] = indices
    return res
