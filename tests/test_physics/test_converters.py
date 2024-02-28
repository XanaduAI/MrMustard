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

"""Tests the converters between representations."""

import numpy as np
import pytest

from mrmustard.physics.representations import Bargmann, Fock
from mrmustard.physics.converters import to_fock
from mrmustard.physics.triples import (
    vacuum_state_Abc,
    coherent_state_Abc,
    displacement_gate_Abc,
    squeezing_gate_Abc,
    beamsplitter_gate_Abc,
    attenuator_Abc,
)
from mrmustard import settings, math


class TestToFock:
    r"""Tests related to the to_fock function to transform any representations into Fock representation."""

    def test_tofock_from_a_fock(self):
        r"""Tests that the to_fock function works for Fock representation."""
        fock_original = Fock(np.arange(16).reshape((4, 4)), batched=False)
        fock_after = to_fock(fock_original)
        assert fock_original == fock_after

        fock_after = to_fock(fock_original, cutoffs=50)
        assert fock_original == fock_after

    def test_tofock_from_a_bargmann_vacuum_state(self):
        r"""Tests that the to_fock function works for a vacuum state in Bargmann representation."""
        vacuum_bargmann = Bargmann(*vacuum_state_Abc(n_modes=2))
        vacuum_fock_no_cutoffs = to_fock(vacuum_bargmann)
        assert vacuum_fock_no_cutoffs.array[0, 0, 0] == 1
        assert vacuum_fock_no_cutoffs.array.shape[-1] == settings.AUTOCUTOFF_MAX_CUTOFF
        assert vacuum_fock_no_cutoffs.array.shape == (1, 100, 100)

        vacuum_fock_with_int_cutoffs = to_fock(vacuum_bargmann, cutoffs=80)
        assert vacuum_fock_with_int_cutoffs.array[0, 0, 0] == 1
        assert vacuum_fock_with_int_cutoffs.array.shape[-1] == 80
        assert vacuum_fock_with_int_cutoffs.array.shape == (1, 80, 80)

    def test_incompatible_cutoffs(self):
        r"""Tests that the ValueError raises when the cutoffs given are incompatible."""
        vacuum_bargmann = Bargmann(*vacuum_state_Abc(n_modes=2))
        with pytest.raises(ValueError):
            to_fock(vacuum_bargmann, cutoffs=[50])

    def test_tofock_from_a_bargmann_coherent_state(self):
        r"""Tests that the to_fock function works for a coherent state in Bargmann representation."""
        coherent_bargmann = Bargmann(*coherent_state_Abc(x=[0.3], y=[0.1]))
        coherent_fock_no_cutoffs = to_fock(coherent_bargmann)
        assert math.allclose(
            coherent_fock_no_cutoffs.array[0, 0], math.exp(-0.5 * (0.3**2 + 0.1**2))
        )
        assert math.allclose(
            coherent_fock_no_cutoffs.array[0, 1],
            (0.3 + 1j * 0.1) * coherent_fock_no_cutoffs.array[0, 0],
        )
        assert math.allclose(
            coherent_fock_no_cutoffs.array[0, 2],
            (0.3 + 1j * 0.1) / math.sqrt(2.0 + 0j) * coherent_fock_no_cutoffs.array[0, 1],
        )
        assert coherent_fock_no_cutoffs.array.shape[-1] == settings.AUTOCUTOFF_MAX_CUTOFF
        assert coherent_fock_no_cutoffs.array.shape == (1, 100)

        coherent_twomode_bargmann = Bargmann(*coherent_state_Abc(x=[0.3, 0.2], y=[0.1]))
        coherent_twomode_fock_no_cutoffs = to_fock(coherent_twomode_bargmann)
        assert math.allclose(
            coherent_twomode_fock_no_cutoffs.array[0, 0, 0],
            math.exp(-0.5 * (0.3**2 + 0.1**2 + 0.2**2 + 0.1**2)),
        )
        assert math.allclose(
            coherent_twomode_fock_no_cutoffs.array[0, 0, 1],
            (0.2 + 1j * 0.1) * coherent_twomode_fock_no_cutoffs.array[0, 0, 0],
        )
        assert math.allclose(
            coherent_twomode_fock_no_cutoffs.array[0, 0, 2],
            (0.2 + 1j * 0.1)
            * coherent_twomode_fock_no_cutoffs.array[0, 0, 1]
            / math.sqrt(2.0 + 0j),
        )
        assert math.allclose(
            coherent_twomode_fock_no_cutoffs.array[0, 1, 2],
            (0.3 + 1j * 0.1) * coherent_twomode_fock_no_cutoffs.array[0, 0, 2],
        )
        assert coherent_twomode_fock_no_cutoffs.array.shape[-1] == settings.AUTOCUTOFF_MAX_CUTOFF
        assert coherent_twomode_fock_no_cutoffs.array.shape == (1, 100, 100)

    def test_tofock_from_a_bargmann_displacement_gate(self):
        r"""Tests that the to_fock function works for a displacement gate in Bargmann representation."""
        dgate_bargmann = Bargmann(*displacement_gate_Abc(x=[0.3], y=[0.1]))
        dgate_fock_with_cutoffs = to_fock(dgate_bargmann, cutoffs=[10, 10])
        assert math.allclose(
            dgate_fock_with_cutoffs.array[0, 0, 0], math.exp(-0.5 * (0.3**2 + 0.1**2))
        )
        assert math.allclose(
            dgate_fock_with_cutoffs.array[0, 1, 0],
            (0.3 + 1j * 0.1) * dgate_fock_with_cutoffs.array[0, 0, 0],
        )
        assert math.allclose(
            dgate_fock_with_cutoffs.array[0, 2, 0],
            (0.3 + 1j * 0.1) * dgate_fock_with_cutoffs.array[0, 1, 0] / math.sqrt(2.0 + 0j),
        )
        assert math.allclose(
            dgate_fock_with_cutoffs.array[0, 1, 1],
            (-0.3 + 1j * 0.1) * dgate_fock_with_cutoffs.array[0, 1, 0]
            + dgate_fock_with_cutoffs.array[0, 0, 0],
        )
        assert dgate_fock_with_cutoffs.array.shape == (1, 10, 10)

    def test_tofock_from_a_bargmann_squeezing_gate(self):
        r"""Tests that the to_fock function works for a squeezing gate in Bargmann representation."""
        sgate_bargmann = Bargmann(*squeezing_gate_Abc(r=[0.3], delta=[0.1]))
        sgate_fock_with_cutoffs = to_fock(sgate_bargmann, cutoffs=[8, 12])
        assert math.allclose(sgate_fock_with_cutoffs.array[0, 0, 0], 1 / math.sqrt(math.cosh(0.3)))
        tanhr = math.sinh(0.3) / math.cosh(0.3)
        assert math.allclose(
            sgate_fock_with_cutoffs.array[0, 2, 0],
            -tanhr * np.exp(1j * 0.1) * sgate_fock_with_cutoffs.array[0, 0, 0] / math.sqrt(2 + 0j),
        )
        assert math.allclose(
            sgate_fock_with_cutoffs.array[0, 0, 2],
            tanhr * np.exp(-1j * 0.1) * sgate_fock_with_cutoffs.array[0, 0, 0] / math.sqrt(2 + 0j),
        )
        assert math.allclose(
            sgate_fock_with_cutoffs.array[0, 1, 1],
            1 / math.cosh(0.3) * sgate_fock_with_cutoffs.array[0, 0, 0],
        )
        assert math.allclose(
            sgate_fock_with_cutoffs.array[0, 2, 1],
            0,
        )
        assert sgate_fock_with_cutoffs.array.shape == (1, 8, 12)

    def test_tofock_from_a_bargmann_beamsplitter_gate(self):
        r"""Tests that the to_fock function works for a BS gate in Bargmann representation."""
        bsgate_bargmann = Bargmann(*beamsplitter_gate_Abc(theta=[0.2], phi=[0.1]))
        bsgate_fock_with_cutoffs = to_fock(bsgate_bargmann, cutoffs=[5, 4, 7, 3])
        V = np.array(
            [
                [math.cos(0.2), -math.exp(-1j * 0.1) * math.sin(0.2)],
                [math.exp(1j * 0.1) * math.sin(0.2), math.cos(0.2)],
            ]
        )
        assert math.allclose(bsgate_fock_with_cutoffs.array[0, 0, 0, 0, 0], 1.0)
        assert math.allclose(bsgate_fock_with_cutoffs.array[0, 0, 1, 0, 0], 0.0)
        assert math.allclose(bsgate_fock_with_cutoffs.array[0, 0, 0, 0, 1], 0.0)
        assert math.allclose(bsgate_fock_with_cutoffs.array[0, 0, 1, 0, 1], V[1, 1])
        assert math.allclose(bsgate_fock_with_cutoffs.array[0, 0, 1, 1, 0], V[1, 0])
        assert math.allclose(bsgate_fock_with_cutoffs.array[0, 1, 0, 1, 0], V[0, 0])
        assert math.allclose(bsgate_fock_with_cutoffs.array[0, 1, 0, 0, 1], V[0, 1])
        assert math.allclose(bsgate_fock_with_cutoffs.array[0, 0, 3, 6, 1], 0.0)
        assert bsgate_fock_with_cutoffs.array.shape == (1, 5, 4, 7, 3)

    def test_tofock_from_a_bargmann_attenuator_channel(self):
        r"""Tests that the to_fock function works for a lossy channel in Bargmann representation."""
        attenuator_bargmann = Bargmann(*attenuator_Abc(eta=0.5))
        attenuator_bargmann_fock_with_cutoffs = to_fock(attenuator_bargmann, cutoffs=[5, 5, 5, 5])
        assert attenuator_bargmann_fock_with_cutoffs.array[0, 0, 0, 0, 0] == 1.0
        assert attenuator_bargmann_fock_with_cutoffs.array[0, 0, 0, 0, 1] == 0.0
        assert attenuator_bargmann_fock_with_cutoffs.array[0, 0, 0, 1, 0] == 0.0
        assert attenuator_bargmann_fock_with_cutoffs.array[0, 0, 0, 2, 0] == 0.0
        assert math.allclose(
            attenuator_bargmann_fock_with_cutoffs.array[0, 0, 0, 1, 1], math.sqrt(0.5)
        )
        assert math.allclose(attenuator_bargmann_fock_with_cutoffs.array[0, 0, 0, 2, 1], 0.0)
        assert math.allclose(attenuator_bargmann_fock_with_cutoffs.array[0, 0, 1, 2, 1], 0.0)
        assert attenuator_bargmann_fock_with_cutoffs.array.shape == (1, 5, 5, 5, 5)
