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

from mrmustard.physics.representations import Representation, Bargmann, Fock
from mrmustard.physics.converters import to_fock
from mrmustard.physics.triples import vacuum_state_Abc, coherent_state_Abc
from mrmustard import settings


class TestToFock:
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
        vacuum_bargmann = Bargmann(*vacuum_state_Abc(n_modes=2))
        with pytest.raises(ValueError):
            to_fock(vacuum_bargmann, cutoffs=[50])

    def test_tofock_from_a_bargmann_coherent_state(self):
        r"""Tests that the to_fock function works for a coherent state in Bargmann representation."""
        coherent_bargmann = Bargmann(*coherent_state_Abc(x=[0.3, 0.2], y=[0.1]))
        coherent_fock_no_cutoffs = to_fock(coherent_bargmann)
        assert coherent_fock_no_cutoffs.array[0, 0, 0] == np.exp(
            -0.5 * (0.3**2 + 0.2**2 + 2 * 0.1**2)
        )
        assert coherent_fock_no_cutoffs.array[0, 1, 0] == (0.3 + 1j * 0.1) * np.exp(
            -0.5 * (0.3**2 + 0.2**2 + 2 * 0.1**2)
        )
        assert coherent_fock_no_cutoffs.array.shape[-1] == settings.AUTOCUTOFF_MAX_CUTOFF
        assert coherent_fock_no_cutoffs.array.shape == (1, 100, 100)
