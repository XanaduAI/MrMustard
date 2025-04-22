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

"""Tests for the lattice/steps.py file"""

import numpy as np
from numba.typed import Dict
from numba import types

from mrmustard.math.lattice.steps import binomial_step_dict, binomial_step_dict_stable_no_prob
from mrmustard.math.lattice.paths import binomial_subspace_basis
from tests.random import Abc_triple


class TestSteps:
    def test_binomial_step_dict_vs_stable(self):
        r"""Tests that binomial_step_dict and binomial_step_dict_stable yield the same results"""
        n = 3
        A, b, _ = Abc_triple(n)

        cutoffs = (3,) * n
        subspace_indices_0 = binomial_subspace_basis(cutoffs=cutoffs, weight=7)
        subspace_indices_1 = binomial_subspace_basis(cutoffs=cutoffs, weight=6)
        subspace_indices_2 = binomial_subspace_basis(cutoffs=cutoffs, weight=5)

        G_normal = Dict.empty(key_type=types.UniTuple(types.int64, n), value_type=types.complex128)
        for idx in subspace_indices_1:
            G_normal[idx] = np.random.rand() + 1.0j * np.random.rand()
        for idx in subspace_indices_2:
            G_normal[idx] = np.random.rand() + 1.0j * np.random.rand()
        G_stable = G_normal.copy()

        G_normal, _ = binomial_step_dict(G_normal, A, b, subspace_indices_0)
        G_stable = binomial_step_dict_stable_no_prob(G_stable, A, b, subspace_indices_0)

        assert set(G_normal.keys()) == set(G_stable.keys())
        for key in subspace_indices_0:
            assert np.isclose(G_normal[key], G_stable[key])
