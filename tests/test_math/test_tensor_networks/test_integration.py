# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains tests for simulating quantum circuits with tensor networks."""

import numpy as np
import pytest

from mrmustard.lab_dev.transformations import BSgate, CXgate, CZgate, Dgate, MZgate, Rgate, Sgate
from mrmustard.math.tensor_networks import connect, contract
from ...conftest import skip_np, skip_jax


class TestTransformations:
    r"""
    Tests that transformations can be contracted by tensor networks.
    """

    @pytest.mark.parametrize("modes", [(0), (1, 2), (3, 4, 5, 6)])
    @pytest.mark.parametrize("dim", [3, 4])
    @pytest.mark.parametrize("default_dim", [2, 5])
    def test_sequence_one_mode_unitaries(self, modes, dim, default_dim):
        r"""
        Tests that a sequence of one-mode unitaries can be contracted correctly.
        """
        s_tens = Sgate(modes, r=0.1).fock_array(shape=(default_dim, dim))
        r_tens = Rgate(modes, theta=0.2).fock_array(dim)
        d_tens = Dgate(modes, x=0.3).fock_array(shape=(dim, default_dim))

        contraction = contract([s_tens, r_tens, d_tens], default_dim)

        s_mat = Sgate(0, 0.1).fock_array(shape=(default_dim, dim))
        r_mat = Rgate(0.2).fock_array(dim)
        d_mat = Dgate(0.3).fock_array(shape=(dim, default_dim))
        expected = np.dot(s_mat, r_mat)
        expected = np.dot(expected, d_mat)
        if len(modes) == 2:
            expected = np.kron(expected, expected)
            expected = expected.reshape(*contraction.shape)
        if len(modes) == 4:
            expected = np.kron(expected, expected)
            expected = np.kron(expected, expected)
            expected = expected.reshape(*contraction.shape)

        assert np.allclose(contraction, expected)

    @pytest.mark.parametrize("modes", [[1, 2]])
    @pytest.mark.parametrize("dim", [3, 20])
    @pytest.mark.parametrize("default_dim", [2, 10])
    def test_sequence_multi_mode_unitaries(self, modes, dim, default_dim):
        r"""
        Tests that a sequence of multi-mode unitaries can be contracted correctly.
        """
        skip_np()
        skip_jax()
        cx_tens = CXgate(modes)
        bs_tens = BSgate(modes, 0.2)
        cz_tens = CZgate(modes)
        mz_tens = MZgate(modes, 0.3)

        for mode in modes:
            connect(cx_tens.output.ket[mode], bs_tens.input.ket[mode], dim)
            connect(bs_tens.output.ket[mode], cz_tens.input.ket[mode], dim)
            connect(cz_tens.output.ket[mode], mz_tens.input.ket[mode], dim)
        contraction = contract([cx_tens, bs_tens, cz_tens, mz_tens], default_dim)

        assert contraction.shape == (default_dim, default_dim, default_dim, default_dim)
        # TODO: find a way to validate the tensor's values
        # --> when states are available, apply to states and compare with the expected dm.
