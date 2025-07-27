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

"""Tests for the ``SqueezedVacuum`` class."""

import pytest

from mrmustard import math
from mrmustard.lab.states import SqueezedVacuum, Vacuum
from mrmustard.lab.transformations import Sgate


class TestSqueezedVacuum:
    r"""
    Tests for the ``SqueezedVacuum`` class.
    """

    modes = [0, 1, 7]
    r = [1, 2, 3]
    phi = [3, 4, 5]

    @pytest.mark.parametrize("modes,r,phi", zip(modes, r, phi))
    def test_init(self, modes, r, phi):
        state = SqueezedVacuum(modes, r, phi)

        assert state.name == "SqueezedVacuum"
        assert state.modes == (modes,)

    def test_fock_representation(self):
        shape = (5,)
        sq_vac = SqueezedVacuum(0, r=1, phi=2)
        sq_vac_fock = sq_vac.fock_array(shape)

        herm_renom = math.hermite_renormalized(*sq_vac.ansatz.triple, shape=shape)
        assert math.allclose(sq_vac_fock, herm_renom)

        sq_vac_batch = SqueezedVacuum(0, r=[1, 1, 1], phi=[2, 2, 2])
        sq_vac_batch_fock = sq_vac_batch.fock_array(shape)
        herm_renom_batch = math.hermite_renormalized(*sq_vac_batch.ansatz.triple, shape=shape)
        assert math.allclose(sq_vac_batch_fock, herm_renom_batch)

    def test_to_fock_lin_sup(self):
        bsgate = (SqueezedVacuum(0, 2, 3) + SqueezedVacuum(0, -2, -3)).to_fock(5)
        assert bsgate.ansatz.batch_dims == 0
        assert bsgate.ansatz.batch_shape == ()
        assert bsgate.ansatz.array.shape == (5,)

    @pytest.mark.parametrize("modes,r,phi", zip(modes, r, phi))
    @pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
    def test_representation(self, modes, r, phi, batch_shape):
        r = math.broadcast_to(r, batch_shape)
        phi = math.broadcast_to(phi, batch_shape)
        rep = SqueezedVacuum(modes, r, phi).ansatz
        exp = (Vacuum(modes) >> Sgate(modes, r, phi)).ansatz
        assert rep == exp

    def test_trainable_parameters(self):
        state1 = SqueezedVacuum(0, 1, 1)
        state2 = SqueezedVacuum(0, 1, 1, r_trainable=True, r_bounds=(-2, 2))
        state3 = SqueezedVacuum(0, 1, 1, phi_trainable=True, phi_bounds=(-2, 2))

        with pytest.raises(AttributeError):
            state1.parameters.r.value = 3

        state2.parameters.r.value = 2
        assert state2.parameters.r.value == 2

        state3.parameters.phi.value = 2
        assert state3.parameters.phi.value == 2
