# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from mrmustard import settings
from mrmustard.lab.abstract import State
from mrmustard.lab.gates import Attenuator, Dgate, Ggate, Sgate
from mrmustard.lab.states import (
    Coherent,
    DisplacedSqueezed,
    Fock,
    Gaussian,
    SqueezedVacuum,
    Thermal,
    Vacuum,
)
from mrmustard.math import Math
from mrmustard.physics import gaussian as gp
from tests.random import angle, medium_float, nmodes, r, angle

math = Math()


####################TestInit###########################################
class TestStatesinit():

    #######################Test Coherent######################
    #With different Init parameters
    @pytest.mark.parametrize("xs, ys",[([0.1,0.2], [-0.1]), (0.5, [-0.1]), (None,medium_float)])
    def test_init_Coherent_paramters_failed(self, xs, ys):
        with pytest.raises(AttributeError):
            Coherent(x=xs, y=ys)


    @pytest.mark.parametrize("xs, ys",[([0.1,0.2], [0.5,-0.1]), ([0.5], [-0.1]), (0.2, -0.3)])
    def test_init_Coherent_with_list_of_correct_parameters(self, xs, ys):
        coh = Coherent(x=xs, y=ys)
        assert np.allclose(coh.representation.data.symplectic, settings.HBAR/2*np.identity(coh.representation.num_modes))
        assert np.allclose(coh.representation.data.displacement, np.sqrt(2*settings.HBAR)*math.concat([xs,ys], axis=0))


    #TODO: This test is used when we refactor the transformations with representation project.
    # def test_coh_state(xy):
    #     """Test coherent state preparation."""
    #     x, y = xy
    #     assert Vacuum(len(x)) >> Dgate(x, y) == Coherent(x, y)


    #######################Test Vacuum######################
    @given(nmodes=nmodes)
    @pytest.mark.parametrize("hbar", (1,2))
    def test_init_vacuum_state_with_different_nmodes_and_hbar(self, nmodes, hbar):
        vac = Vacuum(num_modes=nmodes, hbar=hbar)
        cov = vac.cov
        disp = vac.means
        assert np.allclose(cov, np.eye(2 * nmodes) * hbar / 2)
        assert np.allclose(disp, np.zeros_like(disp))

    #######################Test Fock######################
    @pytest.mark.parametrize("n", (2,3,4,5))
    def test_init_fock_state_single_mode(self, n):
        fock1 = Fock(n=n)
        assert fock1.representation.data.array[n] == 1
        assert fock1.representation.data.array[n-1] == 0

    
    def test_init_fock_state_multimode_without_cutoffs(self):
        fock1 = Fock(n=[1,2,3])
        assert fock1.representation.data.array[1,2,3] == 1


    @pytest.mark.parametrize("n", (2,3,4,5))
    @pytest.mark.parametrize("cutoffs", (6,7))
    def test_init_fock_state_single_mode_with_cutoffs(self, n, cutoffs):
        fock1 = Fock(n=n, cutoffs=cutoffs)
        assert fock1.representation.data.array.shape[0] == cutoffs


    def test_init_fock_state_multimode_with_larger_cutoffs(self):
        fock1 = Fock(n=[3,4], cutoffs=[7,8])
        assert fock1.representation.data.array.shape == (7,8,)

    
    def test_init_fock_state_multimode_with_smaller_cutoffs(self):
        fock1 = Fock(n=[2,5], cutoffs=[1,1])
        assert fock1.representation.data.array.shape == (2+1,5+1,)

    
    def test_init_fock_state_multimode_with_different_cutoffs(self):
        fock1 = Fock(n=[2,5], cutoffs=[1,10])
        assert fock1.representation.data.array.shape == (2+1,10,)


    #######################Test SqueezedVacuum######################
    @pytest.mark.parametrize("r", (0.2,0.5,0.3))
    @pytest.mark.parametrize("phi", (0.2,0.1))
    def test_init_sq_state(self, r, phi):
        sq = SqueezedVacuum(r, phi)
        assert np.allclose(sq.representation.data.symplectic, gp.squeezing_symplectic(r=r, phi=phi))
        assert np.allclose(sq.means, np.zeros_like(sq.means))


    @pytest.mark.parametrize("r,phi", [([0.1],[0.5]),([0.8,0.7],[-0.2,-0.1])])
    def test_init_sq_state_with_a_list_of_parameters(self, r, phi):
        sq = SqueezedVacuum(r, phi)
        assert np.allclose(sq.representation.data.symplectic, gp.squeezing_symplectic(r=r, phi=phi))
        assert np.allclose(sq.means, np.zeros_like(sq.means))


    #With different Init parameters
    @pytest.mark.parametrize("r, phi",[([0.1,0.2], [-0.1]), (0.5, [-0.1]), (None,medium_float)])
    def test_init_sq_state_with_paramters_failed(self, r, phi):
        with pytest.raises(AttributeError):
            SqueezedVacuum(r=r,phi=phi)

    #TODO: This test is used when we refactor the transformations with representation project.
    # @given(r=r, phi=angle)
    # def test_sq_state(r, phi):
    #     """Test squeezed vacuum preparation."""
    #     assert Vacuum(1) >> Sgate(r, phi) == SqueezedVacuum(r, phi)
    #######################Test Gaussian######################

    @given(nmodes=nmodes)
    def test_init_gaussian_states_singlemode(self, nmodes):
        symplectic = math.random_symplectic(num_modes=nmodes)
        eigenvalues = math.ones(nmodes) * settings.HBAR / 2
        g = Gaussian(num_modes=nmodes, symplectic=symplectic, eigenvalues=eigenvalues)
        assert np.allclose(g.cov, settings.HBAR/2*math.matmul( math.matmul(symplectic, math.diag(math.concat([eigenvalues, eigenvalues], axis=0))), math.transpose(symplectic)))
        assert np.allclose(g.means, math.zeros(2*nmodes))


    @given(nmodes=nmodes)
    def test_init_gaussian_states_multimode(self, nmodes):
        symplectic = math.random_symplectic(num_modes=nmodes)
        eigenvalues = math.ones(nmodes) * settings.HBAR / 2
        g = Gaussian(num_modes=nmodes, symplectic=symplectic, eigenvalues=eigenvalues)
        assert np.allclose(g.cov, settings.HBAR/2*math.matmul( math.matmul(symplectic, math.diag(math.concat([eigenvalues, eigenvalues], axis=0))), math.transpose(symplectic)))
        assert np.allclose(g.means, math.zeros(2*nmodes))


    #######################Test Thermal######################

    @given(nbar = r)
    def test_init_thermal_state_singlemode(self, nbar):
        th = Thermal(nbar=nbar)
        g = (2 * math.atleast_1d(nbar) + 1) * settings.HBAR / 2
        assert np.allclose(th.cov, math.diag(math.concat([g, g], axis=-1)))

    @pytest.mark.parametrize("nbar", ([1, 0.5], [0,1.2,-0.3]))
    def test_init_thermal_state_multimode(self, nbar):
        th = Thermal(nbar=nbar)
        g = (2 * math.atleast_1d(nbar) + 1) * settings.HBAR / 2
        assert np.allclose(th.cov, math.diag(math.concat([g, g], axis=-1)))
        assert np.allclose(th.means, math.zeros(2*len(nbar)))

    #######################Test DisplacedSqueezed######################
    @pytest.mark.parametrize("r", (0.2,0.3))
    @pytest.mark.parametrize("phi", (0.2,0.1))
    @pytest.mark.parametrize("x", (0.2,-0.8))
    @pytest.mark.parametrize("y", (0.2,0.1))
    def test_init_displacedsqueezed_state(self, x, y, r, phi):
        dssq = DisplacedSqueezed(x=x, y=y, r=r, phi=phi)
        assert np.allclose(dssq.representation.data.symplectic, gp.squeezing_symplectic(r=r, phi=phi))
        assert np.allclose(dssq.representation.data.displacement, np.sqrt(2*settings.HBAR)*math.concat([x,y], axis=0))

    #With different Init parameters
    @pytest.mark.parametrize("x, y, r, phi",[([0.1,0.2], [-0.1], None, None), (0.5, None, [-0.1], 0.1), (None,medium_float, None, medium_float)])
    def test_init_dispalcedsqueezed_paramters_failed(self, x, y, r, phi):
        with pytest.raises(AttributeError):
            DisplacedSqueezed(x=x, y=y, r=r, phi=phi)


    @pytest.mark.parametrize("xs, ys, rs, phis", [([0.1, 0.9, 0.2, 0.1], [0.5, 0.0, 0.0, 0.1], [-0.2, 1.0, -0.3, -0.5], [0.1, 0.2, 0.2, -0.1]), ([0.1, 0.2], [0.3, 0.4], [0.2, -0.4], [0.1, - 0.2])])
    def test_init_displacedsqueezed_with_list_of_correct_parameters(self, xs, ys, rs, phis):
        print(xs, ys, rs, phis, "here")
        dssq = DisplacedSqueezed(x=xs, y=ys, r=rs, phi=phis)
        assert np.allclose(dssq.representation.data.symplectic, gp.squeezing_symplectic(r=rs, phi=phis))
        assert np.allclose(dssq.representation.data.displacement, np.sqrt(2*settings.HBAR)*math.concat([xs,ys], axis=0))

    #TODO: This test is used when we refactor the transformations with representation project.
    # @given(
    #     x=medium_float,
    #     y=medium_float,
    #     r=r,
    #     phi=angle,
    # )
    # def test_dispsq_state(self, x, y, r, phi):
    #     """Test displaced squeezed state."""
    #     assert Vacuum(1) >> Sgate(r, phi) >> Dgate(x, y) == DisplacedSqueezed(r, phi, x, y)


class TestFockCutoffs():
    """Test if the padding works alongwith Fock representations in different cutoffs."""

    # #TODO: This test is used when we refactor the transformations with representation project.
    # def test_ket_from_pure_dm_new_cutoffs():
    #     "tests that the shape of the internal fock representation reflects the new cutoffs"
    #     state = Vacuum(1) >> Sgate(0.1) >> Dgate(0.1, 0.1)  # weak gaussian state
    #     state = State(dm=state.dm(cutoffs=[20]))  # assign pure dm directly
    #     assert state.ket(cutoffs=[5]).shape.as_list() == [5]  # shape should be [5]

    def test_padding_ket(self):
        "Test that padding a ket works correctly."
        state = State(fock=SqueezedVacuum(r=1.0).ket(cutoffs=[20]), flag_ket=True)
        assert len(state.ket(cutoffs=[10])) == 10
        assert len(state.ket(cutoffs=[50])) == 50

    def test_padding_dm(self):
        "Test that padding a density matrix works correctly."
        state = State(fock = np.random.random([20,20]), flag_ket=False)
        assert state.dm(cutoffs=[30,30]).shape == (30,30,)
        assert state.dm(cutoffs=[10,30]).shape == (10,30,)

    def test_padding_from_ket_to_dm(self):
        "Test that padding a state from ket to dm correctly."
        state = State(fock=SqueezedVacuum(r=1.0).ket(cutoffs=[20]), flag_ket=True)
        assert state.dm().shape == (20,20,)
        assert state.dm(cutoffs=[30,30]).shape == (30,30,)
        assert state.dm(cutoffs=[10,30]).shape == (10,30,)


    #TODO: This test is used when we refactor the transformations with representation project.
    # def test_padding_dm(self):
        #     "Test that padding a density matrix works correctly."
        #     state = State(dm=(SqueezedVacuum(r=1.0) >> Attenuator(0.6)).dm(cutoffs=[20]))
        #     assert tuple(int(c) for c in state.dm(cutoffs=[10]).shape) == (10, 10)
        #     assert tuple(int(c) for c in state._dm.shape) == (20, 20)  # pylint: disable=protected-access

# This part should be
# class TestStatesOthers():

#     @given(r1=r, phi1=angle, r2=r, phi2=angle)
#     def test_join_two_states(r1, phi1, r2, phi2):
#         """Test Sgate acts the same in parallel or individually for two states."""
#         S1 = Vacuum(1) >> Sgate(r=r1, phi=phi1)
#         S2 = Vacuum(1) >> Sgate(r=r2, phi=phi2)
#         S12 = Vacuum(2) >> Sgate(r=[r1, r2], phi=[phi1, phi2])
#         assert S1 & S2 == S12


#     @given(r1=r, phi1=angle, r2=r, phi2=angle, r3=r, phi3=angle)
#     def test_join_three_states(r1, phi1, r2, phi2, r3, phi3):
#         """Test Sgate acts the same in parallel or individually for three states."""
#         S1 = Vacuum(1) >> Sgate(r=r1, phi=phi1)
#         S2 = Vacuum(1) >> Sgate(r=r2, phi=phi2)
#         S3 = Vacuum(1) >> Sgate(r=r3, phi=phi3)
#         S123 = Vacuum(3) >> Sgate(r=[r1, r2, r3], phi=[phi1, phi2, phi3])
#         assert S123 == S1 & S2 & S3

    
#     # @pytest.mark.parametrize("pure", [True, False])
#     # def test_concat_pure_states(pure):
#     #     """Test that fock states concatenate correctly and are separable"""
#     #     state1 = Fock(1, cutoffs=[15])
#     #     state2 = Fock(4, cutoffs=[15])

#     #     if not pure:
#     #         state1 >>= Attenuator(transmissivity=0.95)
#     #         state2 >>= Attenuator(transmissivity=0.9)

#     #     psi = state1 & state2

#     #     # test concatenated state
#     #     psi_dm = math.transpose(math.tensordot(state1.dm(), state2.dm(), [[], []]), [0, 2, 1, 3])
#     #     assert np.allclose(psi.dm(), psi_dm)