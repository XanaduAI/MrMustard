import numpy as np

from mrmustard.lab import Attenuator, Gaussian


class TestStateAlgebra():

    def test_addition():
        """Test that addition of Gaussians is correct"""
        G0 = Gaussian(1, cutoffs=[10])
        G1 = Gaussian(1, cutoffs=[10])

        mixed = G0 + G1

        assert np.allclose(mixed.dm([10]), G0.dm([10]) + G1.dm([10]))


    def test_multiplication_ket():
        """Test that multiplication of Gaussians is correct"""
        G = Gaussian(1, cutoffs=[10])

        scaled = 42.0 * G

        assert np.allclose(scaled.ket(G.cutoffs), 42.0 * G.ket())


    def test_multiplication_dm():
        """Test that multiplication of Gaussians is correct"""
        G = Gaussian(1) >> Attenuator(0.9)

        scaled = 42.0 * G

        assert np.allclose(scaled.dm(), 42.0 * G.dm())


    def test_division_ket():
        """Test that division of Gaussians is correct"""
        G = Gaussian(1, cutoffs=[10])

        scaled = G / 42.0

        assert np.allclose(scaled.ket([10]), G.ket([10]) / 42.0)


    def test_division_dm():
        """Test that division of Gaussians is correct"""
        G = Gaussian(1) >> Attenuator(0.9)

        scaled = G / 42.0

        assert np.allclose(scaled.dm(G.cutoffs), G.dm() / 42.0)



class TestStatesProperties():

    def test_get_modes():
        """Test get_modes returns the states as expected."""
        a = Gaussian(2)
        b = Gaussian(2)
        assert a == (a & b).get_modes([0, 1])
        assert b == (a & b).get_modes([2, 3])

    
#     @given(modes=st.lists(st.integers(), min_size=2, max_size=5, unique=True))
#     def test_getitem_set_modes(modes):
#         """Test that using `State.__getitem__` and `modes`
#         kwarg correctly set the modes of the state."""

#         cutoff = len(modes) + 1
#         ket = np.zeros([cutoff] * len(modes), dtype=np.complex128)
#         ket[1, 1] = 1.0 + 0.0j

#         state1 = State(ket=ket)[modes]
#         state2 = State(ket=ket, modes=modes)

#         assert state1.modes == state2.modes


#     def test_hbar():
#         """Test cov matrix is linear in hbar."""
#         g = Gaussian(2)
#         p = g.purity
#         settings.HBAR = 1.234
#         assert g.purity == p
#         settings.HBAR = 2

#     def test_get_single_mode():
#         """Test get_modes leaves a single-mode state untouched."""
#         a = Gaussian(1)[1]
#         assert a == a.get_modes([1])


#     def test_get_single_mode_fail():
#         """Test get_modes leaves a single-mode state untouched."""
#         a = Gaussian(1)[1]
#         with pytest.raises(ValueError):
#             a.get_modes([0])


#     def test_iter():
#         """Test we can iterate individual modes in states."""
#         a = Gaussian(1)
#         b = Gaussian(2)
#         c = Gaussian(1)
#         for i, mode in enumerate(a & b & c):
#             assert (a, b.get_modes(0), b.get_modes(1), c)[i] == mode

#     def test_the_purity_of_a_Ket_state_tobe_1(self):
#         pass

#     def test_the_purity_of_a_Density_Matrix(self):
#         pass

#     def test_ket_probability(self):
#         pass

#     def test_density_matrix_probability(self):
#         pass

#     # @given(state=n_mode_pure_state(num_modes=1))
#     # def test_the_purity_of_a_pure_state(state):
#     #     purity = gp.purity(state.cov, settings.HBAR)
#     #     expected = 1.0
#     #     assert np.isclose(purity, expected)


#     # @given(nbar=st.floats(0.0, 3.0))
#     # def test_the_purity_of_a_mixed_state(nbar):
#     #     state = Thermal(nbar)
#     #     purity = gp.purity(state.cov, settings.HBAR)
#     #     expected = 1 / (2 * nbar + 1)
#     #     assert np.isclose(purity, expected)


#     # def test_ket_probability():
#     #     "Test that the probability of a ket is calculated correctly."
#     #     state = State(ket=np.array([0.5, 0.5]))
#     #     assert np.isclose(state.probability, 2 * 0.5**2)


#     # def test_dm_probability():
#     #     "Test that the probability of a density matrix is calculated correctly."
#     #     state = State(dm=np.array([[0.4, 0.1], [0.1, 0.4]]))
#     #     assert np.isclose(state.probability, 0.8)


# class TestStatesFunctions():

#     def test_state_repr_small_prob():
#         "test that small probabilities are displayed correctly"
#         state = State(ket=np.array([0.0001, 0.0001]))
#         table = state._repr_markdown_()  # pylint: disable=protected-access
#         assert "2.000e-06 %" in table


#     def test_state_repr_big_prob():
#         "test that big probabilities are displayed correctly"
#         state = State(ket=np.array([0.5, 0.5]))
#         table = state._repr_markdown_()  # pylint: disable=protected-access
#         assert "50.000%" in table


#     @given(m=st.integers(0, 3))
#     def test_modes_after_projection(m):
#         """Test number of modes is correct after single projection."""
#         a = Gaussian(4) << Fock(1)[m]
#         assert np.allclose(a.modes, [k for k in range(4) if k != m])
#         assert len(a.modes) == 3


#     @given(n=st.integers(0, 3), m=st.integers(0, 3))
#     def test_modes_after_double_projection(n, m):
#         """Test number of modes is correct after double projection."""
#         assume(n != m)
#         a = Gaussian(4) >> Dgate(x=1.0)[0, 1, 2, 3] << Fock([1, 2])[n, m]
#         assert np.allclose(a.modes, [k for k in range(4) if k != m and k != n])
#         assert len(a.modes) == 2


#     def test_random_state_is_entangled():
#         """Tests that a Gaussian state generated at random is entangled."""
#         state = Vacuum(2) >> Ggate(num_modes=2)
#         mat = state.cov
#         assert np.allclose(gp.log_negativity(mat, 2), 0.0)
#         assert np.allclose(
#             gp.log_negativity(gp.physical_partial_transpose(mat, [0, 1]), 2), 0.0, atol=1e-7
#         )
#         N1 = gp.log_negativity(gp.physical_partial_transpose(mat, [0]), 2)
#         N2 = gp.log_negativity(gp.physical_partial_transpose(mat, [1]), 2)

#         assert N1 > 0
#         assert N2 > 0
#         assert np.allclose(N1, N2)




#     # @pytest.mark.parametrize("n", ([1, 0, 0], [1, 1, 0], [0, 0, 1]))
#     # @pytest.mark.parametrize("cutoffs", ([2, 2, 2], [2, 3, 3], [3, 3, 2]))
#     # def test_ket_from_pure_dm(n, cutoffs):
#     #     # prepare a fock (pure) state
#     #     fock_state = Fock(n=n, cutoffs=cutoffs)
#     #     dm_fock = fock_state.dm()

#     #     # initialize a new state from the density matrix
#     #     # (no knowledge of the ket)
#     #     test_state = State(dm=dm_fock)
#     #     test_ket = test_state.ket()

#     #     # check test state calculated the same ket as the original state
#     #     assert np.allclose(test_ket, fock_state.ket())