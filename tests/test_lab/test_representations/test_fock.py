####################Test Fock Class with means and cov.##################

# @given(x=st.floats(-1, 1), y=st.floats(-1, 1))
# def test_number_means(x, y):
#     assert np.allclose(State(ket=Coherent(x, y).ket([80])).number_means, x * x + y * y)
#     assert np.allclose(State(dm=Coherent(x, y).dm([80])).number_means, x * x + y * y)


# @given(x=st.floats(-1, 1), y=st.floats(-1, 1))
# def test_number_variances_coh(x, y):
#     assert np.allclose(fock.number_variances(Coherent(x, y).ket([80]), False)[0], x * x + y * y)
#     assert np.allclose(fock.number_variances(Coherent(x, y).dm([80]), True)[0], x * x + y * y)


# def test_number_variances_fock():
#     assert np.allclose(fock.number_variances(Fock(n=1).ket(), False), 0)
#     assert np.allclose(fock.number_variances(Fock(n=1).dm(), True), 0)