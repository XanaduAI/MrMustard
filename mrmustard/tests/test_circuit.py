from mrmustard import *
from hypothesis import given, strategies as st
from thewalrus.fock_gradients import displacement, squeezing, beamsplitter
import numpy as np

@given(x=st.floats(min_value=-2, max_value=2),y = st.floats(min_value=-2, max_value=2))
def test_circuit_fock_representation_displacement(x, y):
    circ = Circuit()
    circ.append(Dgate(x=x, y=y)[0])
    expected = displacement(r= np.sqrt(x**2 + y**2), phi = np.arctan2(y, x), cutoff = 20)
    assert np.allclose(expected, circ.fock(cutoffs=[20,20]))

@given(r=st.floats(min_value=0, max_value=2), phi = st.floats(min_value=0, max_value=2*np.pi))
def test_circuit_fock_representation_squeezing(r, phi):
    circ = Circuit()
    circ.append(Sgate(r=r, phi=phi)[0])
    expected = squeezing(r=r, theta=phi, cutoff = 20)
    assert np.allclose(expected, circ.fock(cutoffs=[20,20]))

@given(theta=st.floats(min_value=0, max_value=2*np.pi), phi = st.floats(min_value=0, max_value=2*np.pi))
def test_circuit_fock_representation_beamsplitter(theta, phi):
    circ = Circuit()
    circ.append(BSgate(theta=theta, phi=phi)[0,1])
    expected = beamsplitter(theta=theta, phi=phi, cutoff = 20)
    assert np.allclose(expected, circ.fock(cutoffs=[20,20,20,20]))
