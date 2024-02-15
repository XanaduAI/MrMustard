import numpy as np
import pytest

from mrmustard import math
from mrmustard.lab import Attenuator, Dgate, Gaussian, Ggate
from mrmustard.physics.bargmann import (
    complex_gaussian_integral,
    contract_two_Abc,
    join_Abc,
    wigner_to_bargmann_Choi,
    wigner_to_bargmann_psi,
    wigner_to_bargmann_rho,
    wigner_to_bargmann_U,
    reorder_abc,
)
from mrmustard.physics import triples


def test_reorder_abc():
    """Test that the reorder_abc function works correctly"""
    A = np.array([[1, 2], [2, 3]])
    b = np.array([4, 5])
    c = np.array(6)
    same = reorder_abc((A, b, c), (0, 1))
    assert all(np.allclose(x, y) for x, y in zip(same, (A, b, c)))
    flipped = reorder_abc((A, b, c), (1, 0))
    assert all(np.allclose(x, y) for x, y in zip(flipped, (A[[1, 0], :][:, [1, 0]], b[[1, 0]], c)))
    c = np.array([[6, 7], [8, 9]])
    flipped = reorder_abc((A, b, c), (1, 0))  #  test transposition of c
    assert all(
        np.allclose(x, y) for x, y in zip(flipped, (A[[1, 0], :][:, [1, 0]], b[[1, 0]], c.T))
    )


def test_wigner_to_bargmann_psi():
    """Test that the Bargmann representation of a ket is correct"""
    G = Gaussian(2) >> Dgate(0.1, 0.2)

    for x, y in zip(G.bargmann(), wigner_to_bargmann_psi(G.cov, G.means)):
        assert np.allclose(x, y)


def test_wigner_to_bargmann_rho():
    """Test that the Bargmann representation of a dm is correct"""
    G = Gaussian(2) >> Dgate(0.1, 0.2) >> Attenuator(0.9)

    for x, y in zip(G.bargmann(), wigner_to_bargmann_rho(G.cov, G.means)):
        assert np.allclose(x, y)


def test_wigner_to_bargmann_U():
    """Test that the Bargmann representation of a unitary is correct"""
    G = Ggate(2) >> Dgate(0.1, 0.2)
    X, _, d = G.XYd(allow_none=False)
    for x, y in zip(G.bargmann(), wigner_to_bargmann_U(X, d)):
        assert np.allclose(x, y)


def test_wigner_to_bargmann_choi():
    """Test that the Bargmann representation of a Choi matrix is correct"""
    G = Ggate(2) >> Dgate(0.1, 0.2) >> Attenuator(0.9)
    X, Y, d = G.XYd(allow_none=False)
    for x, y in zip(G.bargmann(), wigner_to_bargmann_Choi(X, Y, d)):
        assert np.allclose(x, y)


def test_bargmann_numpy_state():
    """Tests that the numpy option of the bargmann method of State works correctly"""
    state = Gaussian(1)
    assert all(isinstance(thing, np.ndarray) for thing in state.bargmann(numpy=True))


def test_bargmann_numpy_transformation():
    """Tests that the numpy option of the bargmann method of State works correctly"""
    transformation = Ggate(1)
    assert all(isinstance(thing, np.ndarray) for thing in transformation.bargmann(numpy=True))

def test_join_Abc():
    """Tests the ``join_Abc`` method."""
    A1, b1, c1 = triples.vacuum_state_Abc(2)
    A2, b2, c2 = triples.displacement_gate_Abc(x=[0.1, 0.2], y=0.3)

    joined_Abc = join_Abc((A1, b1, c1), (A2, b2, c2))
    assert np.allclose(joined_Abc[0], math.block_diag(A1, A2))
    assert np.allclose(joined_Abc[1], math.concat([b1, b2], axis=-1))
    assert np.allclose(joined_Abc[2], math.outer(c1, c2))

    A12 = math.block_diag(A1, A2)
    b12 = math.concat([b1, b2], axis=-1)
    c12 = math.outer(c1, c2)
    return A12, b12, c12

def test_complex_gaussian_integral():
    """Tests the ``complex_gaussian_integral`` method."""
    A1, b1, c1 = triples.vacuum_state_Abc(2)
    A2, b2, c2 = triples.displacement_gate_Abc(x=[0.1, 0.2], y=0.3)
    A3, b3, c3 = triples.displaced_squeezed_vacuum_state_Abc(x=[0.1, 0.2], y=0.3)

    joined_Abc = join_Abc((A1, b1, c1), (A2, b2, c2))

    res1 = complex_gaussian_integral(joined_Abc, [], [])
    assert np.allclose(res1[0], joined_Abc[0])
    assert np.allclose(res1[1], joined_Abc[1])
    assert np.allclose(res1[2], joined_Abc[2])

    res2 = complex_gaussian_integral(joined_Abc, [0, 1], [4, 5])
    assert np.allclose(res2[0], A3)
    assert np.allclose(res2[1], b3)
    assert np.allclose(res2[2], c3)

    res3 = complex_gaussian_integral(join_Abc((A1, b1, c1), (A1, b1, c1)), [0, 1], [2, 3])
    assert np.allclose(res3[0], 0)
    assert np.allclose(res3[1], 0)
    assert np.allclose(res3[2], 1)

def test_complex_gaussian_integral_error():
    """Tests the error of the ``complex_gaussian_integral`` method."""
    A1, b1, c1 = triples.vacuum_state_Abc(2)
    A2, b2, c2 = triples.displacement_gate_Abc(x=[0.1, 0.2], y=0.3)

    with pytest.raises(ValueError, match="idx_z and idx_zconj must have the same length"):
        complex_gaussian_integral(join_Abc((A1, b1, c1), (A2, b2, c2)), [0, 1], [4,])

def test_contract_two_Abc():
    """Tests the error of the ``contract_two_Abc`` method."""
    A1, b1, c1 = triples.vacuum_state_Abc(2)
    A2, b2, c2 = triples.displacement_gate_Abc(x=[0.1, 0.2], y=0.3)

    res1 = contract_two_Abc((A1, b1, c1), (A2, b2, c2), [], [])
    assert np.allclose(res1[0], math.block_diag(A1, A2))
    assert np.allclose(res1[1], [0, 0, 0.1+0.3j, 0.2+0.3j, -0.1+0.3j, -0.2+0.3j])
    assert np.allclose(res1[2], c1 * c2)

    res2 = contract_two_Abc((A1, b1, c1), (A2, b2, c2), [0, 1], [2, 3])
    assert np.allclose(res2[0], math.zeros((2, 2)))
    assert np.allclose(res2[1], [0.1+0.3j, 0.2+0.3j])
    assert np.allclose(res2[2], c1 * c2)

    res3 = contract_two_Abc((A1, b1, c1), (A2, b2, c2), [0, 1], [0, 1])
    assert np.allclose(res3[0], math.zeros((2, 2)))
    assert np.allclose(res3[1], [-0.1+0.3j, -0.2+0.3j])
    assert np.allclose(res3[2], c1 * c2)
