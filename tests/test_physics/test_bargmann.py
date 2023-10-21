import numpy as np
from mrmustard.lab import Attenuator, Dgate, Gaussian, Ggate
from mrmustard.physics.bargmann import (
    wigner_to_bargmann_Choi,
    wigner_to_bargmann_psi,
    wigner_to_bargmann_rho,
    wigner_to_bargmann_U,
    contract_two_Abc,
    reorder_abc,
)
from mrmustard.physics.bargmann_repr import Bargmann
from mrmustard.math import Math

math = Math()


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


def test_abc_contraction_2mode_psi_U():
    "tests that the abc contraction works for U|psi>"
    psi = Gaussian(2)
    U = Ggate(2)

    A1, b1, c1 = psi.bargmann()  # out1ket, out2ket
    A2, b2, c2 = U.bargmann()  # out1ket, out2ket, in1ket, in2ket

    A_abc, b_abc, c_abc = contract_two_Abc((A1, b1, c1), (A2, b2, c2), (0, 1), (2, 3))
    A_mm, b_mm, c_mm = (psi >> U).bargmann()

    assert np.allclose(A_abc, A_mm)
    assert np.allclose(b_abc, b_mm)
    assert np.allclose(abs(c_abc), abs(c_mm))


def test_abc_contraction_2mode_rho_phi():
    "tests that the abc contraction works for rho >> phi"
    rho = Gaussian(2) >> Attenuator([0.1, 0.2]) >> Ggate(2) >> Attenuator([0.4, 0.9])
    phi = Ggate(2) >> Attenuator([0.3, 0.4]) >> Ggate(2)
    # out1bra, out2bra, out1ket, out2ket
    A1, b1, c1 = rho.bargmann()
    # out1bra, out2bra, in1bra, in2bra, out1ket, out2ket, in1ket, in2ket
    A2, b2, c2 = phi.bargmann()

    A_abc, b_abc, c_abc = contract_two_Abc((A1, b1, c1), (A2, b2, c2), (0, 1, 2, 3), (2, 3, 6, 7))

    A_mm, b_mm, c_mm = (rho >> phi).bargmann()

    assert np.allclose(A_abc, A_mm)
    assert np.allclose(b_abc, b_mm)
    assert np.allclose(c_abc, c_mm)


def test_abc_contraction_3mode_rho_2mode_U():
    "tests that the abc contraction works for U rho U_dagger"
    rho = Gaussian(3) >> Attenuator([0.1, 0.2, 0.4]) >> Ggate(3) >> Attenuator([0.4, 0.5, 0.9])
    U = Ggate(2)

    # out1bra, out2bra, out3bra, out1ket, out2ket, out3ket
    A1, b1, c1 = rho.bargmann()
    # out1ket, out2ket, in1ket, in2ket
    A2, b2, c2 = U.bargmann()

    A_abc, b_abc, c_abc = contract_two_Abc(
        (A2, b2, c2), (A1, b1, c1), (2, 3), (4, 5)
    )  # left in out1ket_U, out2ket_U, out1bra_rho, out2bra_rho, out3bra_rho, out1ket_rho

    A_abc, b_abc, c_abc = contract_two_Abc(
        (A_abc, b_abc, c_abc), (math.conj(A2), math.conj(b2), math.conj(c2)), (3, 4), (2, 3)
    )  # left in out1ket_U, out2ket_U, out1bra_rho, out1ket_rho, out1bra_U, out2bra_U

    A_abc, b_abc, c_abc = reorder_abc((A_abc, b_abc, c_abc), (2, 4, 5, 3, 0, 1))

    A_mm, b_mm, c_mm = (rho >> U[1, 2]).bargmann()

    assert np.allclose(A_abc, A_mm)
    assert np.allclose(b_abc, b_mm)
    assert np.allclose(c_abc, c_mm)


def test_Bargmann_2mode_psi_U():
    psi = Gaussian(2)
    U = Ggate(2)

    A1, b1, c1 = psi.bargmann()  # out1ket, out2ket
    A2, b2, c2 = U.bargmann()  # out1ket, out2ket, in1ket, in2ket

    Abc1 = Bargmann(A1, b1, c1)
    Abc2 = Bargmann(A2, b2, c2)

    psiU = Abc1[0, 1] @ Abc2[2, 3]

    A_abc, b_abc, c_abc = psiU.A[0], psiU.b[0], psiU.c[0]
    A_mm, b_mm, c_mm = (psi >> U).bargmann()

    assert np.allclose(A_abc, A_mm)
    assert np.allclose(b_abc, b_mm)
    assert np.allclose(abs(c_abc), abs(c_mm))
