import numpy as np
import pytest

from mrmustard import math
from mrmustard.physics.representations import Bargmann

# ~~~~~~~~~
# Utilities
# ~~~~~~~~~


def Abc_triple(n: int):
    r""""""
    min_magnitude = 1e-9
    max_magnitude = 1

    # complex symmetric matrix A
    A = np.random.uniform(min_magnitude, max_magnitude, (n, n)) + 1.0j * np.random.uniform(
        min_magnitude, max_magnitude, (n, n)
    )
    A = 0.5 * (A + A.T)  # make it symmetric

    # complex vector b
    b = np.random.uniform(min_magnitude, max_magnitude, (n,))

    # complex scalar c
    c = np.random.uniform(min_magnitude, max_magnitude, (1,))

    return A, b, c


# ~~~~~
# Tests
# ~~~~~


class TestBargmannRepresentation:
    r"""
    Tests the init and other algebras related to Bargmann Representation.
    """

    Abc_n1 = Abc_triple(1)
    Abc_n2 = Abc_triple(2)
    Abc_n3 = Abc_triple(3)

    @pytest.mark.parametrize("triple", [Abc_n1, Abc_n2, Abc_n3])
    def test_init_non_batched(self, triple):
        A, b, c = triple
        bargmann = Bargmann(*triple)

        assert np.allclose(bargmann.A, A)
        assert np.allclose(bargmann.b, b)
        assert np.allclose(bargmann.c, c)

    @pytest.mark.parametrize("triple", [Abc_n1, Abc_n2, Abc_n3])
    def test_conj(self, triple):
        A, b, c = triple
        bargmann = Bargmann(*triple).conj()

        assert np.allclose(bargmann.A, math.conj(A))
        assert np.allclose(bargmann.b, math.conj(b))
        assert np.allclose(bargmann.c, math.conj(c))

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_and(self, n):
        triple1 = Abc_triple(n)
        triple2 = Abc_triple(n)

        bargmann = Bargmann(*triple1) & Bargmann(*triple2)

        assert bargmann.A.shape == (1, 2 * n, 2 * n)
        assert bargmann.b.shape == (1, 2 * n)
        assert bargmann.c.shape == (1,)

    @pytest.mark.parametrize("scalar", [0.5, 1.2])
    @pytest.mark.parametrize("triple", [Abc_n1, Abc_n2, Abc_n3])
    def test_mul_with_scalar(self, scalar, triple):
        bargmann1 = Bargmann(*triple)
        bargmann_mul = bargmann1 * scalar

        assert np.allclose(bargmann1.A, bargmann_mul.A)
        assert np.allclose(bargmann1.b, bargmann_mul.b)
        assert np.allclose(bargmann1.c * scalar, bargmann_mul.c)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_mul(self, n):
        triple1 = Abc_triple(n)
        triple2 = Abc_triple(n)

        bargmann1 = Bargmann(*triple1)
        bargmann2 = Bargmann(*triple2)
        bargmann_mul = bargmann1 * bargmann2

        assert np.allclose(bargmann_mul.A, bargmann1.A + bargmann2.A)
        assert np.allclose(bargmann_mul.b, bargmann1.b + bargmann2.b)
        assert np.allclose(bargmann_mul.c, bargmann1.c * bargmann2.c)

    @pytest.mark.parametrize("scalar", [0.5, 1.2])
    @pytest.mark.parametrize("triple", [Abc_n1, Abc_n2, Abc_n3])
    def test_div_with_scalar(self, scalar, triple):
        bargmann1 = Bargmann(*triple)
        bargmann_div = bargmann1 / scalar

        assert np.allclose(bargmann1.A, bargmann_div.A)
        assert np.allclose(bargmann1.b, bargmann_div.b)
        assert np.allclose(bargmann1.c / scalar, bargmann_div.c)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_div(self, n):
        triple1 = Abc_triple(n)
        triple2 = Abc_triple(n)

        bargmann1 = Bargmann(*triple1)
        bargmann2 = Bargmann(*triple2)
        bargmann_div = bargmann1 / bargmann2

        assert np.allclose(bargmann_div.A, bargmann1.A - bargmann2.A)
        assert np.allclose(bargmann_div.b, bargmann1.b - bargmann2.b)
        assert np.allclose(bargmann_div.c, bargmann1.c / bargmann2.c)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_add(self, n):
        triple1 = Abc_triple(n)
        triple2 = Abc_triple(n)

        bargmann1 = Bargmann(*triple1)
        bargmann2 = Bargmann(*triple2)
        bargmann_add = bargmann1 + bargmann2

        assert np.allclose(bargmann_add.A, math.concat([bargmann1.A, bargmann2.A], axis=0))
        assert np.allclose(bargmann_add.b, math.concat([bargmann1.b, bargmann2.b], axis=0))
        assert np.allclose(bargmann_add.c, math.concat([bargmann1.c, bargmann2.c], axis=0))

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_sub(self, n):
        triple1 = Abc_triple(n)
        triple2 = Abc_triple(n)

        bargmann1 = Bargmann(*triple1)
        bargmann2 = Bargmann(*triple2)
        bargmann_add = bargmann1 - bargmann2

        assert np.allclose(bargmann_add.A, math.concat([bargmann1.A, bargmann2.A], axis=0))
        assert np.allclose(bargmann_add.b, math.concat([bargmann1.b, bargmann2.b], axis=0))
        assert np.allclose(bargmann_add.c, math.concat([bargmann1.c, -bargmann2.c], axis=0))

    # def test_trace(self):
    #     bargmann = Bargmann(*Abc_triple(4)).trace([0], [2])
    #     assert np.allclose(bargmann.A.shape, (1, 2, 2))
    #     assert np.allclose(bargmann.b.shape, (1, 2))
    #     assert np.allclose(bargmann.c.shape, (1,))

    def test_reorder(self):
        triple = Abc_triple(3)
        bargmann = Bargmann(*triple).reorder((0, 2, 1))

        assert np.allclose(bargmann.A[0], triple[0][[0, 2, 1], :][:, [0, 2, 1]])
        assert np.allclose(bargmann.b[0], triple[1][[0, 2, 1]])
