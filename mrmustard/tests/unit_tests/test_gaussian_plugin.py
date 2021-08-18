import pytest
import numpy as np
np.random.seed(137)

from mrmustard import GaussianPlugin

gp = GaussianPlugin()

@pytest.mark.parametrize("num_modes", [1, 2, 3])
@pytest.mark.parametrize("hbar", [0.5, 1.0, 2.0])
def test_vacuum_state(num_modes, hbar):
    cov, disp = gp.vacuum_state(num_modes, hbar)
    assert np.allclose(cov, np.eye(2*num_modes) * hbar/2)
    assert np.allclose(disp, np.zeros_like(disp))

# test coherent state single-mode
@pytest.mark.parametrize("hbar", [0.5, 1.0, 2.0])
def test_coherent_state_single(hbar):
    x, y = np.random.uniform(size=(2,1))
    cov, disp = gp.coherent_state(x, y, hbar)
    assert np.allclose(cov, np.eye(2) * hbar/2)
    assert np.allclose(disp, np.concatenate([x, y], axis=0) * np.sqrt(2 * hbar))

# test coherent state multi-mode
@pytest.mark.parametrize("num_modes", [1, 2, 3])
@pytest.mark.parametrize("hbar", [0.5, 1.0, 2.0])
def test_coherent_state_multi(num_modes, hbar):
    x, y = np.random.uniform(size=(2,num_modes))
    cov, disp = gp.coherent_state(x, y, hbar)
    assert np.allclose(cov, np.eye(2*num_modes) * hbar/2)
    assert np.allclose(disp, np.concatenate([x, y], axis=0) * np.sqrt(2 * hbar))

def test_the_purity_of_a_pure_state():
    from mrmustard import Coherent
    state = Coherent(0.5, 1.0)
    purity = gp.purity(state.cov, state.hbar)
    expected = 1.0
    assert np.allclose(purity, expected)

@pytest.mark.parametrize("nbar", [0.1, 1.0, 5.0])
@pytest.mark.parametrize("hbar", [0.5, 1.0, 2.0])
def test_the_purity_of_a_mixed_state(nbar, hbar):
    from mrmustard import Thermal
    state = Thermal(nbar, hbar)
    purity = gp.purity(state.cov, state.hbar)
    expected = 1/(2*nbar + 1)
    assert np.allclose(purity, expected)

