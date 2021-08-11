import pytest
import numpy as np
np.random.seed(137)

from mrmustard import GaussianPlugin

gp = GaussianPlugin()

@pytest.mark.parametrize("num_modes", [1, 2, 3])
@pytest.mark.parametrize("hbar", [0.5, 1.0, 2.0])
def test_vacuum_state():
    cov, disp = gp.vacuum_state(num_modes, hbar)
    assert np.allclose(cov, np.eye(2*num_modes) * hbar/2)
    assert np.allclose(disp, np.zeros_like(disp))

# test coherent state single-mode
@pytest.mark.parametrize("hbar", [0.5, 1.0, 2.0])
def test_coherent_state_single(hbar):
    cov, disp = gp.coherent_state(0.4, 0.5, hbar)
    assert np.allclose(cov, np.eye(2*num_modes) * hbar/2)
    assert np.allclose(disp, np.array([0.4, 0.5]) * self._backend.sqrt(2 * hbar))

# test coherent state multi-mode
@pytest.mark.parametrize("hbar", [0.5, 1.0, 2.0])
def test_coherent_state_multi(hbar):
    cov, disp = gp.coherent_state([0.4, 0.6], [-0.3, 0.5], hbar)
    assert np.allclose(cov, np.eye(2*num_modes) * hbar/2)
    assert np.allclose(disp, np.array([0.4, 0.6, -0.3, 0.5]) * self._backend.sqrt(2 * hbar))