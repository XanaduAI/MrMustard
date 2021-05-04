import pytest
import numpy as np
from mrmustard.backends import HighLevelBackend

class TestBackend:

    backend = HighLevelBackend()

    def test_beamsplitter_identity(self):
        S = self.backend._beam_splitter_symplectic(0.0, 0.0)
        assert np.allclose(S, np.identity(4))

    def test_beamsplitter_symmetric(self):
        S = self.backend._beam_splitter_symplectic(np.pi/4, 0.0)
        D = np.array([[np.sqrt(1/2), -np.sqrt(1/2)],[np.sqrt(1/2),np.sqrt(1/2)]])
        O = np.zeros_like(D)
        assert np.allclose(S, np.block([[D,O],[O,D]]))

    