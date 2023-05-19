from mrmustard.lab import *
from mrmustard.math import Math

math = Math()
import numpy as np

np.set_printoptions(suppress=True, linewidth=250)

gate = Rgate(1.0)

cutoffs = [50]
gaussian_state = SqueezedVacuum(-0.1)
fock_state = State(ket=gaussian_state.ket(cutoffs))

via_fock_space_dm = fock_state >> gate >> Attenuator(0.1)  # .run().dm([10])
