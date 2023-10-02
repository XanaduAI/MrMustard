# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the lattice module"""

import numpy as np

from mrmustard.lab import Gaussian
from mrmustard import settings


def test_vanillaNumba_vs_binomial():
    """Test that the vanilla method (Numba with precision of complex128) and the binomial method give the same result"""
    settings.precision_bits_hermite_poly = 128
    G = Gaussian(2)

    ket_vanilla = G.ket(cutoffs=[10, 10])[:5, :5]
    ket_binomial = G.ket(max_photons=10)[:5, :5]

    assert np.allclose(ket_vanilla, ket_binomial)


def test_vanillaJulia_vs_binomial():
    """Test that the vanilla method (Julia with precision of complex512) and the binomial method give the same result"""
    settings.precision_bits_hermite_poly = 512
    G = Gaussian(2)

    ket_vanilla = G.ket(cutoffs=[10, 10])[:5, :5]
    ket_binomial = G.ket(max_photons=10)[:5, :5]

    assert np.allclose(ket_vanilla, ket_binomial)
