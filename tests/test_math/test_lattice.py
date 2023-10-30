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

import pytest
import numpy as np

from mrmustard.lab import Gaussian
from mrmustard import settings
from mrmustard.math.lattice.strategies.binomial import binomial, binomial_dict

original_precision = settings.PRECISION_BITS_HERMITE_POLY


@pytest.mark.parametrize("precision", settings._allowed_precision_bits_hermite_poly)
def test_vanillaNumba_vs_binomial(precision):
    """Test that the vanilla method and the binomial method give the same result.
    Test is repeated for all possible values of PRECISION_BITS_HERMITE_POLY."""

    settings.PRECISION_BITS_HERMITE_POLY = precision
    G = Gaussian(2)

    ket_vanilla = G.ket(cutoffs=[10, 10])[:5, :5]
    ket_binomial = G.ket(max_photons=10)[:5, :5]

    assert np.allclose(ket_vanilla, ket_binomial)

    settings.PRECISION_BITS_HERMITE_POLY = original_precision

def test_binomial_vs_binomialDict():
    """Test that binomial and binomial_dict give the same result."""

    A, b, c = Gaussian(2).bargmann(numpy=True)
    max_prob = 0.9
    local_cutoffs = (10,10)
    global_cutoff = 15

    G, norm = binomial(local_cutoffs, A, b, c.item(), max_prob, global_cutoff)
    D = binomial_dict(local_cutoffs, A, b, c.item(), max_prob, global_cutoff)

    for idx in D.keys():
        assert np.isclose(D[idx],G[idx])

