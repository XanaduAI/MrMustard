# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Unit tests for the :class:`Math`.
"""

import numpy as np
import pytest

from mrmustard import settings
import mrmustard.math as math


def test_backend_redirection_tf():
    """Test BackendManager class is redirecting calls to the desired backend"""
    n0 = math.backend.name

    math.change_backend("tensorflow")
    assert math.backend.name == "tensorflow"

    math.change_backend("numpy")
    assert math.backend.name == "numpy"

    n0 = math.backend.name


def test_hash_tensor():
    """Test hash of a tensor"""
    tensor = math.astensor([1, 2, 3])
    assert np.allclose(*[math.hash_tensor(tensor) for _ in range(3)])
