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

try:
    import torch

    torch_available = True
except ImportError:
    torch_available = False


def test_backend_redirection_tf():
    """Test Math class is redirecting calls to the backend set on MM settings"""
    settings.BACKEND = "tensorflow"
    assert math.backend.name == "tensorflow"


def test_error_for_wrong_backend():
    """Test error is raise when using a backend that is not allowed"""
    backend = settings.BACKEND
    with pytest.raises(ValueError) as exception_info:
        settings.BACKEND = "unexisting_backend"
        assert exception_info.value.args[0] == "Backend must be either 'tensorflow' or 'torch'"

    # set back to initial value to avoid side effects
    settings.BACKEND = backend


def test_hash_tensor():
    """Test hash of a tensor"""
    tensor = math.astensor([1, 2, 3])
    assert np.allclose(*[math.hash_tensor(tensor) for _ in range(3)])
