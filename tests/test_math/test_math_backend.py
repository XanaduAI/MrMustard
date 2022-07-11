# Copyright 2021 Xanadu Quantum Technologies Inc.

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

import pytest
from mrmustard.math import Math
from mrmustard import settings

try:
    import torch
except ImportError:
    torch_available = False
else:
    torch_available = True


def test_backend_redirection_tf():
    """Test Math class is redirecting calls to the backend set on MM settings"""
    math = Math()

    settings.backend = "tensorflow"
    assert math._MathInterface__instance.__module__ == "mrmustard.math.tensorflow"


@pytest.mark.skipif(not torch_available, reason="Test only works if Torch is installed")
def test_backend_redirection_torch():
    """Test Math class is redirecting calls to the backend set on MM settings"""
    math = Math()

    settings.backend = "torch"
    assert math._MathInterface__instance.__module__ == "mrmustard.math.torch"


def test_error_for_wrong_backend():
    """Test error is raise when using a backend that is not allowed"""
    with pytest.raises(ValueError) as exception_info:
        settings.backend = "unexisiting_backend"
        assert exception_info.value.args[0] == f"Backend must be either 'tensorflow' or 'torch'"
