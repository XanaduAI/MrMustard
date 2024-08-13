# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""tests for the Sauron state class"""

from mrmustard.lab_dev.states import Sauron, Number
import numpy as np


class TestSauron:
    r"""
    Tests for the ``Sauron`` state class.
    """

    def test_init(self):
        """Test that the Sauron state is initialized correctly."""
        state = Sauron([0], n=1)
        assert state.name == "Sauron-1"
        assert np.isclose(state.probability, 1.0)

    def test_equals_number_state(self):
        """Test that the Sauron state is equal to the corresponding number state."""
        assert np.isclose(Sauron([0], n=1, epsilon=0.1) >> Number([0], n=1).dual, 1.0)
        assert np.isclose(Sauron([0], n=2, epsilon=0.1) >> Number([0], n=2).dual, 1.0)
        assert np.isclose(Sauron([0], n=3, epsilon=0.1) >> Number([0], n=3).dual, 1.0)
