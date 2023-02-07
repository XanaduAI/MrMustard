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

"""Test related to visualization on MrMustard."""

from mrmustard.lab import Coherent
from mrmustard.utils.graphics import mikkel_plot


def test_mikkel_plot():
    """Tests that mikkel plot returns figure and axes."""
    dm = Coherent().dm(cutoffs=[10])
    fig, axs = mikkel_plot(dm.numpy())

    assert fig is not None
    assert axs is not None
