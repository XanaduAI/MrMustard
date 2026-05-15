# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for ``path.optimize``."""

from mrmustard.lab import Coherent, Number, Sgate
from mrmustard.path import optimal_path


def test_optimal_path():
    components = [Number(0, n=15), Sgate(0, r=1.0), Coherent(0, alpha=1.0).dual]
    path = optimal_path(components=components, with_BF_heuristic=True)  # with default heuristics
    assert path == [(1, 2), (0, 1)]

    path = optimal_path(components=components, with_BF_heuristic=False)  # without the BF heuristic
    assert path == [(1, 2), (0, 1)]

    path = optimal_path(components=components, n_init=1, verbose=False)
    assert path == [(1, 2), (0, 1)]
