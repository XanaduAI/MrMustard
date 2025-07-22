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

from mrmustard.utils.argsort import argsort_gen


def test_argsort_gen():
    # Test with generators yielding ascending values
    gen_list = [iter(range(i, i + 3)) for i in range(5)]
    assert argsort_gen(gen_list) == [0, 1, 2, 3, 4]

    # Test with generators yielding descending values
    gen_list = [iter(range(i, i - 3, -1)) for i in range(5, 0, -1)]
    assert argsort_gen(gen_list) == [4, 3, 2, 1, 0]

    # Test with empty list
    gen_list = []
    assert argsort_gen(gen_list) == []

    # Test with single generator
    gen_list = [iter(range(3))]
    assert argsort_gen(gen_list) == [0]

    # Test with generators yielding the same first value
    gen_list = [iter(range(i, i + 3)) for i in range(5)]
    gen_list.append(iter(range(3)))  # Add another generator with the same first value
    assert argsort_gen(gen_list) == [0, 5, 1, 2, 3, 4]

    # Test with generators yielding the same values
    gen_list = [iter(range(3)) for _ in range(5)]
    assert argsort_gen(gen_list) == [0, 1, 2, 3, 4]
