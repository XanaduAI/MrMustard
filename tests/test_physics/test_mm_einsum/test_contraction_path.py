# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the standalone contraction path utilities."""

import pytest

from mrmustard.physics.mm_einsum.contraction_path import (
    normalize_path,
    ua_to_linear,
    validate_path,
)


class TestNormalizePath:
    """Tests for normalize_path."""

    def test_none_returns_none(self):
        assert normalize_path(None, "LA") is None

    def test_empty_returns_empty(self):
        assert normalize_path([], "LA") == []

    def test_la_passthrough(self):
        path = [(0, 1), (0, 1)]
        assert normalize_path(path, "LA") == [(0, 1), (0, 1)]

    def test_ua_to_linear(self):
        path = [(0, 2), (1, 3), (0, 3)]
        result = normalize_path(path, "UA")
        assert result == [(0, 2), (0, 1), (0, 1)]

    def test_invalid_path_type_raises(self):
        with pytest.raises(ValueError, match="Invalid path_type"):
            normalize_path([(0, 1)], "INVALID")


class TestValidatePath:
    """Tests for validate_path."""

    def test_none_does_not_raise(self):
        validate_path(None, 3)

    def test_single_operand_empty_ok(self):
        validate_path([], 1)
        validate_path([(0,)], 1)

    def test_single_operand_nonempty_raises(self):
        with pytest.raises(ValueError, match="contraction_path"):
            validate_path([(0, 1)], 1)

    def test_two_operands_exactly_one_step_ok(self):
        validate_path([(0, 1)], 2)

    def test_three_operands_exactly_two_steps_ok(self):
        validate_path([(0, 1), (0, 1)], 3)

    def test_four_operands_three_steps_ok(self):
        validate_path([(0, 1), (0, 1), (0, 1)], 4)

    def test_too_few_steps_raises(self):
        with pytest.raises(ValueError, match="expected 2 step"):
            validate_path([(0, 1)], 3)
        with pytest.raises(ValueError, match="expected 3 step"):
            validate_path([(0, 1), (0, 1)], 4)

    def test_too_many_steps_raises(self):
        with pytest.raises(ValueError, match="expected 1 step"):
            validate_path([(0, 1), (0, 1)], 2)


class TestUaToLinear:
    """Tests for ua_to_linear."""

    def test_two_operands(self):
        assert ua_to_linear([(0, 1)]) == [(0, 1)]

    def test_four_operands_union_assignment(self):
        path = [(0, 2), (1, 3), (0, 3)]
        result = ua_to_linear(path)
        assert len(result) == 3
        assert result == [(0, 2), (0, 1), (0, 1)]
