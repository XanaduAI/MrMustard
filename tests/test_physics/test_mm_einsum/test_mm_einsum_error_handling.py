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

"""Error handling and input validation tests for mm_einsum."""

import pytest

from mrmustard import settings
from mrmustard.physics.ansatz import ArrayAnsatz, PolyExpAnsatz
from mrmustard.physics.mm_einsum import mm_einsum, to_fock
from mrmustard.physics.utils import random_Abc


class TestMmEinsumErrorHandling:
    """Tests for error handling and input validation in mm_einsum."""

    def test_conflicting_index_sizes_in_fock_dims(self):
        """Test that arrays with conflicting sizes are handled correctly."""
        f1 = ArrayAnsatz(settings.get_rng().random((2,)), batch_dims=0)
        f2 = ArrayAnsatz(settings.get_rng().random((3,)), batch_dims=0)
        res = mm_einsum("x,x->x", f1, f2)
        assert isinstance(res, ArrayAnsatz)
        assert res.array.shape == (2,)

    def test_dimension_validation_missing_batch(self):
        """Test validation when batch dimension is not labeled."""
        a = PolyExpAnsatz(*random_Abc(1, batch=(2,)))
        with pytest.raises(ValueError, match="has 1 batch dims but got 0 batch letters"):
            mm_einsum("x,x->", a, a)

    def test_dimension_validation_missing_core(self):
        """Test validation when core dimension is not labeled."""
        a = PolyExpAnsatz(*random_Abc(4))
        with pytest.raises(ValueError, match="missing fock_dims for {'d'}"):
            mm_einsum("abc,abcd->", a, a)

    def test_dimension_validation_too_many_core(self):
        """Test validation when too many core indices are provided."""
        a = PolyExpAnsatz(*random_Abc(1))
        with pytest.raises(ValueError, match="missing fock_dims for {'y'}"):
            mm_einsum("xy,x->", a, a)

    def test_empty_output_different_indices_errors(self):
        """PolyExpAnsatz with different indices and no fock_dims should error."""
        a = PolyExpAnsatz(*random_Abc(1))
        with pytest.raises(ValueError, match="Cannot convert PolyExpAnsatz to Fock"):
            mm_einsum("x,y->", a, a)

    def test_empty_parentheses_in_output(self):
        """Should handle empty parentheses in output."""
        f = ArrayAnsatz(settings.get_rng().random((2, 3)), batch_dims=2)
        res = mm_einsum("HJ->()HJ", f)
        assert isinstance(res, ArrayAnsatz)
        assert res.batch_shape == (2, 3)

    def test_equation_missing_arrow(self):
        """Test validation when equation is missing '->'."""
        a = PolyExpAnsatz(*random_Abc(1))
        with pytest.raises(ValueError, match="'->'"):
            mm_einsum("x,x", a, a)

    def test_equation_too_few_operands(self):
        """Test validation when too few operands are provided."""
        a = PolyExpAnsatz(*random_Abc(1))
        with pytest.raises(ValueError, match="Number of inputs must match"):
            mm_einsum("x,y->xy", a)

    def test_equation_too_many_operands(self):
        """Test validation when too many operands are provided."""
        a = PolyExpAnsatz(*random_Abc(1))
        with pytest.raises(ValueError, match="Number of inputs must match"):
            mm_einsum("x->x", a, a)

    def test_invalid_path_type(self):
        """Should raise error for invalid path_type."""
        a = PolyExpAnsatz(*random_Abc(1))
        result = mm_einsum("x,x->", a, a, path_type="INVALID")
        assert result is not None

    def test_invalid_ssa_path_structure(self):
        """Test SSA path with invalid indices."""
        a = PolyExpAnsatz(*random_Abc(1))
        with pytest.raises(IndexError, match="pop index out of range"):
            mm_einsum(
                "x,y->xy",
                a,
                a,
                contraction_path=[(10, 20)],
                path_type="SSA",
                fock_dims={"x": 5, "y": 5},
            )

    def test_lowercase_in_parentheses(self):
        """Test validation when lowercase letters are grouped in parentheses."""
        a = ArrayAnsatz(settings.get_rng().random((2, 3, 4)), batch_dims=1)
        with pytest.raises(ValueError, match="Lowercase letters cannot be grouped"):
            mm_einsum("Hab->H(ab)", a)

    def test_missing_fock_dims_comprehensive(self):
        """Test comprehensive error message for missing fock_dims."""
        a = PolyExpAnsatz(*random_Abc(1))
        with pytest.raises(ValueError, match=r"missing fock_dims for \{'[yz]', '[yz]'\}"):
            mm_einsum("x,y,z->xyz", a, a, a, fock_dims={"x": 5})

    def test_missing_fock_dims_for_conversion_single_mode(self):
        """Should raise error when PolyExpAnsatz can't be converted due to missing fock_dims."""
        a = PolyExpAnsatz(*random_Abc(1))
        with pytest.raises(ValueError, match="missing fock_dims for {'y'}"):
            mm_einsum("x,y->xy", a, a, fock_dims={"x": 5})

    def test_multiple_arrows_in_equation(self):
        """Should raise when equation has multiple arrows."""
        a = PolyExpAnsatz(*random_Abc(1))
        with pytest.raises(ValueError, match="'->'"):
            mm_einsum("x,x->y->z", a, a.conj, fock_dims={"x": 5, "y": 5, "z": 5})

    def test_nested_parentheses_validation(self):
        """Test validation of nested parentheses in output."""
        a = ArrayAnsatz(settings.get_rng().random((2, 3)), batch_dims=2)
        with pytest.raises(ValueError, match="Nested parentheses"):
            mm_einsum("HJ->((HJ))", a)

    def test_non_pair_tuples_in_path(self):
        """Should handle non-pair tuples in contraction path."""
        a = PolyExpAnsatz(*random_Abc(1))
        with pytest.raises(ValueError, match="too many values to unpack"):
            mm_einsum(
                "x,y,z->xyz",
                a,
                a,
                a,
                contraction_path=[(0, 1, 2)],
                fock_dims={"x": 5, "y": 5, "z": 5},
            )

    def test_out_of_range_path_indices(self):
        """Should raise error for path indices out of range."""
        a = PolyExpAnsatz(*random_Abc(1))
        with pytest.raises(IndexError):
            mm_einsum("x,y->xy", a, a, contraction_path=[(0, 5)], fock_dims={"x": 5, "y": 5})

    def test_unclosed_parentheses_validation(self):
        """Test validation of unclosed parentheses in output."""
        a = ArrayAnsatz(settings.get_rng().random((2, 3)), batch_dims=2)
        with pytest.raises(ValueError, match="Unclosed"):
            mm_einsum("HJ->(HJ", a)

    def test_validation_partial_batch_labels(self):
        """Fails when only some batch dimensions are labeled."""
        a = PolyExpAnsatz(*random_Abc(1, batch=(2,)))
        with pytest.raises(ValueError, match="has 1 batch dims but got 0 batch letters"):
            mm_einsum("Hx,x->H", a, a)

    def test_zero_fock_dimension_raises_error(self):
        """Test that zero fock dimensions raise errors."""
        a = PolyExpAnsatz(*random_Abc(2))
        with pytest.raises(ValueError, match="Fock space dimension is 0"):
            to_fock(a, (5, 0))
