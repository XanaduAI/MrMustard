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

"""Tests for mm_einsum mechanics: batch dimensions, output types, shapes, and structure."""

import numpy as np
import pytest

from mrmustard import math, settings
from mrmustard.lab import GaussianKet, Sgate, SqueezedVacuum
from mrmustard.physics.ansatz import ArrayAnsatz, PolyExpAnsatz
from mrmustard.physics.mm_einsum import mm_einsum, to_fock
from mrmustard.physics.utils import random_Abc


class TestMmEinsumMechanics:
    """Tests for mm_einsum mechanics: batch dimensions, output types, shapes."""

    def test_array_ansatz_to_fock_conversion(self):
        """Test that to_fock works correctly when given an ArrayAnsatz."""
        g = PolyExpAnsatz(*random_Abc(2))
        result = to_fock(to_fock(g, (10, 12)), (8, 10))
        assert isinstance(result, ArrayAnsatz)
        assert result.array.shape == (8, 10)

    @pytest.mark.parametrize(
        "output_spec,expected_batch_shape,expected_array_shape",
        [
            ("HJzw", (2, 3), (2, 3, 5, 5)),
            ("JHzw", (3, 2), (3, 2, 5, 5)),
            ("(JH)zw", (6,), (6, 5, 5)),
            ("(HJ)zw", (6,), (6, 5, 5)),
        ],
    )
    def test_batch_alignment(self, output_spec, expected_batch_shape, expected_array_shape):
        """Test batch alignment and reordering with different batch letters."""
        g2 = PolyExpAnsatz(*random_Abc(1, batch=(2,)))
        g3 = PolyExpAnsatz(*random_Abc(1, batch=(3,)))
        u01 = PolyExpAnsatz(*random_Abc(4))

        res = mm_einsum(
            f"Hx,Jy,zwxy->{output_spec}", g2, g3, u01, fock_dims={"x": 5, "y": 5, "z": 5, "w": 5}
        )

        assert isinstance(res, ArrayAnsatz)
        assert res.batch_shape == expected_batch_shape
        assert res.array.shape == expected_array_shape

    def test_batch_broadcast_basic(self):
        """Test batch broadcasting between different batch dimensions."""
        g1 = SqueezedVacuum(0, [0.1, 0.2], 0.4).ansatz
        g2 = SqueezedVacuum(1, 0.3, 0.6).ansatz

        res = mm_einsum("Hx,y->Hxy", g1, g2, fock_dims={"x": 5, "y": 5})
        assert isinstance(res, ArrayAnsatz)
        assert res.batch_shape == (2,)
        assert res.array.shape == (2, 5, 5)

    @pytest.mark.parametrize(
        "test_case,equation,expected_batch_shape",
        [
            ("with_scalar", "x,Hy->Hxy", (2,)),
            ("empty_batch", "x,x->", ()),
        ],
    )
    def test_batch_broadcasting_scenarios(self, test_case, equation, expected_batch_shape):
        """Test various batch broadcasting scenarios."""
        if test_case == "with_scalar":
            g1 = PolyExpAnsatz(*random_Abc(1))
            g2 = PolyExpAnsatz(*random_Abc(1, batch=(2,)))
            fock_dims = {"x": 5, "y": 5}
        else:
            g1 = PolyExpAnsatz(*random_Abc(1))
            g2 = PolyExpAnsatz(*random_Abc(1))
            fock_dims = {"x": 1}

        res = mm_einsum(equation, g1, g2, fock_dims=fock_dims)
        assert isinstance(res, ArrayAnsatz)
        assert res.batch_shape == expected_batch_shape

    def test_batch_dimension_contraction(self):
        """Contracting over batch dimensions should work correctly."""
        g1 = SqueezedVacuum(0, [0.1, 0.2, 0.3], 0.4).ansatz
        g2 = SqueezedVacuum(1, [0.1, 0.2, 0.3], 0.6).ansatz

        res = mm_einsum("Hx,Hy->Hxy", g1, g2, fock_dims={"x": 5, "y": 5})
        assert isinstance(res, ArrayAnsatz)
        assert res.batch_shape == (3,)
        assert res.array.shape == (3, 5, 5)

    def test_batch_reordering(self):
        ans = PolyExpAnsatz(*random_Abc(0, batch=(1, 2)))
        res = mm_einsum("AB->BA", ans)
        assert res.batch_shape == (2, 1)
        assert res.num_CV_vars == 0

    def test_batch_reordering_polyexp_result(self):
        """Test that batch dimensions are reordered correctly in PolyExpAnsatz results."""
        a0 = PolyExpAnsatz(*random_Abc(1, batch=(2,)))
        a1 = PolyExpAnsatz(*random_Abc(1, batch=(3,)))
        res = mm_einsum("Hx,Jy->JHxy", a0, a1)
        assert isinstance(res, PolyExpAnsatz)
        assert res.batch_shape == (3, 2)

    def test_complex_grouping_with_lin_sup(self):
        """Test grouping that includes linear superposition dimension."""
        g1 = (GaussianKet.random([0]) + GaussianKet.random([0])).ansatz
        g2 = SqueezedVacuum(1, [0.1, 0.2], 0.4).ansatz

        res = mm_einsum("Lx,Hy->(LH)xy", g1, g2, fock_dims={"x": 3, "y": 3})
        assert isinstance(res, ArrayAnsatz)
        assert res.batch_shape == (4,)
        assert res.array.shape == (4, 3, 3)

    def test_contraction_removes_all_batch_dims(self):
        """Test that contracting over all batch dimensions works."""
        g = PolyExpAnsatz(*random_Abc(1, batch=(3,)))

        res = mm_einsum("Hx,Hx->", g, g, fock_dims={"x": 1})
        assert isinstance(res, ArrayAnsatz)
        assert res.batch_shape == ()

    def test_core_reordering_polyexp_result(self):
        """Test that core dimensions are reordered correctly in PolyExpAnsatz results."""
        g = PolyExpAnsatz(*random_Abc(2))

        res = mm_einsum("xy->yx", g)
        assert isinstance(res, PolyExpAnsatz)
        assert res.num_CV_vars == 2

    def test_equation_with_empty_spaces(self):
        """Should handle equations with extra spaces gracefully."""
        a = PolyExpAnsatz(*random_Abc(1))
        res = mm_einsum("x, x->  x", a, a, fock_dims={"x": 5})
        assert isinstance(res, ArrayAnsatz)

    def test_grouping_non_consecutive_error(self):
        """Non-consecutive groups should still work (letters are consecutive in string)."""
        H, J = 2, 3
        array = settings.get_rng().random((H, J, 5))
        f = ArrayAnsatz(array, batch_dims=2)

        res = mm_einsum("HJx->(HJ)x", f)
        assert isinstance(res, ArrayAnsatz)
        assert res.batch_shape == (H * J,)

    def test_high_dimension_arrays(self):
        """Test with high-dimensional core arrays."""
        g = PolyExpAnsatz(*random_Abc(5))
        res = mm_einsum("abcde->abcde", g, fock_dims={"a": 3, "b": 3, "c": 3, "d": 3, "e": 3})
        assert isinstance(res, ArrayAnsatz)
        assert res.array.shape == (3, 3, 3, 3, 3)

    def test_identity_contraction_path(self):
        """Test that empty path with identical states works correctly."""
        g = PolyExpAnsatz(*random_Abc(1))

        res = mm_einsum("x,x->", g, g.conj, fock_dims={"x": 10})
        assert isinstance(res, ArrayAnsatz)
        assert res.batch_shape == ()

    def test_lin_sup_batch_dims_after_fock_conversion(self):
        """Test that batch_dims is correctly adjusted when lin_sup is summed during Fock conversion."""
        g = PolyExpAnsatz(*random_Abc(1, batch=(2,)), lin_sup=True)
        res = mm_einsum("Lx->x", g, fock_dims={"x": 3})
        assert isinstance(res, ArrayAnsatz)
        assert res.batch_dims == 0
        assert res.batch_shape == ()
        assert res.array.shape == (3,)

    def test_lin_sup_with_non_lin_sup(self):
        ans1 = PolyExpAnsatz(*random_Abc(1, batch=(3, 2)), lin_sup=True)
        ans2 = PolyExpAnsatz(*random_Abc(2, batch=(4,)), lin_sup=False)

        res = mm_einsum("ABa,Cab->(AC)Bb", ans1, ans2)
        assert res.batch_shape == (12, 2)
        assert res.num_CV_vars == 1
        assert res._lin_sup is True

    def test_linear_superposition_preserved(self):
        """Test that linear superposition is correctly preserved."""
        g1 = GaussianKet.random([0])
        g2 = (g1 + g1).ansatz

        res = mm_einsum("Lx->Lx", g2)
        assert isinstance(res, PolyExpAnsatz)
        assert res._lin_sup is True
        assert res.batch_shape == (2,)

    def test_large_batch_dimensions(self):
        """Test with many batch dimensions."""
        g1 = SqueezedVacuum(0, [0.1, 0.2], 0.4).ansatz
        g2 = SqueezedVacuum(1, [0.3, 0.4, 0.5], 0.6).ansatz
        g3 = SqueezedVacuum(2, [0.6, 0.7], 0.8).ansatz

        res = mm_einsum("Hx,Jy,Kz->HJKxyz", g1, g2, g3, fock_dims={"x": 3, "y": 3, "z": 3})
        assert isinstance(res, ArrayAnsatz)
        assert res.batch_shape == (2, 3, 2)
        assert res.array.shape == (2, 3, 2, 3, 3, 3)

    def test_large_batch_product(self):
        """Test with many batch dimensions creating large products."""
        array1 = settings.get_rng().random((2, 3, 4, 5))
        array2 = settings.get_rng().random((6, 7, 5))
        f1 = ArrayAnsatz(array1, batch_dims=3)
        f2 = ArrayAnsatz(array2, batch_dims=2)

        res = mm_einsum("HJKx,ABx->(HJKAB)x", f1, f2)
        assert isinstance(res, ArrayAnsatz)
        assert res.batch_shape == (2 * 3 * 4 * 6 * 7,)

    def test_mixed_polyexp_array_no_fock_dims(self):
        """Mixed PolyExpAnsatz and ArrayAnsatz should work even without fock_dims for arrays."""
        g = PolyExpAnsatz(*random_Abc(1))
        f = GaussianKet.random([1]).to_fock().ansatz

        res = mm_einsum("x,y->xy", g, f, fock_dims={"x": 10})
        assert isinstance(res, ArrayAnsatz)
        assert res.array.shape == (10, f.array.shape[0])

    def test_multiple_grouped_batches(self):
        """Multiple parenthesized groups should work correctly."""
        H, J, K, L = 2, 3, 4, 5
        f1 = ArrayAnsatz(settings.get_rng().random((H, J, 7)), batch_dims=2)
        f2 = ArrayAnsatz(settings.get_rng().random((K, L, 7)), batch_dims=2)
        res = mm_einsum("HJx,KLx->(HJ)(KL)", f1, f2)
        assert isinstance(res, ArrayAnsatz)
        assert res.batch_shape == (H * J, K * L)
        assert res.array.shape == (H * J, K * L)

    def test_no_common_indices_no_path(self):
        """PolyExpAnsatz with no common indices and empty path should not contract."""
        g0 = PolyExpAnsatz(*random_Abc(1))
        g1 = PolyExpAnsatz(*random_Abc(1))

        res = mm_einsum("x,y->xy", g0, g1, fock_dims={"x": 5, "y": 5})
        assert isinstance(res, ArrayAnsatz)
        assert res.array.shape == (5, 5)

    def test_output_no_core_only_batch(self):
        """Output with only batch dimensions and no core dimensions."""
        g1 = PolyExpAnsatz(*random_Abc(1))
        g2 = PolyExpAnsatz(*random_Abc(1))

        res = mm_einsum("Hx,Jy->HJ", g1 + g1, g2 + g2, fock_dims={"x": 1, "y": 1})
        assert isinstance(res, ArrayAnsatz)
        assert res.batch_shape == (2, 2)
        assert res.array.shape == (2, 2)
        assert res.core_shape == ()

    def test_output_parentheses_grouping_arrays(self):
        """Output string parentheses collapse multiple batch dims."""
        H, J, a, b, c = 2, 3, 5, 7, 11
        array1 = settings.get_rng().random((H, a, b))
        array2 = settings.get_rng().random((J, b, c))
        f1 = ArrayAnsatz(array1, batch_dims=1)
        f2 = ArrayAnsatz(array2, batch_dims=1)

        res = mm_einsum("Hab,Jbc->(HJ)ac", f1, f2)
        assert isinstance(res, ArrayAnsatz)
        assert res.batch_shape == (H * J,)
        assert res.array.shape == (H * J, a, c)

    @pytest.mark.parametrize(
        "equation,expected_cv_vars,expected_batch_shape",
        [
            ("Aab,BCcd->BACacbd", 4, (3, 1, 2)),
            ("Aab,BCab->BAC", 0, (3, 1, 2)),
        ],
    )
    def test_outer_product(self, equation, expected_cv_vars, expected_batch_shape):
        """Test outer product with batch dimensions."""
        s0 = Sgate(0, np.array([0.1])).ansatz
        s1 = Sgate(0, np.array([[0.1, 0.2]]), np.array([[0.1], [0.2], [0.3]])).ansatz

        res = mm_einsum(equation, s0, s1)

        assert isinstance(res, PolyExpAnsatz)
        assert res.num_CV_vars == expected_cv_vars
        assert res.batch_shape == expected_batch_shape

    def test_path_types(self):
        """Test different path types: LA, SSA, and UA."""
        a = PolyExpAnsatz(*random_Abc(1))
        b = PolyExpAnsatz(*random_Abc(4))
        res_la = mm_einsum(
            "x,y,zwxy->zw",
            a,
            a,
            b,
            contraction_path=[(0, 1), (0, 1)],
            path_type="LA",
            fock_dims={"x": 10, "y": 10, "z": 10, "w": 10},
        )
        res_ssa = mm_einsum(
            "x,y,zwxy->zw",
            a,
            a,
            b,
            contraction_path=[(0, 1), (2, 3)],
            path_type="SSA",
            fock_dims={"x": 10, "y": 10, "z": 10, "w": 10},
        )
        res_ua = mm_einsum(
            "x,y,zwxy->zw",
            a,
            a,
            b,
            contraction_path=[(0, 1), (0, 2)],
            path_type="UA",
            fock_dims={"x": 10, "y": 10, "z": 10, "w": 10},
        )
        assert res_la == res_ssa == res_ua

    def test_polyexp_to_fock_conversion(self):
        """PolyExp + Fock -> convert to arrays and contract."""
        g0 = GaussianKet.random([0])
        f0 = GaussianKet.random([0]).to_fock()
        res = mm_einsum(
            "x,x->",
            g0.ansatz,
            f0.ansatz.conj,
            fock_dims={"x": f0.auto_shape()[0]},
        )
        assert isinstance(res, ArrayAnsatz)

    def test_preserve_lin_sup_without_fock_conversion(self):
        """Linear superposition preserved in output without Fock conversion should stay PolyExpAnsatz."""
        g = (GaussianKet.random([0]) + GaussianKet.random([0])).ansatz

        res = mm_einsum("Lx->Lx", g)
        assert isinstance(res, PolyExpAnsatz)
        assert res.batch_shape == (2,)
        assert res._lin_sup is True

    def test_raw_array_with_batch_dims(self):
        """Test raw numpy arrays get batch_dims from equation."""
        array = settings.get_rng().random((2, 3, 4))

        res = mm_einsum("Hab->Hab", array)
        assert isinstance(res, ArrayAnsatz)
        assert res.batch_shape == (2,)
        assert res.core_shape == (3, 4)

    def test_raw_numpy_array_input(self):
        """Test that raw NumPy arrays are automatically wrapped."""
        array = settings.get_rng().random((5, 6))
        res = mm_einsum("ab,bc->ac", array, array.T)
        assert isinstance(res, ArrayAnsatz)
        expected = array @ array.T
        assert np.allclose(res.array, expected)

    def test_scalar_polyexp_result(self):
        """Test scalar PolyExpAnsatz result from contraction."""
        a = PolyExpAnsatz(*random_Abc(1))
        res = mm_einsum("x,x->", a, a)
        assert isinstance(res, PolyExpAnsatz)
        assert res.num_CV_vars == 0

    def test_single_letter_in_parentheses(self):
        """Single letter in parentheses should be ignored (no grouping)."""
        f = ArrayAnsatz(settings.get_rng().random((2, 5)), batch_dims=1)
        res = mm_einsum("Hx->(H)x", f)
        assert isinstance(res, ArrayAnsatz)
        assert res.batch_shape == (2,)
        assert res.array.shape == (2, 5)

    def test_single_mode_fock_with_batch(self):
        """Single-mode fock with batch dims (arrays)."""
        f = GaussianKet.random([0]).to_fock()
        batched = ArrayAnsatz(np.array([f.fock_array(), f.fock_array()]), batch_dims=1)
        res = mm_einsum("Hx,x->H", batched, f.ansatz.conj)
        assert isinstance(res, ArrayAnsatz)
        assert res.batch_shape == (2,)
        assert math.allclose(res.scalar, f.contract(f.dual).ansatz.scalar)

    def test_single_mode_fock_with_double_batch(self):
        """Fock arrays with two batch dims broadcasting (arrays)."""
        array0123 = settings.get_rng().random((3, 4, 5, 6))
        array012 = settings.get_rng().random((3, 5, 6))
        f0123 = ArrayAnsatz(array0123, batch_dims=2)
        f012 = ArrayAnsatz(array012, batch_dims=1)
        f1 = mm_einsum("HWxy,Hxy->W", f0123, f012)
        assert isinstance(f1, ArrayAnsatz)
        assert f1.batch_shape == (4,)

    def test_single_mode_with_batch(self):
        """Single mode with batch dims (array result)."""
        a = PolyExpAnsatz(*random_Abc(1))
        g = PolyExpAnsatz(*random_Abc(1, batch=(2,)))
        res = mm_einsum("Hx,x->H", g, a, contraction_path=[(0, 1)])
        assert isinstance(res, PolyExpAnsatz)
        assert res.batch_shape == (2,)

    def test_single_operand_identity(self):
        """Test single operand identity transformation to Fock."""
        a = PolyExpAnsatz(*random_Abc(1))
        res = mm_einsum("x->x", a, fock_dims={"x": 10})
        assert isinstance(res, ArrayAnsatz)
        assert res.array.shape == (10,)

    def test_single_operand_with_path(self):
        """Test single operand with empty path."""
        g = PolyExpAnsatz(*random_Abc(1))
        res = mm_einsum("x->x", g, fock_dims={"x": 10})
        assert isinstance(res, ArrayAnsatz)

    def test_three_batch_dims_with_grouping(self):
        """Grouping with three batch dimensions."""
        H, J, K = 2, 3, 4
        array = settings.get_rng().random((H, J, K, 5))
        f = ArrayAnsatz(array, batch_dims=3)

        res = mm_einsum("HJKx->(HJK)x", f)
        assert isinstance(res, ArrayAnsatz)
        assert res.batch_shape == (H * J * K,)
        assert res.array.shape == (H * J * K, 5)

    def test_three_polyexp_outer_product(self):
        """Test outer product with three PolyExpAnsatz operands."""
        a = PolyExpAnsatz(*random_Abc(1))
        res = mm_einsum("x,y,z->xyz", a, a, a, fock_dims={"x": 5, "y": 5, "z": 5})
        assert isinstance(res, ArrayAnsatz)
        assert res.array.shape == (5, 5, 5)

    @pytest.mark.parametrize(
        "equation,fock_dims,result_type,exp_batch_shape",
        [
            ("Lx,Lx->", {"x": 1}, ArrayAnsatz, ()),
            ("Lx,Lx->", None, PolyExpAnsatz, ()),
            ("Lx,Lx->L", {"x": 1}, ArrayAnsatz, (2,)),
            ("Lx,Lx->L", None, PolyExpAnsatz, (2,)),
            ("Lx,Kx->(LK)", {"x": 1}, ArrayAnsatz, (4,)),
            ("Lx,Kx->(LK)", None, PolyExpAnsatz, (4,)),
        ],
    )
    def test_with_linear_superposition(self, equation, fock_dims, result_type, exp_batch_shape):
        """Test linear superposition handling in various cases."""
        a = PolyExpAnsatz(*random_Abc(1, batch=(2,)))
        res = mm_einsum(equation, a, a, fock_dims=fock_dims)
        assert isinstance(res, result_type)
        assert res.batch_shape == exp_batch_shape

    def test_zero_dimensional_batch(self):
        """Test with no batch dimensions at all."""
        g = PolyExpAnsatz(*random_Abc(1))
        res = mm_einsum("x,x->x", g, g, fock_dims={"x": 1})
        assert isinstance(res, ArrayAnsatz)
        assert res.batch_shape == ()
        assert res.array.shape == (1,)
