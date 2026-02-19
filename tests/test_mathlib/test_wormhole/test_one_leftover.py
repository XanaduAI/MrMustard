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

"""Tests for the wormhole_1leftover_dm and wormhole_1leftover_ket functions.

These tests verify that the wormhole algorithm produces identical results
to computing the full Fock tensor and slicing.
"""

import numpy as np
import pytest

from mrmustard import math
from mrmustard.mathlib.lattice.strategies.wormhole import (
    wormhole_1leftover_dm,
    wormhole_1leftover_ket,
)
from mrmustard.physics.utils import random_Abc


class TestWormhole1LeftoverKetAgainstFullTensor:
    """
    Tests comparing Ket wormhole results against full tensor computation.

    These verify that the Ket wormhole produces mathematically correct results.
    """

    @pytest.mark.parametrize("pnr", [(0,), (1,), (2,), (5,)])
    def test_2mode_ket_matches_full_tensor(self, pnr):
        """Wormhole produces same result as slicing full tensor for 2-mode Ket."""
        A, b, c = random_Abc(core_vars=2, seed=42)
        leftover_cutoff = 10

        # Wormhole result
        results = wormhole_1leftover_ket(
            A,
            b,
            c,
            output_cutoff=leftover_cutoff,
            pnr_outcomes=[pnr],
            leftover_mode=0,
        )
        wh_ket = results[pnr]

        # Reference: compute full tensor and slice
        # Shape: (leftover_cutoff+1, pnr+1) for mode 0 leftover
        full_shape = (leftover_cutoff + 1, pnr[0] + 1)
        full_tensor = math.hermite_renormalized(A, b, c, shape=full_shape, stable=True)
        ref_ket = np.array(full_tensor[:, pnr[0]])

        assert np.allclose(wh_ket, ref_ket, atol=1e-10), (
            f"Wormhole result differs from reference at PNR={pnr}\n"
            f"Max abs diff: {np.max(np.abs(wh_ket - ref_ket))}"
        )

    @pytest.mark.parametrize("leftover_mode", [0, 1])
    def test_different_leftover_modes_ket(self, leftover_mode):
        """Wormhole works correctly for different leftover modes in Kets."""
        A, b, c = random_Abc(core_vars=2, seed=123)
        leftover_cutoff = 8
        pnr = (3,)

        results = wormhole_1leftover_ket(
            A,
            b,
            c,
            output_cutoff=leftover_cutoff,
            pnr_outcomes=[pnr],
            leftover_mode=leftover_mode,
        )
        wh_ket = results[pnr]

        # Reference with correct mode ordering
        if leftover_mode == 0:
            full_shape = (leftover_cutoff + 1, pnr[0] + 1)
            full_tensor = math.hermite_renormalized(A, b, c, shape=full_shape, stable=True)
            ref_ket = np.array(full_tensor[:, pnr[0]])
        else:
            full_shape = (pnr[0] + 1, leftover_cutoff + 1)
            full_tensor = math.hermite_renormalized(A, b, c, shape=full_shape, stable=True)
            ref_ket = np.array(full_tensor[pnr[0], :])

        assert np.allclose(wh_ket, ref_ket, atol=1e-10)

    def test_3mode_ket(self):
        """Wormhole works for 3-mode Kets."""
        A, b, c = random_Abc(core_vars=3, seed=456)
        leftover_cutoff = 6
        pnr = (1, 2)  # Measure modes 1 and 2

        results = wormhole_1leftover_ket(
            A,
            b,
            c,
            output_cutoff=leftover_cutoff,
            pnr_outcomes=[pnr],
            leftover_mode=0,
        )
        wh_ket = results[pnr]

        # Reference: shape is (cutoff, pnr1+1, pnr2+1)
        full_shape = (leftover_cutoff + 1, pnr[0] + 1, pnr[1] + 1)
        full_tensor = math.hermite_renormalized(A, b, c, shape=full_shape, stable=True)
        ref_ket = np.array(full_tensor[:, pnr[0], pnr[1]])

        assert np.allclose(wh_ket, ref_ket, atol=1e-10)

    def test_multiple_pnr_outcomes_ket(self):
        """Wormhole computes multiple PNR outcomes correctly for Kets."""
        A, b, c = random_Abc(core_vars=2, seed=789)
        leftover_cutoff = 8
        pnr_outcomes = [(0,), (1,), (3,)]

        results = wormhole_1leftover_ket(
            A,
            b,
            c,
            output_cutoff=leftover_cutoff,
            pnr_outcomes=pnr_outcomes,
            leftover_mode=0,
        )

        # Verify each requested outcome
        for pnr in pnr_outcomes:
            assert pnr in results

            # Reference
            full_shape = (leftover_cutoff + 1, pnr[0] + 1)
            full_tensor = math.hermite_renormalized(A, b, c, shape=full_shape, stable=True)
            ref_ket = np.array(full_tensor[:, pnr[0]])

            assert np.allclose(results[pnr], ref_ket, atol=1e-10), f"Failed at PNR={pnr}"

    def test_output_shape_correct_ket(self):
        """Output Ket arrays have correct shape."""
        A, b, c = random_Abc(core_vars=2, seed=666)
        leftover_cutoff = 12
        pnr = (5,)

        results = wormhole_1leftover_ket(
            A,
            b,
            c,
            output_cutoff=leftover_cutoff,
            pnr_outcomes=[pnr],
            leftover_mode=0,
        )

        expected_shape = (leftover_cutoff + 1,)
        assert results[pnr].shape == expected_shape


class TestWormhole1LeftoverDMAgainstFullTensor:
    """
    Tests comparing wormhole results against full tensor computation.

    These are the most important tests - they verify that the wormhole
    algorithm produces mathematically correct results.
    """

    @pytest.mark.parametrize("pnr", [(0,), (1,), (2,), (5,)])
    def test_2mode_dm_matches_full_tensor(self, pnr):
        """Wormhole produces same result as slicing full tensor for 2-mode DM."""
        A, b, c = random_Abc(core_vars=4, seed=42)
        leftover_cutoff = 10

        # Wormhole result
        results = wormhole_1leftover_dm(
            A,
            b,
            c,
            output_cutoff=leftover_cutoff,
            pnr_outcomes=[pnr],
            leftover_mode=0,
        )
        wh_dm = results[pnr]

        # Reference: compute full tensor and slice
        # Shape: (cutoff, pnr+1, cutoff, pnr+1) for mode 0 leftover
        # Note: MrMustard DM ordering is [bra_0, bra_1, ket_0, ket_1]
        full_shape = (leftover_cutoff + 1, pnr[0] + 1, leftover_cutoff + 1, pnr[0] + 1)
        full_tensor = math.hermite_renormalized(A, b, c, shape=full_shape, stable=True)
        ref_dm = np.array(full_tensor[:, pnr[0], :, pnr[0]])

        assert np.allclose(wh_dm, ref_dm, atol=1e-10), (
            f"Wormhole result differs from reference at PNR={pnr}\n"
            f"Max abs diff: {np.max(np.abs(wh_dm - ref_dm))}"
        )

    @pytest.mark.parametrize("leftover_mode", [0, 1])
    def test_different_leftover_modes(self, leftover_mode):
        """Wormhole works correctly for different leftover modes."""
        A, b, c = random_Abc(core_vars=4, seed=123)
        leftover_cutoff = 8
        pnr = (3,)

        results = wormhole_1leftover_dm(
            A,
            b,
            c,
            output_cutoff=leftover_cutoff,
            pnr_outcomes=[pnr],
            leftover_mode=leftover_mode,
        )
        wh_dm = results[pnr]

        # Reference with correct mode ordering
        if leftover_mode == 0:
            # Leftover is mode 0, measured is mode 1
            full_shape = (leftover_cutoff + 1, pnr[0] + 1, leftover_cutoff + 1, pnr[0] + 1)
            full_tensor = math.hermite_renormalized(A, b, c, shape=full_shape, stable=True)
            ref_dm = np.array(full_tensor[:, pnr[0], :, pnr[0]])
        else:
            # Leftover is mode 1, measured is mode 0
            full_shape = (pnr[0] + 1, leftover_cutoff + 1, pnr[0] + 1, leftover_cutoff + 1)
            full_tensor = math.hermite_renormalized(A, b, c, shape=full_shape, stable=True)
            ref_dm = np.array(full_tensor[pnr[0], :, pnr[0], :])

        assert np.allclose(wh_dm, ref_dm, atol=1e-10)

    def test_3mode_dm(self):
        """Wormhole works for 3-mode density matrices."""
        A, b, c = random_Abc(core_vars=6, seed=456)
        leftover_cutoff = 6
        pnr = (1, 2)  # Measure modes 1 and 2

        results = wormhole_1leftover_dm(
            A,
            b,
            c,
            output_cutoff=leftover_cutoff,
            pnr_outcomes=[pnr],
            leftover_mode=0,
        )
        wh_dm = results[pnr]

        # Reference: shape is (cutoff, pnr1+1, pnr2+1, cutoff, pnr1+1, pnr2+1)
        full_shape = (
            leftover_cutoff + 1,
            pnr[0] + 1,
            pnr[1] + 1,
            leftover_cutoff + 1,
            pnr[0] + 1,
            pnr[1] + 1,
        )
        full_tensor = math.hermite_renormalized(A, b, c, shape=full_shape, stable=True)
        ref_dm = np.array(full_tensor[:, pnr[0], pnr[1], :, pnr[0], pnr[1]])

        assert np.allclose(wh_dm, ref_dm, atol=1e-10)

    def test_multiple_pnr_outcomes(self):
        """Wormhole computes multiple PNR outcomes correctly."""
        A, b, c = random_Abc(core_vars=4, seed=789)
        leftover_cutoff = 8
        pnr_outcomes = [(0,), (1,), (3,)]

        results = wormhole_1leftover_dm(
            A,
            b,
            c,
            output_cutoff=leftover_cutoff,
            pnr_outcomes=pnr_outcomes,
            leftover_mode=0,
        )

        # Verify each requested outcome
        for pnr in pnr_outcomes:
            assert pnr in results

            # Reference
            full_shape = (leftover_cutoff + 1, pnr[0] + 1, leftover_cutoff + 1, pnr[0] + 1)
            full_tensor = math.hermite_renormalized(A, b, c, shape=full_shape, stable=True)
            ref_dm = np.array(full_tensor[:, pnr[0], :, pnr[0]])

            assert np.allclose(results[pnr], ref_dm, atol=1e-10), f"Failed at PNR={pnr}"


class TestWormhole1LeftoverEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_origin_at_zero(self):
        """Algorithm handles origin at (0,0,...) correctly.

        This is a known failure case for the Gray code version which uses solve
        instead of pinv. The backward matrix becomes singular at the origin.
        """
        A, b, c = random_Abc(core_vars=4, seed=111)
        pnr = (0,)  # Origin case
        leftover_cutoff = 5

        results = wormhole_1leftover_dm(
            A,
            b,
            c,
            output_cutoff=leftover_cutoff,
            pnr_outcomes=[pnr],
            leftover_mode=0,
        )

        # Should not raise and should match reference
        full_shape = (leftover_cutoff + 1, 1, leftover_cutoff + 1, 1)
        full_tensor = math.hermite_renormalized(A, b, c, shape=full_shape, stable=True)
        ref_dm = np.array(full_tensor[:, 0, :, 0])

        assert np.allclose(results[pnr], ref_dm, atol=1e-10)

    def test_custom_first_hypercube_pnr(self):
        """Starting from non-zero PNR works correctly."""
        A, b, c = random_Abc(core_vars=4, seed=222)
        leftover_cutoff = 5
        pnr_outcomes = [(3,), (4,)]
        first_pnr = (2,)  # Start at PNR=2 instead of 0

        results = wormhole_1leftover_dm(
            A,
            b,
            c,
            output_cutoff=leftover_cutoff,
            pnr_outcomes=pnr_outcomes,
            leftover_mode=0,
            first_hypercube_pnr=first_pnr,
        )

        # Verify requested outcomes
        for pnr in pnr_outcomes:
            full_shape = (leftover_cutoff + 1, pnr[0] + 1, leftover_cutoff + 1, pnr[0] + 1)
            full_tensor = math.hermite_renormalized(A, b, c, shape=full_shape, stable=True)
            ref_dm = np.array(full_tensor[:, pnr[0], :, pnr[0]])
            assert np.allclose(results[pnr], ref_dm, atol=1e-10)

    def test_invalid_first_hypercube_pnr_raises(self):
        """Raises error if first_hypercube_pnr exceeds targets."""
        A, b, c = random_Abc(core_vars=4, seed=333)

        with pytest.raises(ValueError, match="cannot exceed"):
            wormhole_1leftover_dm(
                A,
                b,
                c,
                output_cutoff=5,
                pnr_outcomes=[(2,)],
                leftover_mode=0,
                first_hypercube_pnr=(3,),  # Exceeds target
            )

    def test_empty_pnr_outcomes_raises_dm(self):
        """Raises error if pnr_outcomes is empty for DM."""
        A, b, c = random_Abc(core_vars=4, seed=334)

        with pytest.raises(ValueError, match="cannot be empty"):
            wormhole_1leftover_dm(
                A,
                b,
                c,
                output_cutoff=5,
                pnr_outcomes=[],
                leftover_mode=0,
            )

    def test_empty_pnr_outcomes_raises_ket(self):
        """Raises error if pnr_outcomes is empty for Ket."""
        A, b, c = random_Abc(core_vars=2, seed=335)

        with pytest.raises(ValueError, match="cannot be empty"):
            wormhole_1leftover_ket(
                A,
                b,
                c,
                output_cutoff=5,
                pnr_outcomes=[],
                leftover_mode=0,
            )

    def test_last_mode_as_leftover(self):
        """Using -1 for leftover_mode correctly selects last mode."""
        A, b, c = random_Abc(core_vars=4, seed=444)
        leftover_cutoff = 5
        pnr = (2,)

        # Using -1 should be equivalent to using 1 for 2 modes
        results_neg1 = wormhole_1leftover_dm(
            A,
            b,
            c,
            output_cutoff=leftover_cutoff,
            pnr_outcomes=[pnr],
            leftover_mode=-1,
        )
        results_1 = wormhole_1leftover_dm(
            A,
            b,
            c,
            output_cutoff=leftover_cutoff,
            pnr_outcomes=[pnr],
            leftover_mode=1,
        )

        assert np.allclose(results_neg1[pnr], results_1[pnr], atol=1e-14)


class TestWormholeDisplacementCases:
    """Tests for various displacement scenarios."""

    def test_zero_displacement_dm(self):
        """Wormhole works correctly with zero displacement (pure squeezed/thermal state)."""
        # Create thermal state with no displacement
        num_modes = 2
        n_bar = np.array([0.5, 1.0])  # Thermal populations

        A = np.zeros((2 * num_modes, 2 * num_modes), dtype=complex)
        for i in range(num_modes):
            tanh_r = n_bar[i] / (n_bar[i] + 1)
            A[i, i + num_modes] = tanh_r
            A[i + num_modes, i] = tanh_r

        # Zero displacement
        b = np.zeros(2 * num_modes, dtype=complex)
        c = 1.0  # No displacement normalization

        leftover_cutoff = 8
        pnr = (3,)

        results = wormhole_1leftover_dm(
            A,
            b,
            c,
            output_cutoff=leftover_cutoff,
            pnr_outcomes=[pnr],
            leftover_mode=0,
        )

        # Reference
        full_shape = (leftover_cutoff + 1, pnr[0] + 1, leftover_cutoff + 1, pnr[0] + 1)
        full_tensor = math.hermite_renormalized(A, b, c, shape=full_shape, stable=True)
        ref_dm = np.array(full_tensor[:, pnr[0], :, pnr[0]])

        assert np.allclose(results[pnr], ref_dm, atol=1e-10)

    def test_zero_displacement_ket(self):
        """Wormhole works correctly with zero displacement for Kets."""
        # Create squeezed vacuum (no displacement)
        num_modes = 2
        A = np.zeros((num_modes, num_modes), dtype=complex)
        A[0, 0] = 0.3  # Squeezing on mode 0
        A[1, 1] = 0.2  # Squeezing on mode 1

        b = np.zeros(num_modes, dtype=complex)  # Zero displacement
        c = 1.0

        leftover_cutoff = 8
        pnr = (2,)

        results = wormhole_1leftover_ket(
            A,
            b,
            c,
            output_cutoff=leftover_cutoff,
            pnr_outcomes=[pnr],
            leftover_mode=0,
        )

        # Reference
        full_shape = (leftover_cutoff + 1, pnr[0] + 1)
        full_tensor = math.hermite_renormalized(A, b, c, shape=full_shape, stable=True)
        ref_ket = np.array(full_tensor[:, pnr[0]])

        assert np.allclose(results[pnr], ref_ket, atol=1e-10)

    def test_large_displacement_dm(self):
        """Wormhole works correctly with larger displacements."""
        num_modes = 2
        n_bar = np.array([0.5, 0.8])

        A = np.zeros((2 * num_modes, 2 * num_modes), dtype=complex)
        for i in range(num_modes):
            tanh_r = n_bar[i] / (n_bar[i] + 1)
            A[i, i + num_modes] = tanh_r
            A[i + num_modes, i] = tanh_r

        # Larger displacement (but still reasonable)
        alpha = np.array([1.5 + 0.5j, 1.0 - 0.8j])
        b = np.zeros(2 * num_modes, dtype=complex)
        for i in range(num_modes):
            b[i] = alpha[i]
            b[i + num_modes] = np.conj(alpha[i])
        c = np.exp(-0.5 * np.sum(np.abs(alpha) ** 2))

        leftover_cutoff = 10
        pnr = (4,)

        results = wormhole_1leftover_dm(
            A,
            b,
            c,
            output_cutoff=leftover_cutoff,
            pnr_outcomes=[pnr],
            leftover_mode=0,
        )

        # Reference
        full_shape = (leftover_cutoff + 1, pnr[0] + 1, leftover_cutoff + 1, pnr[0] + 1)
        full_tensor = math.hermite_renormalized(A, b, c, shape=full_shape, stable=True)
        ref_dm = np.array(full_tensor[:, pnr[0], :, pnr[0]])

        assert np.allclose(results[pnr], ref_dm, atol=1e-10)

    def test_displacement_only_dm(self):
        """Wormhole works correctly with displacement-only states (no squeezing/correlation)."""
        num_modes = 2

        # Identity A matrix (no squeezing/correlation)
        A = np.zeros((2 * num_modes, 2 * num_modes), dtype=complex)

        # Displacement only
        alpha = np.array([0.8 + 0.3j, 0.5 - 0.2j])
        b = np.zeros(2 * num_modes, dtype=complex)
        for i in range(num_modes):
            b[i] = alpha[i]
            b[i + num_modes] = np.conj(alpha[i])
        c = np.exp(-0.5 * np.sum(np.abs(alpha) ** 2))

        leftover_cutoff = 8
        pnr = (2,)

        results = wormhole_1leftover_dm(
            A,
            b,
            c,
            output_cutoff=leftover_cutoff,
            pnr_outcomes=[pnr],
            leftover_mode=0,
        )

        # Reference
        full_shape = (leftover_cutoff + 1, pnr[0] + 1, leftover_cutoff + 1, pnr[0] + 1)
        full_tensor = math.hermite_renormalized(A, b, c, shape=full_shape, stable=True)
        ref_dm = np.array(full_tensor[:, pnr[0], :, pnr[0]])

        assert np.allclose(results[pnr], ref_dm, atol=1e-10)


class TestWormholeComplexBranching:
    """Tests for complex branching scenarios with multiple PNR outcomes."""

    def test_far_apart_pnr_outcomes_dm(self):
        """Wormhole handles PNR outcomes that are far apart, requiring significant traversal."""
        A, b, c = random_Abc(core_vars=4, seed=1000)
        leftover_cutoff = 10

        # Outcomes far apart: requires traversing from 0 to 1, then to 8
        pnr_outcomes = [(0,), (1,), (8,)]

        results = wormhole_1leftover_dm(
            A,
            b,
            c,
            output_cutoff=leftover_cutoff,
            pnr_outcomes=pnr_outcomes,
            leftover_mode=0,
        )

        # Verify each outcome
        for pnr in pnr_outcomes:
            assert pnr in results

            full_shape = (leftover_cutoff + 1, pnr[0] + 1, leftover_cutoff + 1, pnr[0] + 1)
            full_tensor = math.hermite_renormalized(A, b, c, shape=full_shape, stable=True)
            ref_dm = np.array(full_tensor[:, pnr[0], :, pnr[0]])

            assert np.allclose(results[pnr], ref_dm, atol=1e-10), f"Failed at PNR={pnr}"

    def test_far_apart_pnr_outcomes_ket(self):
        """Wormhole handles far-apart PNR outcomes for Kets."""
        A, b, c = random_Abc(core_vars=2, seed=1001)
        leftover_cutoff = 10

        pnr_outcomes = [(0,), (2,), (7,)]

        results = wormhole_1leftover_ket(
            A,
            b,
            c,
            output_cutoff=leftover_cutoff,
            pnr_outcomes=pnr_outcomes,
            leftover_mode=0,
        )

        for pnr in pnr_outcomes:
            assert pnr in results

            full_shape = (leftover_cutoff + 1, pnr[0] + 1)
            full_tensor = math.hermite_renormalized(A, b, c, shape=full_shape, stable=True)
            ref_ket = np.array(full_tensor[:, pnr[0]])

            assert np.allclose(results[pnr], ref_ket, atol=1e-10), f"Failed at PNR={pnr}"

    def test_multimode_branching_dm(self):
        """Wormhole handles complex branching with multiple modes and diverse PNR outcomes."""
        A, b, c = random_Abc(core_vars=6, seed=1002)
        leftover_cutoff = 8

        # Multiple outcomes with different patterns - tests branching in 2D PNR space
        pnr_outcomes = [
            (1, 2),  # Start here
            (3, 1),  # Branch in first dimension
            (1, 5),  # Branch in second dimension
            (4, 3),  # Requires traversal in both dimensions
            (2, 2),  # Backtrack scenario
        ]

        results = wormhole_1leftover_dm(
            A,
            b,
            c,
            output_cutoff=leftover_cutoff,
            pnr_outcomes=pnr_outcomes,
            leftover_mode=0,
        )

        for pnr in pnr_outcomes:
            assert pnr in results

            full_shape = (
                leftover_cutoff + 1,
                pnr[0] + 1,
                pnr[1] + 1,
                leftover_cutoff + 1,
                pnr[0] + 1,
                pnr[1] + 1,
            )
            full_tensor = math.hermite_renormalized(A, b, c, shape=full_shape, stable=True)
            ref_dm = np.array(full_tensor[:, pnr[0], pnr[1], :, pnr[0], pnr[1]])

            assert np.allclose(results[pnr], ref_dm, atol=1e-10), f"Failed at PNR={pnr}"

    def test_multimode_branching_ket(self):
        """Wormhole handles complex branching for multi-mode Kets."""
        A, b, c = random_Abc(core_vars=3, seed=1003)
        leftover_cutoff = 8

        # Diverse outcomes testing branching
        pnr_outcomes = [
            (0, 1),
            (2, 0),
            (1, 4),
            (3, 2),
            (0, 3),
        ]

        results = wormhole_1leftover_ket(
            A,
            b,
            c,
            output_cutoff=leftover_cutoff,
            pnr_outcomes=pnr_outcomes,
            leftover_mode=0,
        )

        for pnr in pnr_outcomes:
            assert pnr in results

            full_shape = (leftover_cutoff + 1, pnr[0] + 1, pnr[1] + 1)
            full_tensor = math.hermite_renormalized(A, b, c, shape=full_shape, stable=True)
            ref_ket = np.array(full_tensor[:, pnr[0], pnr[1]])

            assert np.allclose(results[pnr], ref_ket, atol=1e-10), f"Failed at PNR={pnr}"

    def test_non_sequential_branching_dm(self):
        """Wormhole handles non-sequential PNR outcomes that create complex tree structure."""
        A, b, c = random_Abc(core_vars=4, seed=1004)
        leftover_cutoff = 10

        # Non-sequential outcomes: (0,) -> (5,) -> (2,) -> (8,)
        # This tests that the tree correctly handles backtracking and branching
        pnr_outcomes = [(0,), (5,), (2,), (8,), (1,), (6,)]

        results = wormhole_1leftover_dm(
            A,
            b,
            c,
            output_cutoff=leftover_cutoff,
            pnr_outcomes=pnr_outcomes,
            leftover_mode=0,
        )

        for pnr in pnr_outcomes:
            assert pnr in results

            full_shape = (leftover_cutoff + 1, pnr[0] + 1, leftover_cutoff + 1, pnr[0] + 1)
            full_tensor = math.hermite_renormalized(A, b, c, shape=full_shape, stable=True)
            ref_dm = np.array(full_tensor[:, pnr[0], :, pnr[0]])

            assert np.allclose(results[pnr], ref_dm, atol=1e-10), f"Failed at PNR={pnr}"

    def test_many_outcomes_branching(self):
        """Wormhole efficiently handles many PNR outcomes with shared paths."""
        A, b, c = random_Abc(core_vars=4, seed=1005)
        leftover_cutoff = 8

        # Many outcomes that share common prefixes - tests tree efficiency
        pnr_outcomes = [(i,) for i in range(10)]  # 0 through 9

        results = wormhole_1leftover_dm(
            A,
            b,
            c,
            output_cutoff=leftover_cutoff,
            pnr_outcomes=pnr_outcomes,
            leftover_mode=0,
        )

        # Verify all outcomes
        for pnr in pnr_outcomes:
            assert pnr in results

            full_shape = (leftover_cutoff + 1, pnr[0] + 1, leftover_cutoff + 1, pnr[0] + 1)
            full_tensor = math.hermite_renormalized(A, b, c, shape=full_shape, stable=True)
            ref_dm = np.array(full_tensor[:, pnr[0], :, pnr[0]])

            assert np.allclose(results[pnr], ref_dm, atol=1e-10), f"Failed at PNR={pnr}"


class TestWormholeBatchedDM:
    """Tests for batched density matrix wormhole computation.

    Batched inputs (A with 3 dimensions) should produce identical results
    to running each triple through the single-input path independently.
    """

    def test_batched_dm_matches_single(self):
        """Batched DM computation matches individual single computations."""
        rng = np.random.default_rng(2001)
        A, b, c = random_Abc(core_vars=4, seed=2001)
        batch_size = 3
        leftover_cutoff = 6
        pnr = (2,)

        # Create batch with small perturbations to A, b, and c
        A_perturbations = rng.standard_normal((batch_size, *A.shape)) * 0.01
        b_perturbations = (
            rng.standard_normal((batch_size, *b.shape))
            + 1j * rng.standard_normal((batch_size, *b.shape))
        ) * 0.01
        c_perturbations = (
            rng.standard_normal(batch_size) + 1j * rng.standard_normal(batch_size)
        ) * 0.01

        A_batch = np.stack([A + A_perturbations[i] for i in range(batch_size)])
        b_batch = np.stack([b + b_perturbations[i] for i in range(batch_size)])
        c_batch = np.array([c + c_perturbations[i] for i in range(batch_size)])

        results_batched = wormhole_1leftover_dm(
            A_batch,
            b_batch,
            c_batch,
            output_cutoff=leftover_cutoff,
            pnr_outcomes=[pnr],
            leftover_mode=0,
        )

        # Compare each batch element against single computation
        for i in range(batch_size):
            results_single = wormhole_1leftover_dm(
                A_batch[i],
                b_batch[i],
                c_batch[i],
                output_cutoff=leftover_cutoff,
                pnr_outcomes=[pnr],
                leftover_mode=0,
            )
            assert np.allclose(results_batched[pnr][i], results_single[pnr], atol=1e-10), (
                f"Batched result differs from single at batch index {i}"
            )

    def test_batched_dm_output_shape(self):
        """Batched DM output has correct shape (batch, cutoff+1, cutoff+1)."""
        A, b, c = random_Abc(core_vars=4, seed=2002)
        batch_size = 4
        leftover_cutoff = 5
        pnr = (3,)

        A_batch = np.stack([A] * batch_size)
        b_batch = np.stack([b] * batch_size)
        c_batch = np.array([c] * batch_size)

        results = wormhole_1leftover_dm(
            A_batch,
            b_batch,
            c_batch,
            output_cutoff=leftover_cutoff,
            pnr_outcomes=[pnr],
            leftover_mode=0,
        )

        expected_shape = (batch_size, leftover_cutoff + 1, leftover_cutoff + 1)
        assert results[pnr].shape == expected_shape

    def test_batched_dm_multiple_pnr_outcomes(self):
        """Batched DM handles multiple PNR outcomes correctly."""
        A, b, c = random_Abc(core_vars=4, seed=2003)
        batch_size = 2
        leftover_cutoff = 5
        pnr_outcomes = [(1,), (3,)]

        A_batch = np.stack([A] * batch_size)
        b_batch = np.stack([b] * batch_size)
        c_batch = np.array([c] * batch_size)

        results_batched = wormhole_1leftover_dm(
            A_batch,
            b_batch,
            c_batch,
            output_cutoff=leftover_cutoff,
            pnr_outcomes=pnr_outcomes,
            leftover_mode=0,
        )

        for pnr in pnr_outcomes:
            assert pnr in results_batched
            # Compare against single
            results_single = wormhole_1leftover_dm(
                A,
                b,
                c,
                output_cutoff=leftover_cutoff,
                pnr_outcomes=[pnr],
                leftover_mode=0,
            )
            for i in range(batch_size):
                assert np.allclose(results_batched[pnr][i], results_single[pnr], atol=1e-10), (
                    f"Failed at PNR={pnr}, batch={i}"
                )

    def test_batched_dm_empty_pnr_raises(self):
        """Batched DM raises error for empty pnr_outcomes."""
        A, b, c = random_Abc(core_vars=4, seed=2004)
        A_batch = np.stack([A, A])
        b_batch = np.stack([b, b])
        c_batch = np.array([c, c])

        with pytest.raises(ValueError, match="cannot be empty"):
            wormhole_1leftover_dm(
                A_batch,
                b_batch,
                c_batch,
                output_cutoff=5,
                pnr_outcomes=[],
                leftover_mode=0,
            )

    def test_batched_dm_3mode(self):
        """Batched DM works for 3-mode states."""
        A, b, c = random_Abc(core_vars=6, seed=2005)
        batch_size = 2
        leftover_cutoff = 4
        pnr = (1, 2)

        A_batch = np.stack([A] * batch_size)
        b_batch = np.stack([b] * batch_size)
        c_batch = np.array([c] * batch_size)

        results_batched = wormhole_1leftover_dm(
            A_batch,
            b_batch,
            c_batch,
            output_cutoff=leftover_cutoff,
            pnr_outcomes=[pnr],
            leftover_mode=0,
        )

        results_single = wormhole_1leftover_dm(
            A,
            b,
            c,
            output_cutoff=leftover_cutoff,
            pnr_outcomes=[pnr],
            leftover_mode=0,
        )

        for i in range(batch_size):
            assert np.allclose(results_batched[pnr][i], results_single[pnr], atol=1e-10)


class TestWormholeBatchedKet:
    """Tests for batched ket wormhole computation.

    Batched inputs (A with 3 dimensions) should produce identical results
    to running each triple through the single-input path independently.
    """

    def test_batched_ket_matches_single(self):
        """Batched Ket computation matches individual single computations."""
        rng = np.random.default_rng(3001)
        A, b, c = random_Abc(core_vars=2, seed=3001)
        batch_size = 3
        leftover_cutoff = 8
        pnr = (4,)

        # Create batch with small perturbations to A, b, and c
        A_perturbations = rng.standard_normal((batch_size, *A.shape)) * 0.01
        b_perturbations = (
            rng.standard_normal((batch_size, *b.shape))
            + 1j * rng.standard_normal((batch_size, *b.shape))
        ) * 0.01
        c_perturbations = (
            rng.standard_normal(batch_size) + 1j * rng.standard_normal(batch_size)
        ) * 0.01

        A_batch = np.stack([A + A_perturbations[i] for i in range(batch_size)])
        b_batch = np.stack([b + b_perturbations[i] for i in range(batch_size)])
        c_batch = np.array([c + c_perturbations[i] for i in range(batch_size)])

        results_batched = wormhole_1leftover_ket(
            A_batch,
            b_batch,
            c_batch,
            output_cutoff=leftover_cutoff,
            pnr_outcomes=[pnr],
            leftover_mode=0,
        )

        for i in range(batch_size):
            results_single = wormhole_1leftover_ket(
                A_batch[i],
                b_batch[i],
                c_batch[i],
                output_cutoff=leftover_cutoff,
                pnr_outcomes=[pnr],
                leftover_mode=0,
            )
            assert np.allclose(results_batched[pnr][i], results_single[pnr], atol=1e-10), (
                f"Batched result differs from single at batch index {i}"
            )

    def test_batched_ket_output_shape(self):
        """Batched Ket output has correct shape (batch, cutoff+1)."""
        A, b, c = random_Abc(core_vars=2, seed=3002)
        batch_size = 4
        leftover_cutoff = 7
        pnr = (3,)

        A_batch = np.stack([A] * batch_size)
        b_batch = np.stack([b] * batch_size)
        c_batch = np.array([c] * batch_size)

        results = wormhole_1leftover_ket(
            A_batch,
            b_batch,
            c_batch,
            output_cutoff=leftover_cutoff,
            pnr_outcomes=[pnr],
            leftover_mode=0,
        )

        expected_shape = (batch_size, leftover_cutoff + 1)
        assert results[pnr].shape == expected_shape

    def test_batched_ket_multiple_pnr_outcomes(self):
        """Batched Ket handles multiple PNR outcomes correctly."""
        A, b, c = random_Abc(core_vars=2, seed=3003)
        batch_size = 2
        leftover_cutoff = 6
        pnr_outcomes = [(0,), (2,), (5,)]

        A_batch = np.stack([A] * batch_size)
        b_batch = np.stack([b] * batch_size)
        c_batch = np.array([c] * batch_size)

        results_batched = wormhole_1leftover_ket(
            A_batch,
            b_batch,
            c_batch,
            output_cutoff=leftover_cutoff,
            pnr_outcomes=pnr_outcomes,
            leftover_mode=0,
        )

        for pnr in pnr_outcomes:
            assert pnr in results_batched
            results_single = wormhole_1leftover_ket(
                A,
                b,
                c,
                output_cutoff=leftover_cutoff,
                pnr_outcomes=[pnr],
                leftover_mode=0,
            )
            for i in range(batch_size):
                assert np.allclose(results_batched[pnr][i], results_single[pnr], atol=1e-10), (
                    f"Failed at PNR={pnr}, batch={i}"
                )

    def test_batched_ket_empty_pnr_raises(self):
        """Batched Ket raises error for empty pnr_outcomes."""
        A, b, c = random_Abc(core_vars=2, seed=3004)
        A_batch = np.stack([A, A])
        b_batch = np.stack([b, b])
        c_batch = np.array([c, c])

        with pytest.raises(ValueError, match="cannot be empty"):
            wormhole_1leftover_ket(
                A_batch,
                b_batch,
                c_batch,
                output_cutoff=5,
                pnr_outcomes=[],
                leftover_mode=0,
            )

    def test_batched_ket_3mode(self):
        """Batched Ket works for 3-mode states."""
        A, b, c = random_Abc(core_vars=3, seed=3005)
        batch_size = 2
        leftover_cutoff = 5
        pnr = (1, 2)

        A_batch = np.stack([A] * batch_size)
        b_batch = np.stack([b] * batch_size)
        c_batch = np.array([c] * batch_size)

        results_batched = wormhole_1leftover_ket(
            A_batch,
            b_batch,
            c_batch,
            output_cutoff=leftover_cutoff,
            pnr_outcomes=[pnr],
            leftover_mode=0,
        )

        results_single = wormhole_1leftover_ket(
            A,
            b,
            c,
            output_cutoff=leftover_cutoff,
            pnr_outcomes=[pnr],
            leftover_mode=0,
        )

        for i in range(batch_size):
            assert np.allclose(results_batched[pnr][i], results_single[pnr], atol=1e-10)


class TestWormholeInvalidInputs:
    """Tests for invalid input handling."""

    def test_dm_invalid_ndim_raises(self):
        """DM raises ValueError for A with wrong number of dimensions."""
        A_4d = np.zeros((2, 2, 4, 4), dtype=complex)
        b = np.zeros(4, dtype=complex)

        with pytest.raises(ValueError, match="2 or 3 dimensions"):
            wormhole_1leftover_dm(
                A_4d,
                b,
                1.0,
                output_cutoff=5,
                pnr_outcomes=[(1,)],
                leftover_mode=0,
            )

    def test_ket_invalid_ndim_raises(self):
        """Ket raises ValueError for A with wrong number of dimensions."""
        A_4d = np.zeros((2, 2, 2, 2), dtype=complex)
        b = np.zeros(2, dtype=complex)

        with pytest.raises(ValueError, match="2 or 3 dimensions"):
            wormhole_1leftover_ket(
                A_4d,
                b,
                1.0,
                output_cutoff=5,
                pnr_outcomes=[(1,)],
                leftover_mode=0,
            )
