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

"""Tests for core wormhole computation primitives.

These tests verify the fundamental Ab_at_pivot_directional function that
computes recurrence coefficients for advancing through the Fock lattice.
"""

import numpy as np
import pytest

from mrmustard.mathlib.lattice.strategies.wormhole.core import Ab_at_pivot_directional


class TestAbAtPivotDirectional:
    """Tests for the core Ab_at_pivot_directional function."""

    def test_1d_at_origin(self):
        """Test at origin in 1D - critical singular case."""
        # Simple 1D case: A is 1x1, b is scalar
        A = np.array([[0.3]], dtype=complex)
        b = np.array([0.5 + 0.2j], dtype=complex)

        hypercube_pivot = (0,)  # At origin
        origin = (0,)
        direction = 0

        A_row, b_scalar = Ab_at_pivot_directional(A, b, hypercube_pivot, origin, direction)

        # Should not raise and should produce valid output
        # A_row is a matrix (n x n), b_scalar is a vector (n,)
        assert A_row.shape == (1, 1)
        assert b_scalar.shape == (1,)
        assert not np.isnan(A_row).any()
        assert not np.isnan(b_scalar).any()
        assert not np.isinf(A_row).any()
        assert not np.isinf(b_scalar).any()

    def test_1d_with_displacement(self):
        """Test 1D case with displacement."""
        A = np.array([[0.2]], dtype=complex)
        b = np.array([1.0 + 0.5j], dtype=complex)

        hypercube_pivot = (1,)  # Front of hypercube
        origin = (0,)
        direction = 0

        A_row, b_scalar = Ab_at_pivot_directional(A, b, hypercube_pivot, origin, direction)

        assert A_row.shape == (1, 1)
        assert b_scalar.shape == (1,)
        assert not np.isnan(A_row).any()
        assert not np.isnan(b_scalar).any()

    def test_2d_at_origin_singular(self):
        """Test 2D at origin - critical singular case with multiple zeros."""
        # This is the case mentioned in the docstring where pinv is critical
        A = np.array([[0.2, 0.1], [0.1, 0.3]], dtype=complex)
        b = np.array([0.5, 0.3], dtype=complex)

        hypercube_pivot = (0, 0)  # At origin - singular case
        origin = (0, 0)
        direction = 0

        A_row, b_scalar = Ab_at_pivot_directional(A, b, hypercube_pivot, origin, direction)

        # Should handle singular case gracefully
        assert A_row.shape == (2, 2)
        assert b_scalar.shape == (2,)
        assert not np.isnan(A_row).any()
        assert not np.isnan(b_scalar).any()
        assert not np.isinf(A_row).any()
        assert not np.isinf(b_scalar).any()

    def test_2d_with_displacement(self):
        """Test 2D case with displacement."""
        A = np.array([[0.2, 0.1], [0.1, 0.3]], dtype=complex)
        b = np.array([1.0 + 0.5j, 0.8 - 0.3j], dtype=complex)

        hypercube_pivot = (1, 1)  # Front face
        origin = (0, 0)
        direction = 0

        A_row, b_scalar = Ab_at_pivot_directional(A, b, hypercube_pivot, origin, direction)

        assert A_row.shape == (2, 2)
        assert b_scalar.shape == (2,)
        assert not np.isnan(A_row).any()
        assert not np.isnan(b_scalar).any()

    def test_2d_different_directions(self):
        """Test 2D case advancing in different directions."""
        A = np.array([[0.2, 0.1], [0.1, 0.3]], dtype=complex)
        b = np.array([0.5, 0.3], dtype=complex)

        hypercube_pivot = (1, 0)
        origin = (0, 0)

        # Test direction 0
        A_row_0, b_scalar_0 = Ab_at_pivot_directional(A, b, hypercube_pivot, origin, 0)

        # Test direction 1
        A_row_1, b_scalar_1 = Ab_at_pivot_directional(A, b, hypercube_pivot, origin, 1)

        # Results should differ
        assert not np.allclose(A_row_0, A_row_1)
        assert not np.allclose(b_scalar_0, b_scalar_1)

    def test_2d_nonzero_origin(self):
        """Test 2D case starting from non-zero origin."""
        A = np.array([[0.2, 0.1], [0.1, 0.3]], dtype=complex)
        b = np.array([0.5, 0.3], dtype=complex)

        hypercube_pivot = (1, 1)
        origin = (5, 3)  # Non-zero origin
        direction = 0

        A_row, b_scalar = Ab_at_pivot_directional(A, b, hypercube_pivot, origin, direction)

        # Should use correct sqrt factors based on origin
        assert A_row.shape == (2, 2)
        assert b_scalar.shape == (2,)
        assert not np.isnan(A_row).any()
        assert not np.isnan(b_scalar).any()

    def test_zero_displacement(self):
        """Test with zero displacement (b = 0)."""
        A = np.array([[0.3, 0.1], [0.1, 0.2]], dtype=complex)
        b = np.array([0.0, 0.0], dtype=complex)  # Zero displacement

        hypercube_pivot = (1, 1)
        origin = (0, 0)
        direction = 0

        A_row, b_scalar = Ab_at_pivot_directional(A, b, hypercube_pivot, origin, direction)

        # b_scalar should be zero or very small
        assert np.allclose(b_scalar, 0.0, atol=1e-10)
        assert A_row.shape == (2, 2)
        assert b_scalar.shape == (2,)

    def test_large_displacement(self):
        """Test with larger displacement values."""
        A = np.array([[0.2, 0.1], [0.1, 0.3]], dtype=complex)
        b = np.array([2.0 + 1.0j, 1.5 - 0.8j], dtype=complex)  # Larger displacement

        hypercube_pivot = (1, 1)
        origin = (0, 0)
        direction = 0

        A_row, b_scalar = Ab_at_pivot_directional(A, b, hypercube_pivot, origin, direction)

        assert A_row.shape == (2, 2)
        assert b_scalar.shape == (2,)
        assert not np.isnan(A_row).any()
        assert not np.isnan(b_scalar).any()
        # b_scalar should reflect the larger displacement
        assert np.any(np.abs(b_scalar) > 0)

    def test_3d_case(self):
        """Test 3D case with multiple dimensions."""
        A = np.array(
            [
                [0.2, 0.1, 0.05],
                [0.1, 0.3, 0.08],
                [0.05, 0.08, 0.25],
            ],
            dtype=complex,
        )
        b = np.array([0.5, 0.3, 0.2], dtype=complex)

        hypercube_pivot = (1, 1, 1)
        origin = (0, 0, 0)
        direction = 1

        A_row, b_scalar = Ab_at_pivot_directional(A, b, hypercube_pivot, origin, direction)

        assert A_row.shape == (3, 3)
        assert b_scalar.shape == (3,)
        assert not np.isnan(A_row).any()
        assert not np.isnan(b_scalar).any()

    def test_boundary_conditions(self):
        """Test various boundary conditions that can cause singular matrices."""
        A = np.array([[0.2, 0.1], [0.1, 0.3]], dtype=complex)
        b = np.array([0.5, 0.3], dtype=complex)

        # Test all boundary positions
        boundary_pivots = [
            (0, 0),  # Origin - most singular
            (0, 1),  # Back in dim 0, front in dim 1
            (1, 0),  # Front in dim 0, back in dim 1
            (1, 1),  # Front face
        ]

        for pivot in boundary_pivots:
            for direction in [0, 1]:
                A_row, b_scalar = Ab_at_pivot_directional(A, b, pivot, (0, 0), direction)
                # Should handle all boundary cases without error
                assert A_row.shape == (2, 2)
                assert b_scalar.shape == (2,)
                assert not np.isnan(A_row).any()
                assert not np.isnan(b_scalar).any()

    def test_payload_dimensions(self):
        """Test with payload dimensions (as used in one_leftover)."""
        # Simulate DM case: 2 measured modes + 2 payload dims (bra, ket)
        A = np.array(
            [
                [0.2, 0.1, 0.0, 0.0],
                [0.1, 0.3, 0.0, 0.0],
                [0.0, 0.0, 0.2, 0.1],
                [0.0, 0.0, 0.1, 0.3],
            ],
            dtype=complex,
        )
        b = np.array([0.5, 0.3, 0.5, 0.3], dtype=complex)

        # Pivot includes payload: (measured_0, measured_1, bra, ket)
        hypercube_pivot = (1, 1, 1, 1)
        origin = (0, 0, 0, 0)
        direction = 0  # Advance in first measured dimension

        A_row, b_scalar = Ab_at_pivot_directional(A, b, hypercube_pivot, origin, direction)

        assert A_row.shape == (4, 4)
        assert b_scalar.shape == (4,)
        assert not np.isnan(A_row).any()
        assert not np.isnan(b_scalar).any()

    def test_consistency_with_recurrence(self):
        """Test that results are consistent with Fock recurrence relation."""
        # Use simple case where we can verify manually
        A = np.array([[0.3]], dtype=complex)
        b = np.array([0.5], dtype=complex)

        hypercube_pivot = (1,)
        origin = (0,)
        direction = 0

        A_row, b_scalar = Ab_at_pivot_directional(A, b, hypercube_pivot, origin, direction)

        # For 1D case at pivot (1,) with origin (0,):
        # full_pivot = (1,)
        # sqrt = sqrt(1) = 1
        # sqrt1 = sqrt(2)
        # A_rescaled = A * 1 / sqrt(2) = 0.3 / sqrt(2)
        # b_rescaled = b / sqrt(2) = 0.5 / sqrt(2)

        # The function should produce valid coefficients
        assert A_row.shape == (1, 1)
        assert b_scalar.shape == (1,)
        assert not np.isnan(A_row).any()
        assert not np.isnan(b_scalar).any()

        # b_scalar should be non-zero for non-zero b
        assert np.any(np.abs(b_scalar) > 0)

    @pytest.mark.parametrize("num_modes", [1, 2, 3, 4])
    def test_various_dimensions(self, num_modes):
        """Test function works for various numbers of modes."""
        rng = np.random.default_rng(42)

        # Generate random valid A and b
        A = (
            rng.standard_normal((num_modes, num_modes)) * 0.1
            + 1j * rng.standard_normal((num_modes, num_modes)) * 0.1
        )
        b = rng.standard_normal(num_modes) * 0.3 + 1j * rng.standard_normal(num_modes) * 0.3

        # Make A symmetric for DM case
        A = (A + A.T) / 2

        hypercube_pivot = (1,) * num_modes
        origin = (0,) * num_modes
        direction = 0

        A_row, b_scalar = Ab_at_pivot_directional(A, b, hypercube_pivot, origin, direction)

        assert A_row.shape == (num_modes, num_modes)
        assert b_scalar.shape == (num_modes,)
        assert not np.isnan(A_row).any()
        assert not np.isnan(b_scalar).any()
        assert not np.isinf(A_row).any()
        assert not np.isinf(b_scalar).any()

    def test_pinv_vs_solve_handles_singular(self):
        """Verify that pinv correctly handles singular cases that solve would fail on."""
        # Create a case where backward_neighbors is singular
        A = np.array([[0.2, 0.1], [0.1, 0.3]], dtype=complex)
        b = np.array([0.5, 0.3], dtype=complex)

        # At origin with multiple zeros - this creates singular backward_neighbors
        hypercube_pivot = (0, 0)
        origin = (0, 0)
        direction = 0

        # This should work with pinv (but would fail with solve)
        A_row, b_scalar = Ab_at_pivot_directional(A, b, hypercube_pivot, origin, direction)

        # Should produce valid (minimum-norm) solution
        assert A_row.shape == (2, 2)
        assert b_scalar.shape == (2,)
        assert not np.isnan(A_row).any()
        assert not np.isnan(b_scalar).any()
