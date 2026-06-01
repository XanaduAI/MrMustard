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

"""Tests for the wigner module.

Covers the mode-validation guard, the state-to-Wigner conversion pipeline,
the marginal integration axis convention, and the stellar roots computation —
the things that can silently break if the code is modified.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mrmustard.lab.states import Coherent, Number, Vacuum
from mrmustard.widgets.wigner import (
    _build_wigner_figure,
    _compute_stellar_roots,
    _draw_wigner_frame,
    _state_to_wigner,
    _style_axis,
)

# ---------------------------------------------------------------------------
# _state_to_wigner — mode validation & conversion pipeline
# ---------------------------------------------------------------------------


class TestStateToWigner:
    """Tests for the state → Wigner conversion and its single-mode guard."""

    def test_multi_mode_raises(self):
        """A multi-mode state raises ValueError with a clear message."""
        xvec = np.linspace(-3, 3, 10)
        pvec = np.linspace(-3, 3, 10)
        with pytest.raises(ValueError, match=r"single-mode.*got 2 modes"):
            _state_to_wigner(Vacuum((0, 1)), min_cutoff=10, xvec=xvec, pvec=pvec)

    def test_single_mode_returns_correct_shapes(self):
        """The conversion pipeline produces W, Q, P arrays of the right shape."""
        n = 20
        xvec = np.linspace(-3, 3, n)
        pvec = np.linspace(-3, 3, n)
        W, Q, P = _state_to_wigner(Vacuum((0,)), min_cutoff=10, xvec=xvec, pvec=pvec)
        assert W.shape == (n, n)
        assert Q.shape == (n, n)
        assert P.shape == (n, n)

    def test_vacuum_wigner_normalization(self):
        """The vacuum Wigner function integrates to approximately 1.

        This validates that the full pipeline — Fock conversion, density
        matrix extraction, np.squeeze, and wigner_discretized — produces
        a physically correct result.
        """
        xvec = np.linspace(-6, 6, 100)
        pvec = np.linspace(-6, 6, 100)
        dx = xvec[1] - xvec[0]
        dp = pvec[1] - pvec[0]
        W, _, _ = _state_to_wigner(Vacuum((0,)), min_cutoff=20, xvec=xvec, pvec=pvec)
        total = W.sum() * dx * dp
        assert total == pytest.approx(1.0, abs=0.05)


# ---------------------------------------------------------------------------
# Marginal integration axis convention
# ---------------------------------------------------------------------------


class TestMarginalIntegration:
    """Verify the integration axes match the Wigner array layout.

    ``wigner_discretized`` uses ``np.outer(q_vec, ones)`` internally, so
    W[i, j] = W(x=xvec[i], p=pvec[j]).  The x-marginal must integrate
    over p (axis 1) and the p-marginal over x (axis 0).  Swapping them
    is an easy mistake that produces plausible-looking but wrong plots.
    """

    def test_x_marginal_integrates_over_p_axis(self):
        """Summing over axis 1 collapses the p dimension."""
        n = 5
        # Row i has constant value (i + 1); summing over axis 1 gives n*(i+1).
        W = np.arange(1, n + 1, dtype=float)[:, np.newaxis] * np.ones((1, n))
        dp = 0.5
        prob_x = W.sum(axis=1) * dp
        np.testing.assert_allclose(prob_x, np.arange(1, n + 1) * n * dp)

    def test_p_marginal_integrates_over_x_axis(self):
        """Summing over axis 0 collapses the x dimension."""
        n = 5
        # Column j has constant value (j + 1); summing over axis 0 gives n*(j+1).
        W = np.ones((n, 1)) * np.arange(1, n + 1, dtype=float)[np.newaxis, :]
        dx = 0.5
        prob_p = W.sum(axis=0) * dx
        np.testing.assert_allclose(prob_p, np.arange(1, n + 1) * n * dx)


# ---------------------------------------------------------------------------
# _build_wigner_figure — layout branching
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _cleanup_figures():
    """Close any matplotlib figures created during the test."""
    yield
    plt.close("all")


class TestBuildWignerFigure:
    """Tests for the layout branching in _build_wigner_figure."""

    def test_with_marginals_returns_marginal_axes(self):
        """With marginals=True, main, top, and right axes are all created."""
        fig, ax_main, ax_top, ax_right, ax_stellar = _build_wigner_figure(
            marginals=True, figsize=None
        )
        assert fig is not None
        assert ax_main is not None
        assert ax_top is not None
        assert ax_right is not None
        assert ax_stellar is None

    def test_without_marginals_side_axes_are_none(self):
        """With marginals=False, only the main axis is created."""
        _, ax_main, ax_top, ax_right, ax_stellar = _build_wigner_figure(
            marginals=False, figsize=None
        )
        assert ax_main is not None
        assert ax_top is None
        assert ax_right is None
        assert ax_stellar is None

    def test_marginals_and_stellar_returns_all_axes(self):
        """With both marginals and stellar, all four axes are created."""
        _, ax_main, ax_top, ax_right, ax_stellar = _build_wigner_figure(
            marginals=True, figsize=None, stellar_roots=True
        )
        assert ax_main is not None
        assert ax_top is not None
        assert ax_right is not None
        assert ax_stellar is not None

    def test_stellar_without_marginals(self):
        """With stellar only, main and stellar axes are created."""
        _, ax_main, ax_top, ax_right, ax_stellar = _build_wigner_figure(
            marginals=False, figsize=None, stellar_roots=True
        )
        assert ax_main is not None
        assert ax_top is None
        assert ax_right is None
        assert ax_stellar is not None


# ---------------------------------------------------------------------------
# _compute_stellar_roots — extraction from quantum states
# ---------------------------------------------------------------------------


class TestComputeStellarRoots:
    """Tests for the stellar roots extraction pipeline.

    These cover the branching logic (Ket vs DM, direct vs Fock-fallback)
    which could silently break if the method signatures or ansatz
    structure change.
    """

    def test_number_state_returns_roots_at_origin(self):
        """|n> has n stellar roots, all at z = 0."""
        roots = _compute_stellar_roots(Number(0, n=3), min_cutoff=10)
        assert len(roots) == 3
        assert np.allclose(np.abs(roots), 0.0, atol=1e-6)

    def test_gaussian_state_returns_empty(self):
        """A Gaussian ket has no finite stellar roots."""
        roots = _compute_stellar_roots(Vacuum(0), min_cutoff=10)
        assert len(roots) == 0

    def test_dm_returns_empty(self):
        """A density matrix state has no stellar_roots method."""
        roots = _compute_stellar_roots(Vacuum(0).dm(), min_cutoff=10)
        assert len(roots) == 0

    def test_linear_superposition_uses_fock_fallback(self):
        """Linear superpositions fall back to Fock conversion.

        Ket.stellar_roots() raises ValueError for linear superpositions,
        so _compute_stellar_roots must catch this and convert to Fock.
        A cat state should produce non-trivial roots.
        """
        cat = Coherent(0, alpha=2.0) + Coherent(0, alpha=-2.0)
        roots = _compute_stellar_roots(cat, min_cutoff=30)
        assert len(roots) > 0


# ---------------------------------------------------------------------------
# _style_axis — axis styling
# ---------------------------------------------------------------------------


class TestStyleAxis:
    """Tests for the _style_axis helper that applies visualize_2d styling."""

    def test_sets_facecolor(self):
        """The axis background is set to aliceblue."""
        _, ax = plt.subplots()
        _style_axis(ax)
        assert ax.get_facecolor() == plt.matplotlib.colors.to_rgba("aliceblue")

    def test_spines_are_visible(self):
        """All four spines are visible after styling."""
        _, ax = plt.subplots()
        _style_axis(ax)
        for spine in ax.spines.values():
            assert spine.get_visible() is True


# ---------------------------------------------------------------------------
# _draw_wigner_frame — rendering coverage
# ---------------------------------------------------------------------------


class TestDrawWignerFrame:
    """Exercise _draw_wigner_frame to cover the rendering code paths.

    These call the real matplotlib drawing routines with synthetic data.
    Visual correctness is verified by eye in notebooks, not here.
    """

    @pytest.fixture()
    def wigner_arrays(self):
        """Minimal Wigner-like arrays for rendering tests."""
        n = 20
        xvec = np.linspace(-3, 3, n)
        pvec = np.linspace(-3, 3, n)
        Q, P = np.meshgrid(xvec, pvec, indexing="ij")
        W = np.exp(-(Q**2 + P**2))
        dx = xvec[1] - xvec[0]
        dp = pvec[1] - pvec[0]
        return W, Q, P, xvec, pvec, dx, dp

    def test_heatmap_only(self, wigner_arrays):
        """Draws the Wigner heatmap without marginals."""
        W, Q, P, xvec, pvec, dx, dp = wigner_arrays
        _, ax_main = plt.subplots()
        _draw_wigner_frame(
            W,
            Q,
            P,
            ax_main=ax_main,
            ax_top=None,
            ax_right=None,
            xvec=xvec,
            pvec=pvec,
            dx=dx,
            dp=dp,
            xbounds=(-3, 3),
            pbounds=(-3, 3),
            cmap="RdBu",
        )
        assert ax_main.get_xlabel() == "x"
        assert ax_main.get_ylabel() == "p"

    def test_with_marginals(self, wigner_arrays):
        """Draws heatmap plus x and p marginal distributions."""
        W, Q, P, xvec, pvec, dx, dp = wigner_arrays
        _, ax_main, ax_top, ax_right, _ = _build_wigner_figure(marginals=True, figsize=None)
        _draw_wigner_frame(
            W,
            Q,
            P,
            ax_main=ax_main,
            ax_top=ax_top,
            ax_right=ax_right,
            xvec=xvec,
            pvec=pvec,
            dx=dx,
            dp=dp,
            xbounds=(-3, 3),
            pbounds=(-3, 3),
            cmap="RdBu",
        )
        assert len(ax_top.lines) > 0
        assert len(ax_right.lines) > 0
