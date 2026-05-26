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

"""Tests for ``mrmustard.mathlib.lattice.strategies.fock_diagonals``."""

import numpy as np
import pytest

from mrmustard import math
from mrmustard.lab import Attenuator, BSgate, GaussianDM, SqueezedVacuum
from mrmustard.mathlib.lattice.strategies import (
    fock_diagonals,
    fock_diagonals_1leftover,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fock_axis_len(max_photon_number: int) -> int:
    r"""Fock dimension for photon numbers ``0, …, max_photon_number`` inclusive."""
    return max_photon_number + 1


def _extract_conditional_dm_2modes(control, pnr_cutoff):
    r"""Extract all conditional DMs from a 2-mode hermite_renormalized tensor.

    Returns shape ``(n_L, n_L, n_P)`` with ``n_L`` / ``n_P`` the leftover and PNR
    Fock axis lengths (each ``cutoff + 1`` for that mode).
    """
    idx = np.arange(_fock_axis_len(pnr_cutoff))
    return np.moveaxis(control[:, idx, :, idx], 0, -1)


def _extract_conditional_dm_3modes(control, pnr_cutoff_0, pnr_cutoff_1):
    r"""Extract all conditional DMs from a 3-mode hermite_renormalized tensor."""
    result = control[
        :, :, np.arange(_fock_axis_len(pnr_cutoff_1)), :, :, np.arange(_fock_axis_len(pnr_cutoff_1))
    ]
    result = result[
        :, :, np.arange(_fock_axis_len(pnr_cutoff_0)), :, np.arange(_fock_axis_len(pnr_cutoff_0))
    ]
    return np.moveaxis(result, [-2, -1], [0, 1])


def _extract_conditional_dm_4modes(control, pnr_cutoff_0, pnr_cutoff_1, pnr_cutoff_2):
    r"""Extract all conditional DMs from a 4-mode hermite_renormalized tensor."""
    result = control[
        :,
        :,
        :,
        np.arange(_fock_axis_len(pnr_cutoff_2)),
        :,
        :,
        :,
        np.arange(_fock_axis_len(pnr_cutoff_2)),
    ]
    result = result[
        :,
        :,
        :,
        np.arange(_fock_axis_len(pnr_cutoff_1)),
        :,
        :,
        np.arange(_fock_axis_len(pnr_cutoff_1)),
    ]
    result = result[
        :, :, :, np.arange(_fock_axis_len(pnr_cutoff_0)), :, np.arange(_fock_axis_len(pnr_cutoff_0))
    ]
    return np.moveaxis(result, [-2, -1], [0, 1])


# ---------------------------------------------------------------------------
# Correctness: fock_diagonals_1leftover vs hermite_renormalized
# ---------------------------------------------------------------------------


def test_fock_diagonals_1leftover_2modes_with_displacement():
    r"""2-mode state with displacement: conditional DM matches hermite_renormalized."""
    A, b, c = GaussianDM.random([0, 1], max_disp=1.0, seed=10).bargmann_triple()
    output_cutoff, pnr_cutoff = 8, 4

    fd = fock_diagonals_1leftover(A, b, c, output_cutoff, (pnr_cutoff,))
    nL, nP = _fock_axis_len(output_cutoff), _fock_axis_len(pnr_cutoff)
    control = math.hermite_renormalized(A, b, c, (nL, nP, nL, nP), stable=True)
    assert np.allclose(fd, _extract_conditional_dm_2modes(control, pnr_cutoff))


def test_fock_diagonals_1leftover_2modes_no_displacement():
    r"""2-mode state without displacement (pure GBS): conditional DM matches hermite_renormalized."""
    A, b, c = GaussianDM.random([0, 1], max_disp=0.0, seed=11).bargmann_triple()
    output_cutoff, pnr_cutoff = 8, 4

    fd = fock_diagonals_1leftover(A, b, c, output_cutoff, (pnr_cutoff,))
    nL, nP = _fock_axis_len(output_cutoff), _fock_axis_len(pnr_cutoff)
    control = math.hermite_renormalized(A, b, c, (nL, nP, nL, nP), stable=True)
    assert np.allclose(fd, _extract_conditional_dm_2modes(control, pnr_cutoff))


def test_fock_diagonals_1leftover_3modes_with_displacement():
    r"""3-mode state with displacement: conditional DM matches hermite_renormalized."""
    A, b, c = GaussianDM.random([0, 1, 2], max_disp=1.0, seed=0).bargmann_triple()
    output_cutoff = 3
    pnr_cutoffs = (4, 5)

    fd = fock_diagonals_1leftover(A, b, c, output_cutoff, pnr_cutoffs)
    nL, n0, n1 = (
        _fock_axis_len(output_cutoff),
        _fock_axis_len(pnr_cutoffs[0]),
        _fock_axis_len(pnr_cutoffs[1]),
    )
    control = math.hermite_renormalized(A, b, c, (nL, n0, n1, nL, n0, n1), stable=True)
    assert np.allclose(fd, _extract_conditional_dm_3modes(control, *pnr_cutoffs))


def test_fock_diagonals_1leftover_3modes_no_displacement():
    r"""3-mode state without displacement (pure GBS): conditional DM matches hermite_renormalized."""
    A, b, c = GaussianDM.random([0, 1, 2], max_disp=0.0, seed=12).bargmann_triple()
    output_cutoff = 3
    pnr_cutoffs = (4, 4)

    fd = fock_diagonals_1leftover(A, b, c, output_cutoff, pnr_cutoffs)
    nL, n0, n1 = (
        _fock_axis_len(output_cutoff),
        _fock_axis_len(pnr_cutoffs[0]),
        _fock_axis_len(pnr_cutoffs[1]),
    )
    control = math.hermite_renormalized(A, b, c, (nL, n0, n1, nL, n0, n1), stable=True)
    assert np.allclose(fd, _extract_conditional_dm_3modes(control, *pnr_cutoffs))


def test_fock_diagonals_1leftover_4modes_with_displacement():
    r"""4-mode state with displacement: conditional DM matches hermite_renormalized."""
    A, b, c = GaussianDM.random([0, 1, 2, 3], max_disp=1.0, seed=20).bargmann_triple()
    output_cutoff = 4
    pnr_cutoffs = (2, 2, 2)

    fd = fock_diagonals_1leftover(A, b, c, output_cutoff, pnr_cutoffs)
    nL = _fock_axis_len(output_cutoff)
    n0, n1, n2 = (_fock_axis_len(p) for p in pnr_cutoffs)
    control = math.hermite_renormalized(A, b, c, (nL, n0, n1, n2, nL, n0, n1, n2), stable=True)
    assert np.allclose(fd, _extract_conditional_dm_4modes(control, *pnr_cutoffs))


def test_fock_diagonals_1leftover_4modes_no_displacement():
    r"""4-mode state without displacement (pure GBS): conditional DM matches hermite_renormalized."""
    A, b, c = GaussianDM.random([0, 1, 2, 3], max_disp=0.0, seed=21).bargmann_triple()
    output_cutoff = 4
    pnr_cutoffs = (2, 2, 2)

    fd = fock_diagonals_1leftover(A, b, c, output_cutoff, pnr_cutoffs)
    nL = _fock_axis_len(output_cutoff)
    n0, n1, n2 = (_fock_axis_len(p) for p in pnr_cutoffs)
    control = math.hermite_renormalized(A, b, c, (nL, n0, n1, n2, nL, n0, n1, n2), stable=True)
    assert np.allclose(fd, _extract_conditional_dm_4modes(control, *pnr_cutoffs))


def test_fock_diagonals_1leftover_large_leftover_cutoff():
    r"""Correctness is maintained on a larger leftover cutoff without post-processing.

    Uses a squeezed-network state typical of GBS experiments, which stresses
    the recurrence more than a generic random Gaussian state.
    """
    rho = SqueezedVacuum(0, 1.727) >> SqueezedVacuum(1, -1.727) >> SqueezedVacuum(2, -1.557)
    rho >>= BSgate([0, 1], 0.371) >> BSgate([1, 2], np.pi / 4)
    rho >>= Attenuator(0, 1.0) >> Attenuator(1, 1.0) >> Attenuator(2, 1.0)
    A, b, c = rho.bargmann_triple()

    output_cutoff = 9
    pnr_cutoffs = (4, 4)

    result = fock_diagonals_1leftover(A, b, c, output_cutoff, pnr_cutoffs)
    nL, n0, n1 = (
        _fock_axis_len(output_cutoff),
        _fock_axis_len(pnr_cutoffs[0]),
        _fock_axis_len(pnr_cutoffs[1]),
    )
    control = math.hermite_renormalized(A, b, c, (nL, n0, n1, nL, n0, n1), stable=True)
    assert np.allclose(result, _extract_conditional_dm_3modes(control, *pnr_cutoffs))


def test_fock_diagonals_1leftover_zero_vacuum_amplitude():
    r"""A zero seed ``c = 0`` is still a valid pre-filled origin for the recurrence."""
    A = np.zeros((2, 2), dtype=np.complex128)
    b = np.array([1.0, 0.0], dtype=np.complex128)
    c = np.complex128(0.0)

    result = fock_diagonals_1leftover(A, b, c, output_cutoff=4, pnr_cutoffs=())
    control = math.hermite_renormalized(
        A, b, c, (_fock_axis_len(4), _fock_axis_len(4)), stable=True
    )
    assert np.allclose(result, control)


# ---------------------------------------------------------------------------
# fock_diagonals
# ---------------------------------------------------------------------------


def _diagonal_amp_control(A, b, c, pnr_cutoffs):
    r"""Return all-mode diagonal amplitudes from the stable full tensor."""
    axis_lens = tuple(_fock_axis_len(p) for p in pnr_cutoffs)
    cutoffs = (*axis_lens, *axis_lens)
    control = math.hermite_renormalized(A, b, c, cutoffs, stable=True)
    diagonal_indices = np.indices(axis_lens)
    return control[tuple(diagonal_indices[mode] for mode in range(len(pnr_cutoffs))) * 2]


def test_fock_diagonals_2modes():
    r"""`fock_diagonals` should match the stable full tensor on two modes."""
    A, b, c = (GaussianDM.random([0, 1], max_disp=1.0, seed=0)).bargmann_triple()

    result = fock_diagonals(A, b, c, (10, 10))
    control = _diagonal_amp_control(A, b, c, (10, 10))

    assert np.allclose(result, control)


def test_fock_diagonals_3modes():
    r"""`fock_diagonals` should generalize to multiple modes."""
    A, b, c = (GaussianDM.random([0, 1, 2], max_disp=1.0, seed=1)).bargmann_triple()

    result = fock_diagonals(A, b, c, (3, 4, 5))
    control = _diagonal_amp_control(A, b, c, (3, 4, 5))

    assert np.allclose(result, control)


def test_fock_diagonals_supports_batched_b():
    r"""Column-batched `b` should match stacking unbatched evaluations."""
    A, b, c = (GaussianDM.random([0, 1, 2], max_disp=1.0, seed=2)).bargmann_triple()
    batched_b = np.stack([b, 0.5 * b, -0.25 * b], axis=1)

    result = fock_diagonals(A, batched_b, c, (3, 4, 5))
    control = np.stack(
        [fock_diagonals(A, batched_b[:, batch], c, (3, 4, 5)) for batch in range(3)],
        axis=-1,
    )

    assert np.allclose(result, control)


def test_fock_diagonals_supports_batched_c():
    r"""Batched `c` should match stacking unbatched evaluations."""
    A, b, c = (GaussianDM.random([0, 1, 2], max_disp=1.0, seed=3)).bargmann_triple()
    batched_c = np.array([c, 0.5 * c, -0.25 * c], dtype=np.complex128)

    result = fock_diagonals(A, b, batched_c, (3, 4, 5))
    control = np.stack(
        [fock_diagonals(A, b, batched_c[batch], (3, 4, 5)) for batch in range(3)],
        axis=-1,
    )

    assert np.allclose(result, control)


def test_fock_diagonals_supports_batched_b_and_c():
    r"""Matching batched `b` and `c` should share the same trailing batch axis."""
    A, b, c = (GaussianDM.random([0, 1, 2], max_disp=1.0, seed=4)).bargmann_triple()
    batched_b = np.stack([b, 0.5 * b, -0.25 * b], axis=1)
    batched_c = np.array([c, 0.5 * c, -0.25 * c], dtype=np.complex128)

    result = fock_diagonals(A, batched_b, batched_c, (3, 4, 5))
    control = np.stack(
        [fock_diagonals(A, batched_b[:, batch], batched_c[batch], (3, 4, 5)) for batch in range(3)],
        axis=-1,
    )

    assert np.allclose(result, control)


def test_fock_diagonals_rejects_incompatible_mode_count():
    r"""The scalar recursion should reject cutoffs that imply too many modes."""
    A = np.zeros((4, 4), dtype=np.complex128)
    b = np.zeros(4, dtype=np.complex128)
    c = np.complex128(0.0)

    with pytest.raises(ValueError, match="incompatible dimensions"):
        fock_diagonals(A, b, c, pnr_cutoffs=(2, 2, 2))


def test_fock_diagonals_rejects_negative_cutoffs():
    r"""The scalar recursion should reject negative diagonal cutoffs."""
    A = np.zeros((4, 4), dtype=np.complex128)
    b = np.zeros(4, dtype=np.complex128)
    c = np.complex128(0.0)

    with pytest.raises(ValueError, match="non-negative"):
        fock_diagonals(A, b, c, pnr_cutoffs=(-1, 2))


def test_fock_diagonals_rejects_invalid_b_rank():
    r"""Only vector and column-batched `b` inputs should be accepted."""
    A = np.zeros((4, 4), dtype=np.complex128)
    b = np.zeros((4, 1, 1), dtype=np.complex128)
    c = np.complex128(0.0)

    with pytest.raises(ValueError, match="one-dimensional or two-dimensional"):
        fock_diagonals(A, b, c, pnr_cutoffs=(2, 2))


def test_fock_diagonals_rejects_invalid_c_rank():
    r"""Only scalar and one-dimensional `c` inputs should be accepted."""
    A = np.zeros((4, 4), dtype=np.complex128)
    b = np.zeros(4, dtype=np.complex128)
    c = np.zeros((1, 1), dtype=np.complex128)

    with pytest.raises(ValueError, match="scalar or one-dimensional"):
        fock_diagonals(A, b, c, pnr_cutoffs=(2, 2))


def test_fock_diagonals_rejects_mismatched_b_and_c_batch_sizes():
    r"""Batched `b` and `c` must agree on their shared batch length."""
    A = np.zeros((4, 4), dtype=np.complex128)
    b = np.zeros((4, 2), dtype=np.complex128)
    c = np.zeros(3, dtype=np.complex128)

    with pytest.raises(ValueError, match="batch dimensions"):
        fock_diagonals(A, b, c, pnr_cutoffs=(2, 2))
