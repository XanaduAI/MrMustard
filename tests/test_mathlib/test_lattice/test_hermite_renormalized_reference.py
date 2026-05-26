# Copyright 2026 Xanadu Quantum Technologies Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Precision tests for ``math.hermite_renormalized``.

Tests are performed against a precomputed reference data file ``data/hermite_reference.npz``.  This
file is produced by ``data/generate_hermite_reference.py`` using the multi-precision math library
``mpmath`` at ~100 decimal digits and rounded to ``complex128`` via IEEE round-to-nearest-even.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from mrmustard import math

REF_PATH = Path(__file__).parent / "data" / "hermite_reference.npz"

# Default tolerances for the precision comparison.
# Element-wise bound is |G - G_ref| <= ATOL + RTOL * |G_ref| (the same form as ``numpy.isclose``).
# ATOL dominates where the reference is near zero (and so catches leakage into structural zeros);
# RTOL dominates where the reference has O(1) or larger magnitude.
ATOL_ZERO = 1e-14
ATOL = 5e-13
RTOL = 1e-13
ULP_TOL = 8.0  # ~8 ULP of relative to the peak-|G_ref| scale in complex128
# Relative-error guard for the auxiliary diagnostic metric printed on failure: skip tiny reference
# entries where rounding dominates the ratio.
RELATIVE_FLOOR = 1e-12


# Per-case tolerance overrides.  Keys may be either ``name`` (applies to both ``stable`` variants)
# or ``(name, stable)`` (applies to just that variant). Use the ``(name, stable)`` form when only
# one variant needs loosening — that way the tighter default remains as a regression tripwire on the
# other variant.
CASE_TOLERANCES: dict = {
    # Oscillatory regime rho(A) = 2: H_n(0.5) passes through a local minimum near n = 44, producing
    # ~3 ULP of relative-error loss via cancellation. Intrinsic to the recurrence conditioning for
    # rho(A) > 1, not to the pivot choice -- all algorithmic variants are equally affected.
    "hermite_poly_x0p5": {"rtol": 5e-13},
    # Dgate at rho(A) = 1. Vanilla algorithm drifts to ~54 ULP (~99 ULP on mac?) over the (20, 20)
    # tensor; averaged pivots stays at ~1.3 ULP.
    ("dgate_a0p5+0p25j", False): {"ulp_tol": 100.0},
    # Generic rho(A) ~ 1 two-dim case, Sgate-like. Vanilla is ~31 ULP at cutoff (30, 30);
    # averaged pivots ~0.4 ULP.
    ("sgate_s-0p5_t0p75", False): {"ulp_tol": 50.0},
    # Generic rho(A) slightly above 1 two-dim, Sgate-like. Vanilla: ~83 ULP at cutoff (30, 30);
    # pivot average ~0.75 ULP.
    ("sgate_s-0p5_t0p875", False): {"ulp_tol": 200.0},
    # Physical Sgate triple at tanh(r) = 1/4, phi = 0.  rho(A) = 1 by unitarity.  Vanilla ~16 ULP
    # over (25, 25); pivot average ~1 ULP.
    ("sgate_physical_tanhr0p25", False): {"ulp_tol": 30.0},
}


def _tol(name: str, stable: bool) -> dict:
    """Return effective tolerances for a given case.

    Applies per-case overrides from ``CASE_TOLERANCES``, otherwise the module-level defaults.

    Lookup precedence: ``(name, stable)`` > ``name`` > defaults.
    """
    override = CASE_TOLERANCES.get((name, stable), CASE_TOLERANCES.get(name, {}))
    return {
        "atol": override.get("atol", ATOL),
        "rtol": override.get("rtol", RTOL),
        "ulp_tol": override.get("ulp_tol", ULP_TOL),
        "atol_zero": override.get("atol_zero", ATOL_ZERO),
    }


def _load_reference() -> tuple[list[str], dict]:
    """Load the reference archive."""
    if not REF_PATH.is_file():
        raise FileNotFoundError(
            f"Reference data file not found: {REF_PATH}.  Regenerate by running: "
            "`python MrMustard/tests/test_mathlib/test_lattice/data/generate_hermite_reference.py`."
        )
    data = np.load(REF_PATH, allow_pickle=False)
    manifest = json.loads(str(data["manifest"]))
    names = [c["name"] for c in manifest["cases"]]
    return names, data


_NAMES, _DATA = _load_reference()
_MANIFEST = json.loads(str(_DATA["manifest"]))
_DESCRIPTIONS = {c["name"]: c["description"] for c in _MANIFEST["cases"]}
_SHAPES = {c["name"]: tuple(c["shape"]) for c in _MANIFEST["cases"]}


def _case(name: str) -> dict:
    """Fetch the ``(A, b, c, G_ref)`` tuple and metadata for a named case."""
    return {
        "A": _DATA[f"{name}/A"],
        "b": _DATA[f"{name}/b"],
        "c": complex(_DATA[f"{name}/c"]),
        "shape": _SHAPES[name],
        "G_ref": _DATA[f"{name}/G_ref"],
        "description": _DESCRIPTIONS[name],
    }


def local_ulp_idx_and_val(A, A_ref):
    """Find the index and value of the maximum local ULP error between A and A_ref."""
    mask = A_ref != 0
    if not np.any(mask):
        return None, float("nan")
    abs_ref = np.abs(A_ref)
    safe_denom = np.where(mask, np.spacing(abs_ref), 1.0)
    err_array = np.where(
        mask,
        np.abs(A - A_ref) / safe_denom,
        0.0,
    )
    err_array_masked = np.where(mask, err_array, -np.inf)
    idx = np.unravel_index(int(np.argmax(err_array_masked)), err_array.shape)
    return idx, float(err_array[idx])


@pytest.mark.parametrize("name", _NAMES)
@pytest.mark.parametrize("stable", [False, True])
def test_matches_reference(name: str, stable: bool) -> None:
    """Test the math.hermite_renormalized output against the reference cases."""
    case = _case(name)
    tol = _tol(name, stable)
    G_ref = case["G_ref"]
    G = np.asarray(
        math.hermite_renormalized(case["A"], case["b"], case["c"], case["shape"], stable=stable)
    )
    assert G.shape == case["G_ref"].shape, (
        f"{name} (stable={stable}): shape mismatch {G.shape} vs {case['G_ref'].shape}"
    )

    # Absolute error
    abs_err_array = np.abs(G - G_ref)
    abs_err_worst_idx = np.unravel_index(int(np.argmax(abs_err_array)), abs_err_array.shape)
    abs_err = abs_err_array[abs_err_worst_idx]

    max_ref = float(np.max(np.abs(G_ref)))
    global_ulp_err = abs_err / (np.finfo(np.float64).eps * max_ref) if max_ref > 0 else 0.0

    # Local ULP error
    local_ulp_err_idx_real, local_ulp_err_real = local_ulp_idx_and_val(np.real(G), np.real(G_ref))
    local_ulp_err_idx_imag, local_ulp_err_imag = local_ulp_idx_and_val(np.imag(G), np.imag(G_ref))
    local_ulp_err_idx_abs, local_ulp_err_abs = local_ulp_idx_and_val(np.abs(G), np.abs(G_ref))

    # Structural zero residual
    zero_mask = case["G_ref"] == 0
    if np.any(zero_mask):
        residual_worst_idx = np.unravel_index(
            int(np.argmax(np.where(zero_mask, abs_err_array, -np.inf))),
            abs_err_array.shape,
        )
        residual = float(np.max(np.abs(G[zero_mask])))
    else:
        residual_worst_idx = None
        residual = float(np.nan)

    # Relative Error
    rel_err_mask = np.abs(G_ref) >= RELATIVE_FLOOR
    abs_G_ref_safe_denom = np.where(rel_err_mask, np.abs(G_ref), 1.0)
    rel_err_array = np.where(rel_err_mask, np.abs(G - G_ref) / abs_G_ref_safe_denom, np.nan)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, message="All-NaN slice encountered"
        )
        rel_err = float(np.nanmax(rel_err_array))
    rel_err_worst_idx = np.unravel_index(int(np.nanargmax(rel_err_array)), rel_err_array.shape)

    # Numpy isclose-style ratio metric, used to find worst index.
    ratio_array = abs_err_array / (tol["atol"] + tol["rtol"] * np.abs(G_ref))
    ratio_worst_idx = np.unravel_index(int(np.argmax(ratio_array)), ratio_array.shape)
    ratio_worst = float(ratio_array[ratio_worst_idx])

    print(
        f"{name} (stable={stable})\n"
        "-Absolute error------------------------------------------\n"
        f"    G         = {G[abs_err_worst_idx]!r}\n"
        f"    G_ref     = {G_ref[abs_err_worst_idx]!r}\n"
        f"    abs_err   = {abs_err:10.3e}  idx={abs_err_worst_idx}\n"
        f"              = {global_ulp_err:10.3f} ULP (globally)\n"
        "-Max local ULP difference--------------------------------\n"
        f"    G         = {np.nan if np.isnan(local_ulp_err_real) else G[local_ulp_err_idx_real]!r}\n"
        f"    G_ref     = {np.nan if np.isnan(local_ulp_err_real) else G_ref[local_ulp_err_idx_real]!r}\n"
        f"  Re(ulp_err) = {local_ulp_err_real:10.2f}  idx={local_ulp_err_idx_real}\n"
        f"    G         = {np.nan if np.isnan(local_ulp_err_imag) else G[local_ulp_err_idx_imag]!r}\n"
        f"    G_ref     = {np.nan if np.isnan(local_ulp_err_imag) else G_ref[local_ulp_err_idx_imag]!r}\n"
        f"  Im(ulp_err) = {local_ulp_err_imag:10.2f}  idx={local_ulp_err_idx_imag}\n"
        f"    G         = {np.nan if np.isnan(local_ulp_err_abs) else G[local_ulp_err_idx_abs]!r}\n"
        f"    G_ref     = {np.nan if np.isnan(local_ulp_err_abs) else G_ref[local_ulp_err_idx_abs]!r}\n"
        f"  |ulp_err|   = {local_ulp_err_abs:10.2f}  idx={local_ulp_err_idx_abs}\n"
        "-Residual zero error-------------------------------------\n"
        f"    G         = {np.nan if np.isnan(residual) else G[residual_worst_idx]!r}\n"
        f"    G_ref     = {np.nan if np.isnan(residual) else G_ref[residual_worst_idx]!r}\n"
        f"    abs_err_0 = {residual:10.3e}  idx={residual_worst_idx}\n"
        "-Relative error------------------------------------------\n"
        f"    G         = {G[rel_err_worst_idx]!r}\n"
        f"    G_ref     = {G_ref[rel_err_worst_idx]!r}\n"
        f"    rel_err   = {rel_err:10.3e}  idx={rel_err_worst_idx}\n"
        "-Numpy isclose-style ratio-------------------------------\n"
        f"    G         = {G[ratio_worst_idx]!r}\n"
        f"    G_ref     = {G_ref[ratio_worst_idx]!r}\n"
        f"    err_ratio = {ratio_worst:10.3e}  idx={ratio_worst_idx}\n"
        "---------------------------------------------------------\n"
        f"  (ATOL={tol['atol']:.1e}, ATOL_0={tol['atol_zero']:.1e}, "
        f"RTOL={tol['rtol']:.1e}, ULP_TOL={tol['ulp_tol']:.1f}, "
        f"ratio = |val| / (ATOL + RTOL * |ref|))\n"
        f"  {case['description']}"
    )

    assert global_ulp_err < tol["ulp_tol"], (
        f"{name} (stable={stable}): global_ulp_err {global_ulp_err:.3f} "
        f">= ulp_tol {tol['ulp_tol']:.1f}"
    )

    # Primary correctness check: numpy-isclose-style combined bound.
    # Scale-adaptive: tight at structural zeros, loose for large |G_ref|.
    assert ratio_worst < 1.0, (
        f"{name} (stable={stable}): worst ratio {ratio_worst:.3e} at {ratio_worst_idx}"
    )

    # Sharper check at structural zeros, where ATOL_ZERO < ATOL.
    if np.any(zero_mask):
        assert residual < tol["atol_zero"], (
            f"{name} (stable={stable}): leakage {residual:.3e} into structural zeros "
            f"at {residual_worst_idx}"
        )
