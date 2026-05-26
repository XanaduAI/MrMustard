#!/usr/bin/env python3
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

r"""Generate the reference data file ``hermite_reference.npz``.

This file is used by ``test_hermite_renormalized_reference.py``.  It is expected that this is only
rerun when the cases considered have changed.  When re-run, the reference file should be checked
into the repository.

For every reference case we store ``(A, b, c)`` together with a high-precision evaluation of the
reference equation for the particular case at the requested tensor shape.  Reference arrays are
computed entirely in ``mpmath`` at a working precision well above the float64 / complex128 dynamic
range and then cast back to ``complex128`` with the standard IEEE round-to-nearest-even convention
via Python's ``complex(mpc)``.

The triples ``A``, ``b``, ``c`` are chosen to be **dyadic rationals** where possible so that the
input is exactly representable as a float.  In the cases where this is not possible (e.g. gates),
at least one parameter is chosen to be a dyadic rational.

The reference case equations are:

- general 1-D, result expressable in terms of a Hermite polynomial,
- 2-D with A matrix off-diagonal only, b=0, result expressable in terms of a Kronecker delta,
- 2-D displacement-gate-style, result expressable in terms of Laguerre polynomials,
- 2-D squeezing-gate-style with general A and b=0, result expressable in terms of a finite
  multinomial sum,
- 4-D beamsplitter-gate-style with block-anti-diagonal A and b=0, result expressable in terms of a
  Schwinger sum.
- multi-dimensional block-diagonal cases built by taking the tensor product of per-block reference
  arrays.
"""

from __future__ import annotations

import io
import json
import math
import zipfile
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import numpy as np
from numpy.lib import format as npy_format

# High precision for all ground-truth evaluations.  At ~100 decimal digits we have over 6 times
# the ~16-digit precision of complex128 components, so the rounded-down reference array is bit-exact
# under round-to-nearest.  To test if precision is sufficient, change the digits of precision (the
# DPS variable below) and check if the output is the same.
DPS = 100
HERE = Path(__file__).parent
OUTPUT = HERE / "hermite_reference.npz"


def _assert_dyadic_scalar(x: float, odd_part_bits: int = 20) -> None:
    """Raise unless ``x`` is a 'reasonably small' dyadic rational p/2^q.

    Every IEEE float64 is exactly a dyadic rational; what we are checking here is that the value
    ``x`` has a small odd mantissa part, i.e. was intentionally entered as an exact value.

    Typical inputs like 0.5, 0.25, 0.375, 1.5, -0.625, 2.0 have odd parts bit sizes of 1, 1, 3, 3,
    5, 1 respectively.  Non-dyadic decimals like 0.1 or 0.3 have odd parts of more than 50 bits in
    double precision. Rejection occurs when the odd part is bigger than 2^``odd_part_bits``.
    """
    if not math.isfinite(x):
        raise ValueError(f"{x!r} is not finite")
    if x == 0.0:
        return
    frac = Fraction(x)  # exact dyadic rational that this float represents
    odd_part = abs(frac.numerator)  # Cannot be zero since x != 0.0
    while odd_part % 2 == 0:
        odd_part //= 2
    if odd_part >= (1 << odd_part_bits):
        raise ValueError(
            f"{x!r} is not a 'nice' dyadic rational (odd part = {odd_part}, exceeds 2^{odd_part_bits}). "
            "Use values like 0.5, 0.25, 0.125, 0.375, 0.75, 1.5, etc."
        )


def _assert_dyadic_complex(z: complex) -> None:
    _assert_dyadic_scalar(z.real)
    _assert_dyadic_scalar(z.imag)


def _assert_array_dyadic(arr: np.ndarray) -> None:
    for v in arr.flat:
        _assert_dyadic_complex(v)


def _mpc(z: complex) -> mp.mpc:
    """Promote a Python scalar (possibly complex) to an ``mpmath.mpc``.

    For dyadic-rational Python floats, ``mp.mpf(x)`` gives the exact binary value of ``x`` -- there
    is no rounding involved.
    """
    return mp.mpc(mp.mpf(z.real), mp.mpf(z.imag))


def _mp_factorial_sqrt(n: int) -> mp.mpf:
    return mp.sqrt(mp.factorial(n))


def _flush_zero_complex_noise(r: float, i: float) -> complex:
    """Zero-out a complex component that is more than ~64 decades below the other.

    These are always mpmath round-off residuals of an algebraic zero, never legitimate complex128
    signal (whose dynamic range is ~15 decades).
    """
    ar, ai = abs(r), abs(i)
    if ar > 0.0 and ai > 0.0:
        if ai < ar * 1e-64:
            i = 0.0
        elif ar < ai * 1e-64:
            r = 0.0
    return complex(r, i)


def _mp_to_complex128(arr: np.ndarray) -> np.ndarray:
    """Cast an object-array of ``mpc`` values to ``complex128`` via round-to-nearest-even."""
    flat = np.empty(arr.size, dtype=np.complex128)
    for idx, x in enumerate(arr.ravel()):
        flat[idx] = _flush_zero_complex_noise(float(x.real), float(x.imag))
    return flat.reshape(arr.shape)


# ---------------------------------------------------------------------------
# Closed-form reference evaluators (all operate on ``mpmath`` objects)
# ---------------------------------------------------------------------------


def _general_1d_reference(A: complex, b: complex, c: complex, N: int) -> np.ndarray:
    r"""General closed form for the 1-D renormalised Hermite recurrence.

    The recurrence
        G_{n+1} = (1/sqrt(n+1)) * (b * G_n + A * sqrt(n) * G_{n-1}),  G_0 = c

    has the exact solution
        G_n = c * H_n(b / sqrt(-2 A)) * (-A/2)^(n/2) / sqrt(n!),     (A != 0)
        G_n = c * b^n / sqrt(n!),                                    (A == 0)

    where ``H_n`` is the physicist Hermite polynomial.  The formula is branch-safe: choosing either
    branch of ``sqrt(-A/2)`` produces the same ``H_n(...) * u^n`` product.
    """
    A_mp = _mpc(A)
    b_mp = _mpc(b)
    c_mp = _mpc(c)

    out = np.empty(N, dtype=object)
    if A_mp == 0:
        for n in range(N):
            out[n] = c_mp * (b_mp**n) / _mp_factorial_sqrt(n)
        return out

    u = mp.sqrt(-A_mp / 2)  # any branch is fine; u^n absorbs it
    sqrt_neg_2A = mp.sqrt(-2 * A_mp)
    arg = b_mp / sqrt_neg_2A
    for n in range(N):
        # mp.hermite(n, z) is H_n(z) (physicist convention).  For A=-2,
        # b=2x this reduces to the pure Hermite case H_n(x).
        out[n] = c_mp * mp.hermite(n, arg) * (u**n) / _mp_factorial_sqrt(n)
    return out


def _2d_offdiag_reference(g: complex, c: complex, N: int) -> np.ndarray:
    r"""Closed form for ``A = [[0, g], [g, 0]], b = 0``:

        G_{n,m} = c * g^n * delta_{n,m}.

    Derivation: the generating function is ``c * exp(g x_0 x_1)``, whose coefficient of ``x_0^n
    x_1^m / (sqrt(n!) sqrt(m!))`` is ``c * g^n * delta_{n,m}``.  Off-diagonals are therefore exactly
    zero.
    """
    g_mp = _mpc(g)
    c_mp = _mpc(c)
    out = np.full((N, N), mp.mpc(0), dtype=object)
    np.fill_diagonal(out, [c_mp * g_mp**n for n in range(N)])
    return out


def _laguerre_poly(k: int, a: int, x: mp.mpf | mp.mpc) -> mp.mpc:
    """Generalised Laguerre polynomial ``L_k^{(a)}(x)`` via its finite series.

        L_k^{(a)}(x) = sum_{j=0..k} (-1)^j * C(k+a, k-j) * x^j / j!

    Using the direct polynomial form avoids the hypergeometric-series convergence issues that
    ``mpmath.laguerre`` can hit at large arguments.
    """
    s = mp.mpc(0)
    for j in range(k + 1):
        binom = mp.gamma(k + a + 1) / (mp.gamma(k - j + 1) * mp.gamma(a + j + 1))
        s = s + (-1) ** j * binom * (x**j) / mp.factorial(j)
    return s


def _dgate_reference(alpha: complex, N: int, M: int) -> np.ndarray:
    r"""Closed form for the Dgate triple ``A=[[0,1],[1,0]], b=[alpha,-conj(alpha)], c=1``.

    The displacement-operator matrix elements are

        <n|D(alpha)|m> = sqrt(m!/n!) alpha^{n-m} exp(-|alpha|^2/2) L_m^{n-m}(|alpha|^2)

    for ``n >= m``, with an analogous formula for ``n < m``.  The Abc triple convention in MrMustard
    normalises ``c = 1`` rather than carrying the ``exp(-|alpha|^2/2)`` factor, so the reference for
    this triple is ``G_{n,m} = exp(|alpha|^2/2) * <n|D(alpha)|m>``.  The two exponentials cancel
    exactly, leaving the purely polynomial form

        G_{n,m} = sqrt(m!/n!) * alpha^{n-m} * L_m^{n-m}(|alpha|^2)           (n >= m)
                = sqrt(n!/m!) * (-conj(alpha))^{m-n} * L_n^{m-n}(|alpha|^2)  (n <  m).

    This triple has spectral radius ``rho(A) = 1`` (eigenvalues are +/-1).
    """
    alpha_mp = _mpc(alpha)
    alpha_conj = alpha_mp.conjugate()
    a_re = mp.mpf(complex(alpha).real)
    a_im = mp.mpf(complex(alpha).imag)
    abs2 = a_re * a_re + a_im * a_im
    out = np.empty((N, M), dtype=object)
    for n, m in np.ndindex(out.shape):
        if n >= m:
            coef = mp.sqrt(mp.factorial(m) / mp.factorial(n))
            out[n, m] = coef * (alpha_mp ** (n - m)) * _laguerre_poly(m, n - m, abs2)
        else:
            coef = mp.sqrt(mp.factorial(n) / mp.factorial(m))
            out[n, m] = coef * ((-alpha_conj) ** (m - n)) * _laguerre_poly(n, m - n, abs2)
    return out


def _sgate_reference(A: np.ndarray, c: complex, shape: tuple[int, int]) -> np.ndarray:
    r"""Closed form for any symmetric 2-D triple ``b = 0``, ``A = [[a00, a01], [a01, a11]]``.

    The generating function is

        F(z_0, z_1) = c exp(a00 z_0^2 / 2 + a11 z_1^2 / 2 + a01 z_0 z_1),

    whose Taylor coefficients, rescaled by ``sqrt(n! m!)`` to match the renormalised-Hermite
    convention, are

        G_{n,m} = c * sqrt(n! m!) * sum_{p,q,r >= 0  with  2p+r=n, 2q+r=m}
                   (a00/2)^p (a11/2)^q a01^r / (p! q! r!)

    The constraints ``2p+r=n`` and ``2q+r=m`` force ``r`` to have the same parity as ``n`` and as
    ``m``; in particular ``n`` and ``m`` must have the same parity for a nonzero result, so all
    ``(n + m) % 2 == 1`` entries are exactly zero.

    Covers the ``_make_sgate_Abc(s, t)`` triples (``a00 = s, a11 = -s, a01 = t``), the physical
    ``Sgate(r, phi)`` triple with its specific (non-dyadic) entries, and any generalisation thereof.
    """
    a00 = _mpc(complex(A[0, 0]))
    a11 = _mpc(complex(A[1, 1]))
    a01 = _mpc(complex(A[0, 1]))
    c_mp = _mpc(complex(c))
    N, M = shape
    out = np.empty((N, M), dtype=object)
    for n, m in np.ndindex(out.shape):
        if (n - m) % 2 != 0:
            out[n, m] = mp.mpc(0)
            continue
        s = mp.mpc(0)
        # r has the same parity as n (and m); step by 2.
        r_min = 0 if (n % 2 == 0) else 1
        r_max = min(n, m)
        for r in range(r_min, r_max + 1, 2):
            p = (n - r) // 2
            q = (m - r) // 2
            term = (
                (a00 / 2) ** p
                * (a11 / 2) ** q
                * a01**r
                / (mp.factorial(p) * mp.factorial(q) * mp.factorial(r))
            )
            s = s + term
        out[n, m] = c_mp * mp.sqrt(mp.factorial(n) * mp.factorial(m)) * s
    return out


def _bsgate_reference(A: np.ndarray, c: complex, shape: tuple[int, int, int, int]) -> np.ndarray:
    r"""Closed form for a 4-D block-anti-diagonal triple.

    This covers ``b = 0``: ``A = [[0, V], [V.T, 0]]`` with ``V`` a 2x2 matrix.

    The generating function is

        F(z) = c exp(z^T A z / 2) = c exp(z_0 z_2 V_00 + z_0 z_3 V_01
                                       + z_1 z_2 V_10 + z_1 z_3 V_11),

    whose only nonvanishing Taylor coefficients have ``n0 + n1 = n2 + n3`` (so all entries violating
    the "photon number conservation" identity ``n0 + n1 - n2 - n3 != 0`` are exactly zero).  For a
    conserving entry, summing over 2x2 grids (contingency tables) with row sums ``(n0, n1)`` and
    column sums ``(n2, n3)``,

        G_{n0,n1,n2,n3} = c * sqrt(n0! n1! n2! n3!)
                          * sum_k  V_00^p_00 V_01^p_01 V_10^p_10 V_11^p_11
                                   / (p_00! p_01! p_10! p_11!)

    where ``p_00 = k, p_01 = n0 - k, p_10 = n2 - k, p_11 = n1 - n2 + k`` and ``k`` ranges over the
    integers keeping every ``p_ij >= 0``.

    Covers the ``_make_bsgate_Abc(V)`` triples (including the physical ``BSgate(theta, phi)`` triple)
    and any generalisation.
    """
    V = np.array(
        [
            [_mpc(complex(A[0, 2])), _mpc(complex(A[0, 3]))],
            [_mpc(complex(A[1, 2])), _mpc(complex(A[1, 3]))],
        ],
        dtype=object,
    )
    c_mp = _mpc(complex(c))
    out = np.empty(shape, dtype=object)
    for n0, n1, n2, n3 in np.ndindex(out.shape):
        if n0 + n1 != n2 + n3:
            out[n0, n1, n2, n3] = mp.mpc(0)
            continue
        k_min = max(0, n2 - n1)
        k_max = min(n0, n2)
        s = mp.mpc(0)
        for k in range(k_min, k_max + 1):
            p00 = k
            p01 = n0 - k
            p10 = n2 - k
            p11 = n1 - n2 + k
            if min(p00, p01, p10, p11) < 0:
                continue
            term = (
                V[0, 0] ** p00
                * V[0, 1] ** p01
                * V[1, 0] ** p10
                * V[1, 1] ** p11
                / (mp.factorial(p00) * mp.factorial(p01) * mp.factorial(p10) * mp.factorial(p11))
            )
            s = s + term
        out[n0, n1, n2, n3] = (
            c_mp
            * mp.sqrt(mp.factorial(n0) * mp.factorial(n1) * mp.factorial(n2) * mp.factorial(n3))
            * s
        )
    return out


def _product_reference(G_factors: list[np.ndarray]) -> np.ndarray:
    """Exact tensor product of mpmath-dtype reference arrays.

    The inputs are ``object`` arrays of ``mpmath.mpc`` values and may have any rank (1D, 2D, ...).
    Products are computed elementwise at the active mpmath working precision, so the result is exact
    up to the final ``complex(mpc)`` cast in ``_mp_to_complex128``.
    """
    for g in G_factors:
        if g.dtype != object or not all(isinstance(x, mp.mpc) for x in g.flat):
            raise TypeError(
                "product reference requires mpmath-dtype inputs so multiplication "
                f"stays at full precision; got dtype={g.dtype}"
            )
    out = G_factors[0]
    for g in G_factors[1:]:
        # ``np.multiply.outer`` on object arrays dispatches to Python's ``*``
        # operator, which for ``mpmath.mpc`` stays at the active working
        # precision.
        out = np.multiply.outer(out, g)
    return out


def _make_1d_Abc(A: complex, b: complex, c: complex) -> tuple[np.ndarray, np.ndarray, complex]:
    A_arr = np.array([[complex(A)]], dtype=np.complex128)
    b_arr = np.array([complex(b)], dtype=np.complex128)
    c_val = complex(c)
    _assert_array_dyadic(A_arr)
    _assert_array_dyadic(b_arr)
    _assert_dyadic_complex(c_val)
    return A_arr, b_arr, c_val


def _make_2d_offdiag_Abc(g: complex, c: complex) -> tuple[np.ndarray, np.ndarray, complex]:
    A_arr = np.array([[0.0 + 0j, complex(g)], [complex(g), 0.0 + 0j]], dtype=np.complex128)
    b_arr = np.zeros(2, dtype=np.complex128)
    c_val = complex(c)
    _assert_array_dyadic(A_arr)
    _assert_array_dyadic(b_arr)
    _assert_dyadic_complex(c_val)
    return A_arr, b_arr, c_val


def _make_dgate_Abc(alpha: complex) -> tuple[np.ndarray, np.ndarray, complex]:
    """Build the (A, b, c) triple for the displacement gate ``D(alpha)``.

    The gate Abc triple in MrMustard's convention is ``A = [[0, 1], [1, 0]]``,
    ``b = [alpha, -conj(alpha)]``, ``c = 1``. The ``exp(-|alpha|^2/2)`` factor is absorbed by the
    reference computation.
    """
    alpha_c = complex(alpha)
    A_arr = np.array([[0.0 + 0j, 1.0 + 0j], [1.0 + 0j, 0.0 + 0j]], dtype=np.complex128)
    b_arr = np.array([alpha_c, -alpha_c.conjugate()], dtype=np.complex128)
    c_val = 1.0 + 0j
    _assert_array_dyadic(A_arr)
    _assert_array_dyadic(b_arr)
    _assert_dyadic_complex(c_val)
    return A_arr, b_arr, c_val


def _make_sgate_Abc(s: complex, t: complex) -> tuple[np.ndarray, np.ndarray, complex]:
    r"""Build an ``Sgate``-style symmetric 2-D triple with ``b = 0``, ``c = 1``.

    ``A = [[s, t], [t, -s]]`` (symmetric, purely ``A``-driven), so the two eigenvalues are ``+/-
    sqrt(s^2 + t^2)``.  With real dyadic ``s, t``, the spectral radius ``rho(A) = sqrt(s^2 + t^2)``
    is freely tunable -- sub-unit, at exactly 1, or just past 1 -- by choice of the two entries.
    """
    A_arr = np.array([[complex(s), complex(t)], [complex(t), complex(-s)]], dtype=np.complex128)
    b_arr = np.zeros(2, dtype=np.complex128)
    c_val = 1.0 + 0j
    _assert_array_dyadic(A_arr)
    _assert_array_dyadic(b_arr)
    _assert_dyadic_complex(c_val)
    return A_arr, b_arr, c_val


def _make_sgate_physical_Abc(
    tanh_r: float, phi: float = 0.0
) -> tuple[np.ndarray, np.ndarray, complex]:
    r"""Build the *physical* Sgate triple with dyadic ``tanh r``.

    The A matrix is completed by a one-ULP fp64 rounding of ``sech r``.

    MrMustard's ``squeezing_gate_Abc(r, phi)`` has
        A = [[-e^{i phi} tanh r,  sech r       ],
             [ sech r,             e^{-i phi} tanh r]],
        b = 0,
        c = 1 / sqrt(cosh r) = sqrt(sech r).

    Parametrising by ``tanh r`` (dyadic) rather than ``r`` (generic) keeps the diagonal entries of
    ``A`` bit-exact in float64.  The off-diagonal ``sech r = sqrt(1 - tanh^2 r)`` is not dyadic for
    generic ``tanh r``, so its float64 value is computed once and then used *as* the true parameter
    for the reference evaluator -- i.e. the reference is the exact Taylor coefficient for whatever
    (A, b, c) actually lives in float64 memory, not for the "true" ``sech r`` at infinite precision.
    Similarly ``cosh r = 1 / sech r`` and ``c = sqrt(sech r)`` are taken as the float64 values.

    With this convention the triple satisfies ``|A_{00}|^2 + |A_{01}|^2 = 1`` to within one ULP, and
    rho(A) = 1 up to that one-ULP.  The test measures pure recurrence error; the one-ULP
    input-rounding mismatch is absorbed into the test-time ULP tolerance.
    """
    _assert_dyadic_scalar(tanh_r)
    _assert_dyadic_scalar(phi)
    if not (0.0 <= tanh_r < 1.0):
        raise ValueError(f"tanh_r must lie in [0, 1); got {tanh_r!r}")
    sech_r = float(mp.sqrt(1 - mp.mpf(tanh_r) ** 2))  # fp64-rounded
    c_val = float(mp.sqrt(mp.mpf(sech_r)))  # fp64-rounded
    eiphi = complex(mp.cos(phi), mp.sin(phi))  # dyadic for phi=0
    A_arr = np.array(
        [
            [-eiphi * tanh_r, sech_r + 0j],
            [sech_r + 0j, eiphi.conjugate() * tanh_r],
        ],
        dtype=np.complex128,
    )
    b_arr = np.zeros(2, dtype=np.complex128)
    c_c = c_val + 0j
    return A_arr, b_arr, c_c


def _make_bsgate_Abc(V: np.ndarray) -> tuple[np.ndarray, np.ndarray, complex]:
    r"""Build a ``BSgate``-style 4-D triple ``A = [[0, V], [V^T, 0]]``, ``b = 0``, ``c = 1``.

    ``A`` is the block anti-diagonal built from a 2x2 matrix ``V``.  The eigenvalues of ``A`` are
    ``+/- sigma_i(V)`` (the singular values of ``V``), so ``rho(A) = ||V||_2``.  The physical
    ``BSgate(theta, phi)`` has a unitary ``V``, which gives ``rho(A) = 1`` exactly; our dyadic
    approximation gives ``||V||_2`` slightly above or below 1.
    """
    V = np.asarray(V, dtype=np.complex128)
    assert V.shape == (2, 2)
    Z = np.zeros((2, 2), dtype=np.complex128)
    A_arr = np.block([[Z, V], [V.T, Z]])
    b_arr = np.zeros(4, dtype=np.complex128)
    c_val = 1.0 + 0j
    _assert_array_dyadic(A_arr)
    _assert_array_dyadic(b_arr)
    _assert_dyadic_complex(c_val)
    return A_arr, b_arr, c_val


def _make_bsgate_physical_Abc(
    cos_theta: float, phi: float = 0.0
) -> tuple[np.ndarray, np.ndarray, complex]:
    r"""Build the *physical* BSgate triple with dyadic ``cos theta``.

    The A matrix is completed by a one-ULP fp64 rounding of ``sin theta``.

    MrMustard's ``beamsplitter_gate_Abc(theta, phi)`` has
        V = [[ cos theta,  -e^{-i phi} sin theta ],
             [ e^{i phi} sin theta,  cos theta ]],
        A = [[0, V], [V^T, 0]],
        b = 0,
        c = 1.

    Parametrising by ``cos theta`` (dyadic) rather than ``theta`` (generic) keeps the diagonal of
    ``V`` bit-exact in float64.  The off-diagonal ``sin theta = sqrt(1 - cos^2 theta)`` is not
    dyadic for generic ``cos theta``, so its float64 value is computed once and then used as the
    true parameter for the reference.  With this choice the dyadic guard is bypassed on the
    off-diagonal entries and will be within one-ULP of the physical value.
    """
    _assert_dyadic_scalar(cos_theta)
    _assert_dyadic_scalar(phi)
    if not (-1.0 <= cos_theta <= 1.0):
        raise ValueError(f"cos_theta must lie in [-1, 1]; got {cos_theta!r}")
    sin_theta = float(mp.sqrt(1 - mp.mpf(cos_theta) ** 2))
    eiphi = complex(mp.cos(phi), mp.sin(phi))
    V = np.array(
        [
            [cos_theta + 0j, -eiphi.conjugate() * sin_theta],
            [eiphi * sin_theta, cos_theta + 0j],
        ],
        dtype=np.complex128,
    )
    Z = np.zeros((2, 2), dtype=np.complex128)
    A_arr = np.block([[Z, V], [V.T, Z]])
    b_arr = np.zeros(4, dtype=np.complex128)
    c_val = 1.0 + 0j
    return A_arr, b_arr, c_val


def _blockdiag(As: list[np.ndarray]) -> np.ndarray:
    """Block-diagonal concatenation of square arrays."""
    D = sum(a.shape[0] for a in As)
    out = np.zeros((D, D), dtype=np.complex128)
    i = 0
    for a in As:
        d = a.shape[0]
        out[i : i + d, i : i + d] = a
        i += d
    return out


def _product_Abc(
    triples: list[tuple[np.ndarray, np.ndarray, complex]],
) -> tuple[np.ndarray, np.ndarray, complex]:
    """Combine per-block ``(A, b, c)`` into a single block-diagonal triple.

    The product ``c`` is computed as a dyadic product of the individual ``c`` values, so if every
    factor is dyadic the result is dyadic too (up to potential cancellation during multiplication).
    The resulting triple is validated to be dyadic.
    """
    A = _blockdiag([t[0] for t in triples])
    b = np.concatenate([t[1] for t in triples])
    # Accumulate c as a Python complex product; dyadic * dyadic = dyadic (up
    # to a growing odd mantissa part that must still fit in ODD_PART_BITS).
    c: complex = 1.0 + 0j
    for _, _, ci in triples:
        c = c * complex(ci)
    _assert_array_dyadic(A)
    _assert_array_dyadic(b)
    _assert_dyadic_complex(c)
    return A, b, c


def _build_cases() -> list[dict]:
    """Return the full list of reference cases."""

    cases: list[dict] = []

    # --- Vacuum -------------------------------------------------------
    A, b, c = _make_1d_Abc(0.0 + 0j, 0.0 + 0j, 1.0 + 0j)
    G_mp = np.full(8, mp.mpc(0), dtype=object)
    G_mp[0] = mp.mpc(1)
    cases.append(
        {
            "name": "vacuum_1d",
            "description": "Vacuum: A=0, b=0, c=1.  G_n = delta_{n,0}.",
            "A": A,
            "b": b,
            "c": c,
            "shape": (8,),
            "G_ref": _mp_to_complex128(G_mp),
        }
    )

    # --- 1-D, b only (A=0): coherent-like --------------------------
    # Exercises the b-path of the recurrence in isolation.  Closed form reduces to G_n = c * b^n / sqrt(n!).
    for A, b, c, tag, short in [
        (0.0 + 0j, 0.5 + 0j, 1.0 + 0j, "coherent_b_r", "A=0, b=0.5, c=1"),
        (0.0 + 0j, 0.75 + 0.125j, 1.0 + 0j, "coherent_b_c", "A=0, b=0.75+0.125j, c=1"),
        (0.0 + 0j, 0.5 - 0.25j, 0.75 + 0.5j, "coherent_scaled", "A=0, b=0.5-0.25j, c=0.75+0.5j"),
    ]:
        A_, b_, c_ = _make_1d_Abc(A, b, c)
        N = 100
        G_mp = _general_1d_reference(A, b, c, N)
        cases.append(
            {
                "name": tag,
                "description": (f"1-D, b-path only (A=0): G_n = c * b^n / sqrt(n!).  {short}."),
                "A": A_,
                "b": b_,
                "c": c_,
                "shape": (N,),
                "G_ref": _mp_to_complex128(G_mp),
            }
        )

    # --- 1-D, A only (b=0): squeezed-like (odd indices exact zero) -
    # For b=0 the recurrence decouples the parities: G_1, G_3, G_5, ... are analytically zero for
    # ANY A, and must come out as 0+0j in the reference.
    for A, b, c, tag, short in [
        (-0.5 + 0j, 0.0 + 0j, 1.0 + 0j, "squeezed_A_r", "A=-0.5, b=0, c=1"),
        (0.0 + 0.5j, 0.0 + 0j, 1.0 + 0j, "squeezed_A_i", "A=0.5j, b=0, c=1"),
        (-0.5 + 0.25j, 0.0 + 0j, 1.0 + 0j, "squeezed_A_c", "A=-0.5+0.25j, b=0, c=1"),
    ]:
        A_, b_, c_ = _make_1d_Abc(A, b, c)
        N = 100
        G_mp = _general_1d_reference(A, b, c, N)
        cases.append(
            {
                "name": tag,
                "description": (
                    f"1-D, A-path only (b=0): all odd-index entries must be exactly zero.  {short}."
                ),
                "A": A_,
                "b": b_,
                "c": c_,
                "shape": (N,),
                "G_ref": _mp_to_complex128(G_mp),
            }
        )

    # --- 1-D, A and b both nonzero: displaced-squeezed-like --------
    # Exercises both the A and b paths of the recurrence simultaneously, including a case with a
    # nontrivial complex ``c`` scale.
    for A, b, c, tag, short in [
        (-0.5 + 0j, 0.5 + 0j, 1.0 + 0j, "dsv_real", "A=-0.5, b=0.5, c=1"),
        (-0.5 + 0.25j, 0.75 + 0.125j, 1.0 + 0j, "dsv_complex", "A=-0.5+0.25j, b=0.75+0.125j, c=1"),
        (
            -0.25 - 0.125j,
            0.5 + 0.25j,
            0.75 - 0.5j,
            "dsv_scaled",
            "A=-0.25-0.125j, b=0.5+0.25j, c=0.75-0.5j",
        ),
    ]:
        A_, b_, c_ = _make_1d_Abc(A, b, c)
        N = 100
        G_mp = _general_1d_reference(A, b, c, N)
        cases.append(
            {
                "name": tag,
                "description": (
                    "1-D, both A and b nonzero: G_n = c * H_n(b/sqrt(-2A))"
                    f" * (-A/2)^(n/2) / sqrt(n!).  {short}."
                ),
                "A": A_,
                "b": b_,
                "c": c_,
                "shape": (N,),
                "G_ref": _mp_to_complex128(G_mp),
            }
        )

    # --- Raw Hermite polynomials --------------------------------------
    # A = -2, b = 2x, c = 1 gives G_n = H_n(x)/sqrt(n!).  Each x is chosen dyadic so that 2x is
    # dyadic and exactly representable.
    for x, tag in [
        (0.0 + 0j, "x0"),  # b = 0    -- odd G_n exactly zero
        (0.5 + 0j, "x0p5"),  # b = 1
        (0.25 + 0.5j, "x0p25+0p5j"),  # b = 0.5+1j
    ]:
        A_, b_, c_ = _make_1d_Abc(-2.0 + 0j, 2.0 * complex(x), 1.0 + 0j)
        N = 100
        G_mp = _general_1d_reference(-2.0 + 0j, 2.0 * complex(x), 1.0 + 0j, N)
        cases.append(
            {
                "name": f"hermite_poly_{tag}",
                "description": (
                    "Physicist Hermite polynomials: A=[[-2]], b=[2x], c=1. "
                    f"G_n = H_n(x)/sqrt(n!), x={x!r}."
                ),
                "A": A_,
                "b": b_,
                "c": c_,
                "shape": (N,),
                "G_ref": _mp_to_complex128(G_mp),
            }
        )

    # --- 2-D off-diagonal A (TMSV-like / thermal-like) -------------
    # A = [[0, g], [g, 0]], b = 0.  Closed form G_{n,m} = c * g^n * delta_{n,m}. Off-diagonal
    # entries must be exactly zero.
    for g, c, tag, short in [
        (0.5 + 0j, 1.0 + 0j, "tmsv_g_r", "g=0.5, c=1"),
        (0.25 + 0.125j, 0.5 + 0j, "tmsv_g_c", "g=0.25+0.125j, c=0.5"),
        (0.5 + 0j, 0.5 + 0j, "tmsv_g_r_c0p5", "g=0.5, c=0.5 (thermal nbar=1 analogue)"),
    ]:
        A_, b_, c_ = _make_2d_offdiag_Abc(g, c)
        N = 50
        G_mp = _2d_offdiag_reference(g, c, N)
        cases.append(
            {
                "name": tag,
                "description": (
                    "2-D off-diagonal A: A=[[0,g],[g,0]], b=0. "
                    f"G_{{n,m}} = c * g^n * delta_{{n,m}}.  {short}."
                    "  All off-diagonals must be exactly zero."
                ),
                "A": A_,
                "b": b_,
                "c": c_,
                "shape": (N, N),
                "G_ref": _mp_to_complex128(G_mp),
            }
        )

    # --- Displacement gate: A=[[0,1],[1,0]], b=[alpha,-conj(alpha)], c=1
    # rho(A) = 1 (eigenvalues +/- 1), so this case sits exactly on the edge of the algorithm's
    # stable regime.  It is the simplest 2-D triple that exercises the stable=True
    # pivot-averaging branch.
    for alpha, tag, short in [
        (0.5 + 0.25j, "dgate_a0p5+0p25j", "alpha=0.5+0.25j"),
        (0.25 - 0.125j, "dgate_a0p25-0p125j", "alpha=0.25-0.125j"),
    ]:
        A_, b_, c_ = _make_dgate_Abc(alpha)
        N = 50
        G_mp = _dgate_reference(alpha, N, N)
        cases.append(
            {
                "name": tag,
                "description": (
                    "Displacement gate D(alpha): A=[[0,1],[1,0]], "
                    "b=[alpha,-conj(alpha)], c=1.  Spectral radius rho(A)=1 "
                    "(boundary of stable regime).  "
                    "G_{n,m} = exp(|alpha|^2/2) * <n|D(alpha)|m> "
                    "= sqrt(m!/n!) * alpha^{n-m} * L_m^{n-m}(|alpha|^2) "
                    "for n >= m (Laguerre-polynomial closed form).  "
                    f"{short}."
                ),
                "A": A_,
                "b": b_,
                "c": c_,
                "shape": (N, N),
                "G_ref": _mp_to_complex128(G_mp),
            }
        )

    # --- Squeezing-gate-style: A=[[s,t],[t,-s]], b=0, c=1 -----------
    # Two-D symmetric A with b = 0 and c = 1 with the structure of the SqueezingGate Abc triple.
    # Eigenvalues are ``+/- sqrt(s^2 + t^2)``, so ``rho(A)`` is directly tunable.  This exercises
    # the A-path of the 2-D recurrence at / past the physical rho(A) = 1 boundary
    for s, t, shape, tag, short in [
        # rho = sqrt(0.25 + 0.5625) = 0.901 -- sub-unit, moderate regime
        (-0.5, 0.75, (30, 30), "sgate_s-0p5_t0p75", "s=-0.5, t=0.75 (rho(A)~0.90)"),
        # rho = sqrt(0.25 + 0.7656...) = sqrt(1.015625) ~ 1.008 -- just past 1
        (
            -0.5,
            0.875,
            (30, 30),
            "sgate_s-0p5_t0p875",
            "s=-0.5, t=0.875 (rho(A)~1.008, just past the physical boundary)",
        ),
    ]:
        A_, b_, c_ = _make_sgate_Abc(s, t)
        G_mp = _sgate_reference(A_, c_, shape)
        cases.append(
            {
                "name": tag,
                "description": (
                    "Sgate-style symmetric 2-D triple: A=[[s,t],[t,-s]], "
                    "b=0, c=1.  Exercises the A-path of the 2-D recurrence "
                    f"at the rho(A) boundary.  {short}."
                ),
                "A": A_,
                "b": b_,
                "c": c_,
                "shape": shape,
                "G_ref": _mp_to_complex128(G_mp),
            }
        )

    # --- Physical Sgate at tanh r = 1/4, phi = 0 -------------------
    # The actual MrMustard squeezing_gate_Abc triple, built with a dyadic ``tanh r`` and a one-ULP
    # fp64 rounding of ``sech r`` as the off-diagonal. rho(A) = 1 up to one ULP in the
    # off-diagonals.
    A_, b_, c_ = _make_sgate_physical_Abc(tanh_r=0.25, phi=0.0)
    shape_sgate = (25, 25)
    G_mp = _sgate_reference(A_, c_, shape_sgate)
    cases.append(
        {
            "name": "sgate_physical_tanhr0p25",
            "description": (
                "Physical Sgate triple (MrMustard's squeezing_gate_Abc) at "
                "tanh r = 1/4, phi = 0.  A = [[-tanh r, sech r], [sech r, "
                "tanh r]], b = 0, c = 1/sqrt(cosh r).  tanh r and the "
                "diagonal are bit-exact dyadic; sech r and c are fp64-rounded. "
                "rho(A) = 1 to within one ULP (Sgate is unitary).  Reference "
                "is the multinomial closed form of exp(z^T A z / 2), "
                "algebraically independent from the forward Hermite recurrence."
            ),
            "A": A_,
            "b": b_,
            "c": c_,
            "shape": shape_sgate,
            "G_ref": _mp_to_complex128(G_mp),
        }
    )

    # --- Beamsplitter-gate-style: 4-D A=[[0,V],[V^T,0]], b=0, c=1 ---
    # Block anti-diagonal 4x4 A built from a 2x2 matrix V.  The full beamsplitter_gate_Abc in
    # MrMustard has exactly this form with V a unitary 2x2 (so ||V||_2 = 1, rho(A) = 1).  Our dyadic
    # V has singular-value max 0.901, chosen just below the boundary.
    V = np.array([[0.75 + 0j, 0.5 + 0j], [-0.5 + 0j, 0.75 + 0j]], dtype=np.complex128)
    A_, b_, c_ = _make_bsgate_Abc(V)
    shape = (10, 10, 10, 10)
    G_mp = _bsgate_reference(A_, c_, shape)
    cases.append(
        {
            "name": "bsgate_V_diag0p75_off0p5",
            "description": (
                "BSgate-style 4-D block-anti-diagonal triple: "
                "A=[[0,V],[V^T,0]], b=0, c=1, with V=[[0.75,0.5],[-0.5,0.75]] "
                "(spectral norm ~ 0.901, just below the physical rho(A)=1 "
                "boundary)."
            ),
            "A": A_,
            "b": b_,
            "c": c_,
            "shape": shape,
            "G_ref": _mp_to_complex128(G_mp),
        }
    )

    # --- Physical BSgate at cos theta = 1/2, phi = 0 ----------------
    # The actual MrMustard beamsplitter_gate_Abc triple, built with a dyadic ``cos theta`` and a
    # one-ULP fp64 rounding of ``sin theta`` as the off-diagonal of V.  ||V||_2 = 1 up to one ULP
    # (BSgate is unitary).
    A_, b_, c_ = _make_bsgate_physical_Abc(cos_theta=0.5, phi=0.0)
    shape_bs = (8, 8, 8, 8)
    G_mp = _bsgate_reference(A_, c_, shape_bs)
    cases.append(
        {
            "name": "bsgate_physical_costheta0p5",
            "description": (
                "Physical BSgate triple (MrMustard's beamsplitter_gate_Abc) "
                "at cos theta = 1/2, phi = 0 (theta = pi/3).  V = "
                "[[cos theta, -sin theta], [sin theta, cos theta]], "
                "A = [[0, V], [V^T, 0]], b = 0, c = 1.  cos theta is "
                "bit-exact dyadic; sin theta is fp64-rounded.  ||V||_2 = 1 "
                "to within one ULP (BSgate is unitary).  Reference is the "
                "2x2 contingency-table closed form, algebraically independent "
                "from the forward Hermite recurrence."
            ),
            "A": A_,
            "b": b_,
            "c": c_,
            "shape": shape_bs,
            "G_ref": _mp_to_complex128(G_mp),
        }
    )

    # --- 2D product: (A=0, b=0.5) X (A=-0.5, b=0) ---------------------
    # Block-diagonal in 2 dims, formed from two 1D cases.  Factorises as the outer product of the
    # two 1-D references.
    A1, b1, c1 = _make_1d_Abc(0.0 + 0j, 0.5 + 0j, 1.0 + 0j)
    A2, b2, c2 = _make_1d_Abc(-0.5 + 0j, 0.0 + 0j, 1.0 + 0j)
    A, b, c = _product_Abc([(A1, b1, c1), (A2, b2, c2)])
    N1, N2 = 50, 50
    G1 = _general_1d_reference(0.0 + 0j, 0.5 + 0j, 1.0 + 0j, N1)
    G2 = _general_1d_reference(-0.5 + 0j, 0.0 + 0j, 1.0 + 0j, N2)
    G_mp = _product_reference([G1, G2])
    cases.append(
        {
            "name": "product_coh_sqz",
            "description": (
                "2D product: (A=0, b=0.5) tensor (A=-0.5, b=0).  Block-diagonal"
                " A, factorising reference.  Odd indices along axis 1 must be"
                " exactly zero (inherited from the A-only factor)."
            ),
            "A": A,
            "b": b,
            "c": c,
            "shape": (N1, N2),
            "G_ref": _mp_to_complex128(G_mp),
        }
    )

    # --- 3D product: (A=0, b=...) X (2x2 off-diag) --------------------
    # 1D factor tensor 2D factor.  Exercises 3D level-scan bookkeeping with a dense axis mixed with
    # a sparse (delta_{n2,n3}) pair of axes.
    A1, b1, c1 = _make_1d_Abc(0.0 + 0j, 0.5 - 0.25j, 1.0 + 0j)
    A2, b2, c2 = _make_2d_offdiag_Abc(0.5 + 0j, 1.0 + 0j)
    A, b, c = _product_Abc([(A1, b1, c1), (A2, b2, c2)])
    N1_3d, N2_3d = 50, 50
    G1 = _general_1d_reference(0.0 + 0j, 0.5 - 0.25j, 1.0 + 0j, N1_3d)
    G2 = _2d_offdiag_reference(0.5 + 0j, 1.0 + 0j, N2_3d)
    G_mp = _product_reference([G1, G2])
    cases.append(
        {
            "name": "product_coh_tmsv_3d",
            "description": (
                "3D product: (A=0, b=0.5-0.25j) tensor (A=[[0,0.5],[0.5,0]],"
                " b=0).  Block-diagonal A.  G_{n1,n2,n3} = "
                "(0.5-0.25j)^n1 / sqrt(n1!) * 0.5^n2 * delta_{n2,n3}."
                "  All entries with n2 != n3 must be exactly zero."
            ),
            "A": A,
            "b": b,
            "c": c,
            "shape": (N1_3d, N2_3d, N2_3d),
            "G_ref": _mp_to_complex128(G_mp),
        }
    )

    # --- 4D product: two independent 2x2 off-diagonal blocks ---------
    # Doubly sparse: only entries with n1 == n2 AND n3 == n4 are nonzero, so 4032 of 4096 entries
    # must be exactly zero.  Uses different g for the two blocks so that any cross-block leakage
    # would be distinguishable from within-block leakage by magnitude.
    A1, b1, c1 = _make_2d_offdiag_Abc(0.5 + 0j, 1.0 + 0j)
    A2, b2, c2 = _make_2d_offdiag_Abc(0.25 + 0.125j, 1.0 + 0j)
    A, b, c = _product_Abc([(A1, b1, c1), (A2, b2, c2)])
    N_4d = 20
    G1 = _2d_offdiag_reference(0.5 + 0j, 1.0 + 0j, N_4d)
    G2 = _2d_offdiag_reference(0.25 + 0.125j, 1.0 + 0j, N_4d)
    G_mp = _product_reference([G1, G2])
    cases.append(
        {
            "name": "product_tmsv_tmsv_4d",
            "description": (
                "4D product: two independent 2-D off-diagonal blocks, "
                "g_a=0.5 and g_b=0.25+0.125j, c=1.  "
                "G_{n1,n2,n3,n4} = g_a^n1 * g_b^n3 * delta_{n1,n2} * delta_{n3,n4}."
                "  Only N^2 of N^4 entries are nonzero."
            ),
            "A": A,
            "b": b,
            "c": c,
            "shape": (N_4d, N_4d, N_4d, N_4d),
            "G_ref": _mp_to_complex128(G_mp),
        }
    )

    return cases


def _save_deterministic_npz(path, payload):
    """Byte-stable replacement for np.savez_compressed.

    Differences from np.savez_compressed:
      - entries are written in sorted name order (not dict-insertion order)
      - each entry's mtime is pinned to the zip epoch (1980-01-01)
      - create_system is pinned to 0 (FAT), not the host OS

    Given identical array contents, the resulting file is byte-identical across runs, platforms, and
    mpmath DPS settings.
    """
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for name in sorted(payload):
            arr = np.asanyarray(payload[name])
            buf = io.BytesIO()
            npy_format.write_array(buf, arr, allow_pickle=False)
            info = zipfile.ZipInfo(
                filename=name + ".npy",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 0
            zf.writestr(info, buf.getvalue())


def main() -> None:
    mp.mp.dps = DPS

    cases = _build_cases()

    # Pack into a single .npz with per-case arrays and a JSON manifest.
    payload: dict[str, np.ndarray] = {}
    manifest = []
    for case in cases:
        name = case["name"]
        payload[f"{name}/A"] = np.ascontiguousarray(case["A"], dtype=np.complex128)
        payload[f"{name}/b"] = np.ascontiguousarray(case["b"], dtype=np.complex128)
        payload[f"{name}/c"] = np.asarray(case["c"], dtype=np.complex128)
        payload[f"{name}/G_ref"] = np.ascontiguousarray(case["G_ref"], dtype=np.complex128)
        manifest.append(
            {
                "name": name,
                "description": case["description"],
                "shape": list(case["shape"]),
            }
        )

    payload["manifest"] = np.array(json.dumps({"cases": manifest}, sort_keys=True))

    _save_deterministic_npz(OUTPUT, payload)
    size_kb = OUTPUT.stat().st_size / 1024
    print(f"Wrote {OUTPUT.name} ({size_kb:.1f} KiB, {len(cases)} cases)")


if __name__ == "__main__":
    main()
