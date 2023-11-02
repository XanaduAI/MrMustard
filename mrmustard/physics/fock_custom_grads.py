# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=redefined-outer-name

"""
This module contains functions for performing calculations on Fock states that require custom gradients.
Once this file or its methods are imported, changing the backend leads to errors.
"""

from typing import Sequence
import numpy as np

from mrmustard.math.lattice import strategies
import mrmustard.math as math

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~ static functions ~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@math.custom_gradient
def displacement(x, y, shape, tol=1e-15):
    r"""creates a single mode displacement matrix"""
    alpha = math.asnumpy(x) + 1j * math.asnumpy(y)

    if np.sqrt(x * x + y * y) > tol:
        gate = strategies.displacement(tuple(shape), alpha)
    else:
        gate = math.eye(max(shape), dtype="complex128")[: shape[0], : shape[1]]

    ret = math.astensor(gate, dtype=gate.dtype.name)
    if math.which == "numpy":
        return ret

    def grad(dL_dDc):
        dD_da, dD_dac = strategies.jacobian_displacement(math.asnumpy(gate), alpha)
        dL_dac = np.sum(np.conj(dL_dDc) * dD_dac + dL_dDc * np.conj(dD_da))
        dLdx = 2 * np.real(dL_dac)
        dLdy = 2 * np.imag(dL_dac)
        return math.astensor(dLdx, dtype=x.dtype), math.astensor(dLdy, dtype=y.dtype)

    return ret, grad


@math.custom_gradient
def beamsplitter(theta: float, phi: float, shape: Sequence[int], method: str):
    r"""Creates a beamsplitter tensor with given cutoffs using a numba-based fock lattice strategy.

    Args:
        theta (float): transmittivity angle of the beamsplitter
        phi (float): phase angle of the beamsplitter
        cutoffs (int,int): cutoff dimensions of the two modes
    """
    if method == "vanilla":
        bs_unitary = strategies.beamsplitter(shape, math.asnumpy(theta), math.asnumpy(phi))
    elif method == "schwinger":
        bs_unitary = strategies.beamsplitter_schwinger(
            shape, math.asnumpy(theta), math.asnumpy(phi)
        )
    else:
        raise ValueError(
            f"Unknown beamsplitter method {method}. Options are 'vanilla' and 'schwinger'."
        )

    ret = math.astensor(bs_unitary, dtype=bs_unitary.dtype.name)
    if math.which == "numpy":
        return ret

    def vjp(dLdGc):
        dtheta, dphi = strategies.beamsplitter_vjp(
            math.asnumpy(bs_unitary),
            math.asnumpy(math.conj(dLdGc)),
            math.asnumpy(theta),
            math.asnumpy(phi),
        )
        return math.astensor(dtheta, dtype=theta.dtype), math.astensor(dphi, dtype=phi.dtype)

    return ret, vjp


@math.custom_gradient
def squeezer(r, phi, shape):
    r"""creates a single mode squeezer matrix using a numba-based fock lattice strategy"""
    sq_unitary = strategies.squeezer(shape, math.asnumpy(r), math.asnumpy(phi))

    ret = math.astensor(sq_unitary, dtype=sq_unitary.dtype.name)
    if math.which == "numpy":
        return ret

    def vjp(dLdGc):
        dr, dphi = strategies.squeezer_vjp(
            math.asnumpy(sq_unitary),
            math.asnumpy(math.conj(dLdGc)),
            math.asnumpy(r),
            math.asnumpy(phi),
        )
        return math.astensor(dr, dtype=r.dtype), math.astensor(dphi, phi.dtype)

    return ret, vjp


@math.custom_gradient
def squeezed(r, phi, shape):
    r"""creates a single mode squeezed state using a numba-based fock lattice strategy"""
    sq_ket = strategies.squeezed(shape, math.asnumpy(r), math.asnumpy(phi))

    ret = math.astensor(sq_ket, dtype=sq_ket.dtype.name)
    if math.which == "numpy":
        return ret

    def vjp(dLdGc):
        dr, dphi = strategies.squeezed_vjp(
            math.asnumpy(sq_ket),
            math.asnumpy(math.conj(dLdGc)),
            math.asnumpy(r),
            math.asnumpy(phi),
        )
        return math.astensor(dr, dtype=r.dtype), math.astensor(dphi, phi.dtype)

    return ret, vjp
