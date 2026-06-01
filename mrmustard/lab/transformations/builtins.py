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

"""This module contains methods for creating built-in ``Transformation`` ``Ansatz``."""

from __future__ import annotations

from collections.abc import Sequence

from mrmustard import math
from mrmustard.physics import symplectics, triples
from mrmustard.physics.ansatz import ArrayAnsatz, PolyExpAnsatz
from mrmustard.physics.bargmann_utils import symplectic_to_bargmann_Abc
from mrmustard.utils.typing import ComplexMatrix, RealMatrix

__all__ = [
    "amplifier_channel",
    "attenuator_channel",
    "beamsplitter_gate",
    "beamsplitter_gate_fock",
    "cx_gate",
    "cz_gate",
    "displacement_gate",
    "displacement_gate_fock",
    "fock_damping_operation",
    "gaussian_gate",
    "gaussian_random_noise_channel",
    "identity_gate",
    "interferometer_gate",
    "mz_gate",
    "p_gate",
    "real_interferometer_gate",
    "rotation_gate",
    "squeezing_gate",
    "squeezing_gate_fock",
    "twomode_squeezing_gate",
]


def amplifier_channel(
    gain: float | Sequence[float],
    lin_sup: bool = False,
) -> PolyExpAnsatz:
    r"""The noisy amplifier channel in the Bargmann representation.

    Args:
        gain: The gain.
        lin_sup: Whether to include linear superposition axes in the batch dimensions.

    Returns:
        The noisy amplifier channel in the Bargmann representation.
    """
    A, b, c = triples.amplifier_Abc(gain)
    return PolyExpAnsatz(A=A, b=b, c=c, lin_sup=lin_sup)


def attenuator_channel(
    transmissivity: float | Sequence[float],
    lin_sup: bool = False,
) -> PolyExpAnsatz:
    r"""The noisy attenuator channel in the Bargmann representation.

    Args:
        transmissivity: The transmissivity.
        lin_sup: Whether to include linear superposition axes in the batch dimensions.

    Returns:
        The noisy attenuator channel in the Bargmann representation.
    """
    A, b, c = triples.attenuator_Abc(transmissivity)
    return PolyExpAnsatz(A=A, b=b, c=c, lin_sup=lin_sup)


def beamsplitter_gate(
    theta: float | Sequence[float],
    phi: float | Sequence[float] = 0.0,
    lin_sup: bool = False,
) -> PolyExpAnsatz:
    r"""The beam splitter gate in the Bargmann representation.

    Args:
        theta: The transmissivity angle.
        phi: The phase angle.
        lin_sup: Whether to include linear superposition axes in the batch dimensions.

    Returns:
        The beam splitter gate in the Bargmann representation.
    """
    A, b, c = triples.beamsplitter_gate_Abc(theta, phi)
    return PolyExpAnsatz(A=A, b=b, c=c, lin_sup=lin_sup)


def beamsplitter_gate_fock(
    theta: float | Sequence[float],
    phi: float | Sequence[float] = 0.0,
    shape: int | Sequence[int] | None = None,
    method: str = "stable",
    lin_sup: bool = False,
) -> ArrayAnsatz:
    r"""The beamsplitter gate in the Fock representation.

    Args:
        theta: The transmissivity angle.
        phi: The phase angle.
        shape: The shape of the resulting Fock array.
        method: The method to use to compute the Fock array.
        lin_sup: Whether to include linear superposition axes in the batch dimensions.

    Returns:
        The beamsplitter gate in the Fock representation.
    """
    thetas, phis = math.broadcast_arrays(
        theta,
        phi,
    )
    batch_dims = thetas.ndim
    ret = math.astensor(math.beamsplitter(thetas, phis, shape=shape, method=method))
    if lin_sup:
        ret = math.sum(ret, axis=batch_dims - 1)
    return ArrayAnsatz(ret, batch_dims)


def cx_gate(s: float | Sequence[float], lin_sup: bool = False) -> PolyExpAnsatz:
    r"""The controlled-X gate in the Bargmann representation.

    Args:
        s: The control parameter.
        lin_sup: Whether to include linear superposition axes in the batch dimensions.

    Returns:
        The controlled-X gate in the Bargmann representation.
    """
    A, b, c = symplectic_to_bargmann_Abc(symplectics.cxgate_symplectic(s))
    return PolyExpAnsatz(A=A, b=b, c=c, lin_sup=lin_sup)


def cz_gate(s: float | Sequence[float], lin_sup: bool = False) -> PolyExpAnsatz:
    r"""The controlled-Z gate in the Bargmann representation.

    Args:
        s: The control parameter.
        lin_sup: Whether to include linear superposition axes in the batch dimensions.
    """
    A, b, c = symplectic_to_bargmann_Abc(symplectics.czgate_symplectic(s))
    return PolyExpAnsatz(A=A, b=b, c=c, lin_sup=lin_sup)


def displacement_gate(
    alpha: complex | Sequence[complex],
    lin_sup: bool = False,
) -> PolyExpAnsatz:
    r"""The displacement gate in the Bargmann representation.

    Args:
        alpha: The displacement in the complex phase space.
        lin_sup: Whether to include linear superposition axes in the batch dimensions.

    Returns:
        The displacement gate in the Bargmann representation.
    """
    A, b, c = triples.displacement_gate_Abc(alpha)
    return PolyExpAnsatz(A=A, b=b, c=c, lin_sup=lin_sup)


def displacement_gate_fock(
    alpha: complex | Sequence[complex],
    shape: int | Sequence[int],
    lin_sup: bool = False,
) -> ArrayAnsatz:
    r"""The displacement gate in the Fock representation.

    Args:
        alpha: The displacement in the complex phase space.
        shape: The shape of the resulting Fock array.
        lin_sup: Whether to include linear superposition axes in the batch dimensions.

    Returns:
        The displacement gate in the Fock representation.
    """
    batch_dims = alpha.ndim
    ret = math.astensor(math.displacement(alpha, shape=shape))
    if lin_sup:
        ret = math.sum(ret, axis=batch_dims - 1)
    return ArrayAnsatz(ret, batch_dims)


def fock_damping_operation(
    damping: float | Sequence[float],
    lin_sup: bool = False,
) -> PolyExpAnsatz:
    r"""The Fock damping operator in the Bargmann representation.

    Args:
        damping: The damping parameter.
        lin_sup: Whether to include linear superposition axes in the batch dimensions.

    Returns:
        The Fock damping operator in the Bargmann representation.
    """
    A, b, c = triples.fock_damping_Abc(damping)
    return PolyExpAnsatz(A=A, b=b, c=c, lin_sup=lin_sup)


def gaussian_random_noise_channel(
    Y: RealMatrix,
    lin_sup: bool = False,
) -> PolyExpAnsatz:
    r"""The Gaussian random noise channel in the Bargmann representation.

    Args:
        Y: The Y matrix of the Gaussian random noise channel.
        lin_sup: Whether to include linear superposition axes in the batch dimensions.

    Returns:
        The Gaussian random noise channel in the Bargmann representation.
    """
    A, b, c = triples.gaussian_random_noise_Abc(Y)
    return PolyExpAnsatz(A=A, b=b, c=c, lin_sup=lin_sup)


def gaussian_gate(
    symplectic: RealMatrix,
    lin_sup: bool = False,
) -> PolyExpAnsatz:
    r"""The Gaussian gate in the Bargmann representation.

    Args:
        symplectic: The symplectic matrix of the Gaussian gate.
        lin_sup: Whether to include linear superposition axes in the batch dimensions.

    Returns:
        The Gaussian gate in the Bargmann representation.
    """
    A, b, c = symplectic_to_bargmann_Abc(symplectic)
    return PolyExpAnsatz(A=A, b=b, c=c, lin_sup=lin_sup)


def identity_gate(
    n_modes: int,
    lin_sup: bool = False,
) -> PolyExpAnsatz:
    r"""The identity gate in the Bargmann representation.

    Args:
        n_modes: The number of modes.
        lin_sup: Whether to include linear superposition axes in the batch dimensions.

    Returns:
        The identity gate in the Bargmann representation.
    """
    A, b, c = triples.identity_Abc(n_modes)
    return PolyExpAnsatz(A=A, b=b, c=c, lin_sup=lin_sup)


def interferometer_gate(
    unitary: ComplexMatrix,
    lin_sup: bool = False,
) -> PolyExpAnsatz:
    r"""The interferometer gate in the Bargmann representation.

    Args:
        unitary: The unitary matrix of the interferometer gate.
        lin_sup: Whether to include linear superposition axes in the batch dimensions.

    Returns:
        The interferometer gate in the Bargmann representation.
    """
    A, b, c = symplectic_to_bargmann_Abc(symplectics.interferometer_symplectic(unitary))
    return PolyExpAnsatz(A=A, b=b, c=c, lin_sup=lin_sup)


def mz_gate(
    phi_a: float | Sequence[float],
    phi_b: float | Sequence[float],
    internal: bool,
    lin_sup: bool = False,
) -> PolyExpAnsatz:
    r"""The MZ gate in the Bargmann representation.

    Args:
        phi_a: The phase in the upper arm of the MZ interferometer.
        phi_b: The phase in the lower arm of the MZ interferometer.
        internal: Whether the phases are both in the internal arms.
        lin_sup: Whether to include linear superposition axes in the batch dimensions.

    Returns:
        The MZ gate in the Bargmann representation.
    """
    A, b, c = symplectic_to_bargmann_Abc(symplectics.mzgate_symplectic(phi_a, phi_b, internal))
    return PolyExpAnsatz(A=A, b=b, c=c, lin_sup=lin_sup)


def p_gate(
    shearing: float | Sequence[float],
    lin_sup: bool = False,
) -> PolyExpAnsatz:
    r"""The P gate in the Bargmann representation.

    Args:
        shearing: The shearing parameter.
        lin_sup: Whether to include linear superposition axes in the batch dimensions.

    Returns:
        The P gate in the Bargmann representation.
    """
    A, b, c = symplectic_to_bargmann_Abc(symplectics.pgate_symplectic(1, shearing))
    return PolyExpAnsatz(A=A, b=b, c=c, lin_sup=lin_sup)


def real_interferometer_gate(
    orthogonal: RealMatrix,
    lin_sup: bool = False,
) -> PolyExpAnsatz:
    r"""The real interferometer gate in the Bargmann representation.

    Args:
        orthogonal: The orthogonal matrix of the real interferometer gate.
        lin_sup: Whether to include linear superposition axes in the batch dimensions.

    Returns:
        The real interferometer gate in the Bargmann representation.
    """
    A, b, c = symplectic_to_bargmann_Abc(symplectics.realinterferometer_symplectic(orthogonal))
    return PolyExpAnsatz(A=A, b=b, c=c, lin_sup=lin_sup)


def rotation_gate(
    theta: float | Sequence[float],
    lin_sup: bool = False,
) -> PolyExpAnsatz:
    r"""The rotation gate in the Bargmann representation.

    Args:
        theta: The rotation angle.
        lin_sup: Whether to include linear superposition axes in the batch dimensions.

    Returns:
        The rotation gate in the Bargmann representation.
    """
    A, b, c = triples.rotation_gate_Abc(theta)
    return PolyExpAnsatz(A=A, b=b, c=c, lin_sup=lin_sup)


def squeezing_gate(
    r: float | Sequence[float],
    phi: float | Sequence[float] = 0.0,
    lin_sup: bool = False,
) -> PolyExpAnsatz:
    r"""The squeezing gate in the Bargmann representation.

    Args:
        r: The squeezing magnitude.
        phi: The squeezing angle.
        lin_sup: Whether to include linear superposition axes in the batch dimensions.

    Returns:
        The squeezing gate in the Bargmann representation.
    """
    A, b, c = triples.squeezing_gate_Abc(r, phi)
    return PolyExpAnsatz(A=A, b=b, c=c, lin_sup=lin_sup)


def squeezing_gate_fock(
    r: float | Sequence[float],
    phi: float | Sequence[float] = 0.0,
    shape: int | Sequence[int] | None = None,
    lin_sup: bool = False,
) -> ArrayAnsatz:
    r"""The squeezing gate in the Fock representation.

    Args:
        r: The squeezing magnitude.
        phi: The squeezing angle.
        shape: The shape of the resulting Fock array.
        lin_sup: Whether to include linear superposition axes in the batch dimensions.

    Returns:
        The squeezing gate in the Fock representation.
    """
    rs, phis = math.broadcast_arrays(
        r,
        phi,
    )
    batch_dims = rs.ndim
    ret = math.astensor(math.squeezer(rs, phis, shape=shape))
    if lin_sup:
        ret = math.sum(ret, axis=batch_dims - 1)
    return ArrayAnsatz(ret, batch_dims)


def twomode_squeezing_gate(
    r: float | Sequence[float],
    phi: float | Sequence[float] = 0.0,
    lin_sup: bool = False,
) -> PolyExpAnsatz:
    r"""The two-mode squeezing gate in the Bargmann representation.

    Args:
        r: The squeezing amplitude.
        phi: The phase angle.
        lin_sup: Whether to include linear superposition axes in the batch dimensions.

    Returns:
        The two-mode squeezing gate in the Bargmann representation.
    """
    A, b, c = triples.twomode_squeezing_gate_Abc(r, phi)
    return PolyExpAnsatz(A=A, b=b, c=c, lin_sup=lin_sup)
