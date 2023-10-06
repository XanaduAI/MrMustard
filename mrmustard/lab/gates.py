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

# pylint: disable=no-member

"""
This module defines gates and operations that can be applied to quantum modes to construct a quantum circuit.
"""

from typing import List, Optional, Sequence, Tuple, Union
import numpy as np
from mrmustard import settings
from mrmustard.lab.abstract import State, Transformation
from mrmustard.math import Math
from mrmustard.physics import fock, gaussian
from mrmustard.training import Parametrized
from mrmustard.typing import ComplexMatrix, RealMatrix

math = Math()

__all__ = [
    "Dgate",
    "Sgate",
    "Rgate",
    "Pgate",
    "Ggate",
    "BSgate",
    "MZgate",
    "S2gate",
    "CZgate",
    "CXgate",
    "Interferometer",
    "RealInterferometer",
    "Attenuator",
    "Amplifier",
    "AdditiveNoise",
    "PhaseNoise",
]


class Dgate(Parametrized, Transformation):
    r"""Displacement gate.

    If ``len(modes) > 1`` the gate is applied in parallel to all of the modes provided.

    If a parameter is a single float, the parallel instances of the gate share that parameter.

    To apply mode-specific values use a list of floats. One can optionally set bounds for each
    parameter, which the optimizer will respect.

    Args:
        x (float or List[float]): the list of displacements along the x axis
        x_bounds (float, float): bounds for the displacement along the x axis
        x_trainable (bool): whether x is a trainable variable
        y (float or List[float]): the list of displacements along the y axis
        y_bounds (float, float): bounds for the displacement along the y axis
        y_trainable bool: whether y is a trainable variable
        modes (optional, List[int]): the list of modes this gate is applied to
    """

    def __init__(
        self,
        x: Union[Optional[float], Optional[List[float]]] = 0.0,
        y: Union[Optional[float], Optional[List[float]]] = 0.0,
        x_trainable: bool = False,
        y_trainable: bool = False,
        x_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        y_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        modes: Optional[List[int]] = None,
    ):
        super().__init__(
            x=x,
            y=y,
            x_trainable=x_trainable,
            y_trainable=y_trainable,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
        )
        self._modes = modes
        self.is_gaussian = True
        self.short_name = "D"

    @property
    def d_vector(self):
        return gaussian.displacement(self.x.value, self.y.value)

    def U(self, cutoffs: Sequence[int]):
        r"""Returns the unitary representation of the Displacement gate using
        the Laguerre polynomials.

        Arguments:
            cutoffs (Sequence[int]): the Hilbert space dimension cutoff for each mode

        Returns:
           Raises:
               ValueError: if the length of the cutoffs array is different from N and 2N
        """

        N = self.num_modes
        x = self.x.value * math.ones(N, dtype=self.x.value.dtype)
        y = self.y.value * math.ones(N, dtype=self.y.value.dtype)
        if len(cutoffs) == N:
            shape = tuple(cutoffs) * 2
        elif len(cutoffs) == 2 * N:
            shape = tuple(cutoffs)
        else:
            raise ValueError(
                "len(cutoffs) should be either equal to the number of modes or twice the number of modes (for output-input)."
            )

        if N > 1:
            # calculate displacement unitary for each mode and concatenate with outer product
            Ud = None
            for idx, out_in in enumerate(zip(shape[:N], shape[N:])):
                if Ud is None:
                    Ud = fock.displacement(x[idx], y[idx], shape=out_in)
                else:
                    U_next = fock.displacement(x[idx], y[idx], shape=out_in)
                    Ud = math.outer(Ud, U_next)

            return math.transpose(
                Ud,
                list(range(0, 2 * N, 2)) + list(range(1, 2 * N, 2)),
            )
        else:
            return fock.displacement(x[0], y[0], shape=shape)


class Sgate(Parametrized, Transformation):
    r"""Squeezing gate.

    If ``len(modes) > 1`` the gate is applied in parallel to all of the modes provided.

    If a parameter is a single float, the parallel instances of the gate share that parameter.

    To apply mode-specific values use a list of floats. One can optionally set bounds for each
    parameter, which the optimizer will respect.

    Args:
        r (float or List[float]): the list of squeezing magnitudes
        r_bounds (float, float): bounds for the squeezing magnitudes
        r_trainable (bool): whether r is a trainable variable
        phi (float or List[float]): the list of squeezing angles
        phi_bounds (float, float): bounds for the squeezing angles
        phi_trainable bool: whether phi is a trainable variable
        modes (optional, List[int]): the list of modes this gate is applied to
    """

    def __init__(
        self,
        r: Union[Optional[float], Optional[List[float]]] = 0.0,
        phi: Union[Optional[float], Optional[List[float]]] = 0.0,
        r_trainable: bool = False,
        phi_trainable: bool = False,
        r_bounds: Tuple[Optional[float], Optional[float]] = (0.0, None),
        phi_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        modes: Optional[List[int]] = None,
    ):
        super().__init__(
            r=r,
            phi=phi,
            r_trainable=r_trainable,
            phi_trainable=phi_trainable,
            r_bounds=r_bounds,
            phi_bounds=phi_bounds,
        )
        self._modes = modes
        self.is_gaussian = True
        self.short_name = "S"

    def U(self, cutoffs: Sequence[int]):
        r"""Returns the unitary representation of the Squeezing gate.
        Args:
            cutoffs (Sequence[int]): the Hilbert space dimension cutoff for each mode

        Returns:
            array[complex]: the unitary matrix
        """
        N = self.num_modes
        if len(cutoffs) == N:
            shape = tuple(cutoffs) * 2
        elif len(cutoffs) == 2 * N:
            shape = tuple(cutoffs)
        else:
            raise ValueError(
                "len(cutoffs) should be either equal to the number of modes or twice the number of modes (for output-input)."
            )
        # this works both or scalar r/phi and vector r/phi:
        r = self.r.value * math.ones(N, dtype=self.r.value.dtype)
        phi = self.phi.value * math.ones(N, dtype=self.phi.value.dtype)

        if N > 1:
            # calculate squeezing unitary for each mode and concatenate with outer product
            Us = None
            for idx, single_shape in enumerate(zip(shape[:N], shape[N:])):
                if Us is None:
                    Us = fock.squeezer(r[idx], phi[idx], shape=single_shape)
                else:
                    U_next = fock.squeezer(r[idx], phi[idx], shape=single_shape)
                    Us = math.outer(Us, U_next)
            return math.transpose(
                Us,
                list(range(0, 2 * N, 2)) + list(range(1, 2 * N, 2)),
            )
        else:
            return fock.squeezer(r[0], phi[0], shape=shape)

    @property
    def X_matrix(self):
        return gaussian.squeezing_symplectic(self.r.value, self.phi.value)


class Rgate(Parametrized, Transformation):
    r"""Rotation gate.

    If ``len(modes) > 1`` the gate is applied in parallel to all of the modes provided.

    If a parameter is a single float, the parallel instances of the gate share that parameter.

    To apply mode-specific values use a list of floats. One can optionally set bounds for each
    parameter, which the optimizer will respect.

    Args:
        modes (List[int]): the list of modes this gate is applied to
        angle (float or List[float]): the list of rotation angles
        angle_bounds (float, float): bounds for the rotation angles
        angle_trainable bool: whether angle is a trainable variable
        modes (optional, List[int]): the list of modes this gate is applied to
    """

    def __init__(
        self,
        angle: Union[Optional[float], Optional[List[float]]] = 0.0,
        angle_trainable: bool = False,
        angle_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        modes: Optional[List[int]] = None,
    ):
        super().__init__(
            angle=angle,
            angle_trainable=angle_trainable,
            angle_bounds=angle_bounds,
        )
        self._modes = modes
        self.is_gaussian = True
        self.short_name = "R"

    @property
    def X_matrix(self):
        return gaussian.rotation_symplectic(self.angle.value)

    def U(self, cutoffs: Sequence[int], diag_only=False):
        r"""Returns the unitary representation of the Rotation gate.

        Args:
            cutoffs (Sequence[int]): cutoff dimension for each mode.
            diag_only (bool): if True, only return the diagonal of the unitary matrix.

        Returns:
            array[complex]: the unitary matrix
        """
        if diag_only:
            raise NotImplementedError("Rgate does not support diag_only=True yet")
        N = self.num_modes
        if len(cutoffs) == N:
            shape = tuple(cutoffs) * 2
        elif len(cutoffs) == 2 * N:
            shape = tuple(cutoffs)
        else:
            raise ValueError(
                "len(cutoffs) should be either equal to the number of modes or twice the number of modes (for output-input)."
            )
        angles = self.angle.value * math.ones(self.num_modes, dtype=self.angle.value.dtype)

        # calculate rotation unitary for each mode and concatenate with outer product
        Ur = None
        for idx, cutoff in enumerate(shape[: self.num_modes]):
            theta = math.arange(cutoff) * angles[idx]
            if Ur is None:
                Ur = math.diag(math.make_complex(math.cos(theta), math.sin(theta)))
            else:
                U_next = math.diag(math.make_complex(math.cos(theta), math.sin(theta)))
                Ur = math.outer(Ur, U_next)

        # return total unitary with indexes reordered according to MM convention
        return math.transpose(
            Ur,
            list(range(0, 2 * self.num_modes, 2)) + list(range(1, 2 * self.num_modes, 2)),
        )


class Pgate(Parametrized, Transformation):
    r"""Quadratic phase gate.

    If len(modes) > 1 the gate is applied in parallel to all of the modes provided. If a parameter
    is a single float, the parallel instances of the gate share that parameter. To apply
    mode-specific values use a list of floats. One can optionally set bounds for each parameter,
    which the optimizer will respect.

    Args:
        modes (List[int]): the list of modes this gate is applied to
        shearing (float or List[float]): the list of shearing parameters
        shearing_bounds (float, float): bounds for the shearing parameters
        shearing_trainable bool: whether shearing is a trainable variable
        modes (optional, List[int]): the list of modes this gate is applied to
    """

    def __init__(
        self,
        shearing: Union[Optional[float], Optional[List[float]]] = 0.0,
        shearing_trainable: bool = False,
        shearing_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        modes: Optional[List[int]] = None,
    ):
        super().__init__(
            shearing=shearing,
            shearing_trainable=shearing_trainable,
            shearing_bounds=shearing_bounds,
        )
        self._modes = modes
        self.is_gaussian = True
        self.short_name = "P"

    @property
    def X_matrix(self):
        return gaussian.quadratic_phase(self.shearing.value)


class CXgate(Parametrized, Transformation):
    r"""Controlled X gate.

    It applies to a single pair of modes. One can optionally set bounds for each parameter, which
    the optimizer will respect.

    Args:
        s (float): control parameter
        s_bounds (float, float): bounds for the control angle
        s_trainable (bool): whether s is a trainable variable
        modes (optional, List[int]): the list of modes this gate is applied to
    """

    def __init__(
        self,
        s: Optional[float] = 0.0,
        s_trainable: bool = False,
        s_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        modes: Optional[List[int]] = None,
    ):
        super().__init__(
            s=s,
            s_trainable=s_trainable,
            s_bounds=s_bounds,
        )
        self._modes = modes
        self.is_gaussian = True
        self.short_name = "CX"

    @property
    def X_matrix(self):
        return gaussian.controlled_X(self.s.value)


class CZgate(Parametrized, Transformation):
    r"""Controlled Z gate.

    It applies to a single pair of modes. One can optionally set bounds for each parameter, which
    the optimizer will respect.

    Args:
        s (float): control parameter
        s_bounds (float, float): bounds for the control angle
        s_trainable (bool): whether s is a trainable variable
        modes (optional, List[int]): the list of modes this gate is applied to
    """

    def __init__(
        self,
        s: Optional[float] = 0.0,
        s_trainable: bool = False,
        s_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        modes: Optional[List[int]] = None,
    ):
        super().__init__(
            s=s,
            s_trainable=s_trainable,
            s_bounds=s_bounds,
        )
        self._modes = modes
        self.is_gaussian = True
        self.short_name = "CZ"

    @property
    def X_matrix(self):
        return gaussian.controlled_Z(self.s.value)


class BSgate(Parametrized, Transformation):
    r"""Beam splitter gate.

    It applies to a single pair of modes.
    One can optionally set bounds for each parameter, which the optimizer will respect.

    Args:
        theta (float): the transmissivity angle
        theta_bounds (float, float): bounds for the transmissivity angle
        theta_trainable (bool): whether theta is a trainable variable
        phi (float): the phase angle
        phi_bounds (float, float): bounds for the phase angle
        phi_trainable bool: whether phi is a trainable variable
        modes (optional, List[int]): the list of modes this gate is applied to
    """

    def __init__(
        self,
        theta: Optional[float] = 0.0,
        phi: Optional[float] = 0.0,
        theta_trainable: bool = False,
        phi_trainable: bool = False,
        theta_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        phi_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        modes: Optional[List[int]] = None,
    ):
        super().__init__(
            theta=theta,
            phi=phi,
            theta_trainable=theta_trainable,
            phi_trainable=phi_trainable,
            theta_bounds=theta_bounds,
            phi_bounds=phi_bounds,
        )
        self._modes = modes
        self.is_gaussian = True
        self.short_name = "BS"

    def U(self, cutoffs: Optional[List[int]], method=None):
        r"""Returns the symplectic transformation matrix for the beam splitter.

        Args:
            cutoffs (List[int]): the list of cutoff dimensions for each mode
                in the order (out_0, out_1, in_0, in_1).
            method (str): the method used to compute the unitary matrix. Options are:
                * 'vanilla': uses the standard method
                * 'schwinger': slower, but numerically stable
            default is set in settings.DEFAULT_BS_METHOD (with 'vanilla' by default)

        Returns:
            array[complex]: the unitary tensor of the beamsplitter
        """
        if len(cutoffs) == 4:
            shape = tuple(cutoffs)
        elif len(cutoffs) == 2:
            shape = tuple(cutoffs) + tuple(cutoffs)
        else:
            raise ValueError(f"Invalid len(cutoffs): {len(cutoffs)} (should be 2 or 4).")
        return fock.beamsplitter(
            self.theta.value,
            self.phi.value,
            shape=shape,
            method=method or settings.DEFAULT_BS_METHOD,
        )

    @property
    def X_matrix(self):
        return gaussian.beam_splitter_symplectic(self.theta.value, self.phi.value)

    def _validate_modes(self, modes):
        if len(modes) != 2:
            raise ValueError(
                f"Invalid number of modes: {len(modes)} (should be 2). Perhaps you are looking for Interferometer."
            )


class MZgate(Parametrized, Transformation):
    r"""Mach-Zehnder gate.

    It supports two conventions:
        1. if ``internal=True``, both phases act inside the interferometer: ``phi_a`` on the upper arm, ``phi_b`` on the lower arm;
        2. if ``internal = False``, both phases act on the upper arm: ``phi_a`` before the first BS, ``phi_b`` after the first BS.

    One can optionally set bounds for each parameter, which the optimizer will respect.

    Args:
        phi_a (float): the phase in the upper arm of the MZ interferometer
        phi_a_bounds (float, float): bounds for phi_a
        phi_a_trainable (bool): whether phi_a is a trainable variable
        phi_b (float): the phase in the lower arm or external of the MZ interferometer
        phi_b_bounds (float, float): bounds for phi_b
        phi_b_trainable (bool): whether phi_b is a trainable variable
        internal (bool): whether phases are both in the internal arms (default is False)
        modes (optional, List[int]): the list of modes this gate is applied to
    """

    def __init__(
        self,
        phi_a: Optional[float] = 0.0,
        phi_b: Optional[float] = 0.0,
        phi_a_trainable: bool = False,
        phi_b_trainable: bool = False,
        phi_a_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        phi_b_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        internal: bool = False,
        modes: Optional[List[int]] = None,
    ):
        super().__init__(
            phi_a=phi_a,
            phi_b=phi_b,
            phi_a_trainable=phi_a_trainable,
            phi_b_trainable=phi_b_trainable,
            phi_a_bounds=phi_a_bounds,
            phi_b_bounds=phi_b_bounds,
        )
        self._internal = internal
        self._modes = modes
        self.is_gaussian = True
        self.short_name = "MZ"

    @property
    def X_matrix(self):
        return gaussian.mz_symplectic(self.phi_a.value, self.phi_b.value, internal=self._internal)

    def _validate_modes(self, modes):
        if len(modes) != 2:
            raise ValueError(
                f"Invalid number of modes: {len(modes)} (should be 2). Perhaps you are looking for Interferometer?"
            )


class S2gate(Parametrized, Transformation):
    r"""Two-mode squeezing gate.

    It applies to a single pair of modes. One can optionally set bounds for each parameter, which the optimizer will respect.

    Args:
        r (float): the squeezing magnitude
        r_bounds (float, float): bounds for the squeezing magnitude
        r_trainable (bool): whether r is a trainable variable
        phi (float): the squeezing angle
        phi_bounds (float, float): bounds for the squeezing angle
        phi_trainable bool: whether phi is a trainable variable
        modes (optional, List[int]): the list of modes this gate is applied to
    """

    def __init__(
        self,
        r: Optional[float] = 0.0,
        phi: Optional[float] = 0.0,
        r_trainable: bool = False,
        phi_trainable: bool = False,
        r_bounds: Tuple[Optional[float], Optional[float]] = (0.0, None),
        phi_bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        modes: Optional[List[int]] = None,
    ):
        super().__init__(
            r=r,
            phi=phi,
            r_trainable=r_trainable,
            phi_trainable=phi_trainable,
            r_bounds=r_bounds,
            phi_bounds=phi_bounds,
        )
        self._modes = modes
        self.is_gaussian = True
        self.short_name = "S2"

    @property
    def X_matrix(self):
        return gaussian.two_mode_squeezing_symplectic(self.r.value, self.phi.value)

    def _validate_modes(self, modes):
        if len(modes) != 2:
            raise ValueError(f"Invalid number of modes: {len(modes)} (should be 2")


class Interferometer(Parametrized, Transformation):
    r"""N-mode interferometer.

    It corresponds to a Ggate with zero mean and a ``2N x 2N`` unitary symplectic matrix.

    Args:
        num_modes (int): the num_modes-mode interferometer
        unitary (2d array): a valid unitary matrix U. For N modes it must have shape `(N,N)`
        unitary_trainable (bool): whether unitary is a trainable variable
        modes (optional, List[int]): the list of modes this gate is applied to
    """

    def __init__(
        self,
        num_modes: int,
        unitary: Optional[ComplexMatrix] = None,
        unitary_trainable: bool = False,
        modes: Optional[List[int]] = None,
    ):
        if modes is not None and num_modes != len(modes):
            raise ValueError(f"Invalid number of modes: got {len(modes)}, should be {num_modes}")
        if unitary is None:
            unitary = math.random_unitary(num_modes)
        super().__init__(
            unitary=unitary,
            unitary_trainable=unitary_trainable,
        )
        self._modes = modes or list(range(num_modes))
        self.is_gaussian = True
        self.short_name = "I"

    @property
    def X_matrix(self):
        return math.block(
            [
                [math.real(self.unitary.value), -math.imag(self.unitary.value)],
                [math.imag(self.unitary.value), math.real(self.unitary.value)],
            ]
        )

    def _validate_modes(self, modes):
        if len(modes) != self.unitary.value.shape[-1]:
            raise ValueError(
                f"Invalid number of modes: {len(modes)} (should be {self.unitary.shape[-1]})"
            )

    def __repr__(self):
        modes = self.modes
        unitary = repr(math.asnumpy(self.unitary.value)).replace("\n", "")
        return f"Interferometer(num_modes = {len(modes)}, unitary = {unitary}){modes}"


class RealInterferometer(Parametrized, Transformation):
    r"""N-mode interferometer parametrized by an NxN orthogonal matrix (or 2N x 2N block-diagonal orthogonal matrix). This interferometer does not mix q and p.
    Does not mix q's and p's.

    Args:
        orthogonal (2d array, optional): a real unitary (orthogonal) matrix. For N modes it must have shape `(N,N)`.
            If set to `None` a random real unitary (orthogonal) matrix is used.
        orthogonal_trainable (bool): whether orthogonal is a trainable variable
    """

    def __init__(
        self,
        num_modes: int,
        orthogonal: Optional[RealMatrix] = None,
        orthogonal_trainable: bool = False,
        modes: Optional[List[int]] = None,
    ):
        if modes is not None and (num_modes != len(modes)):
            raise ValueError(f"Invalid number of modes: got {len(modes)}, should be {num_modes}")
        if orthogonal is None:
            orthogonal = math.random_orthogonal(num_modes)
        super().__init__(orthogonal=orthogonal, orthogonal_trainable=orthogonal_trainable)
        self._modes = modes or list(range(num_modes))
        self._is_gaussian = True
        self.short_name = "RI"

    @property
    def X_matrix(self):
        return math.block(
            [
                [self.orthogonal.value, -math.zeros_like(self.orthogonal.value)],
                [math.zeros_like(self.orthogonal.value), self.orthogonal.value],
            ]
        )

    def _validate_modes(self, modes):
        if len(modes) != self.orthogonal.value.shape[-1]:
            raise ValueError(
                f"Invalid number of modes: {len(modes)} (should be {self.orthogonal.value.shape[-1]})"
            )

    def __repr__(self):
        modes = self.modes
        orthogonal = repr(math.asnumpy(self.orthogonal.value)).replace("\n", "")
        return f"RealInterferometer(num_modes = {len(modes)}, orthogonal = {orthogonal}){modes}"


class Ggate(Parametrized, Transformation):
    r"""A generic N-mode Gaussian unitary transformation with zero displacement.

    If a symplectic matrix is not provided, one will be picked at random with effective squeezing
    strength ``r`` in ``[0, 1]`` for each mode.

    Args:
        num_modes (int): the number of modes this gate is acting on.
        symplectic (2d array): a valid symplectic matrix in XXPP order. For N modes it must have shape ``(2N,2N)``.
        symplectic_trainable (bool): whether symplectic is a trainable variable.
    """

    def __init__(
        self,
        num_modes: int,
        symplectic: Optional[RealMatrix] = None,
        symplectic_trainable: bool = False,
        modes: Optional[List[int]] = None,
    ):
        if modes is not None and (num_modes != len(modes)):
            raise ValueError(f"Invalid number of modes: got {len(modes)}, should be {num_modes}")
        if symplectic is None:
            symplectic = math.random_symplectic(num_modes)
        super().__init__(
            symplectic=symplectic,
            symplectic_trainable=symplectic_trainable,
        )
        self._modes = modes or list(range(num_modes))
        self.is_gaussian = True
        self.short_name = "G"

    @property
    def X_matrix(self):
        return self.symplectic.value

    def _validate_modes(self, modes):
        if len(modes) != self.symplectic.value.shape[1] // 2:
            raise ValueError(
                f"Invalid number of modes: {len(modes)} (should be {self.symplectic.value.shape[1] // 2})"
            )

    def __repr__(self):
        modes = self.modes
        symplectic = repr(math.asnumpy(self.symplectic.value)).replace("\n", "")
        return f"Ggate(num_modes = {len(modes)}, symplectic = {symplectic}){modes}"


# ~~~~~~~~~~~~~
# NON-UNITARY
# ~~~~~~~~~~~~~


# pylint: disable=no-member
class Attenuator(Parametrized, Transformation):
    r"""The noisy attenuator channel.

    It corresponds to mixing with a thermal environment and applying the pure loss channel. The pure
    lossy channel is recovered for nbar = 0 (i.e. mixing with vacuum).

    The CPT channel is given by

    .. math::

        X = sqrt(transmissivity) * I
        Y = (1-transmissivity) * (2*nbar + 1) * (hbar / 2) * I

    If ``len(modes) > 1`` the gate is applied in parallel to all of the modes provided.
    If ``transmissivity`` is a single float, the parallel instances of the gate share that parameter.

    To apply mode-specific values use a list of floats.

    One can optionally set bounds for `transmissivity`, which the optimizer will respect.

    Args:
        transmissivity (float or List[float]): the list of transmissivities
        nbar (float): the average number of photons in the thermal state
        transmissivity_trainable (bool): whether transmissivity is a trainable variable
        nbar_trainable (bool): whether nbar is a trainable variable
        transmissivity_bounds (float, float): bounds for the transmissivity
        nbar_bounds (float, float): bounds for the average number of photons in the thermal state
        modes (optional, List[int]): the list of modes this gate is applied to
    """

    def __init__(
        self,
        transmissivity: Union[Optional[float], Optional[List[float]]] = 1.0,
        nbar: float = 0.0,
        transmissivity_trainable: bool = False,
        nbar_trainable: bool = False,
        transmissivity_bounds: Tuple[Optional[float], Optional[float]] = (0.0, 1.0),
        nbar_bounds: Tuple[Optional[float], Optional[float]] = (0.0, None),
        modes: Optional[List[int]] = None,
    ):
        super().__init__(
            transmissivity=transmissivity,
            nbar=nbar,
            transmissivity_trainable=transmissivity_trainable,
            nbar_trainable=nbar_trainable,
            transmissivity_bounds=transmissivity_bounds,
            nbar_bounds=nbar_bounds,
        )
        self._modes = modes
        self.is_unitary = False
        self.is_gaussian = True
        self.short_name = "Att"

    @property
    def X_matrix(self):
        return gaussian.loss_XYd(self.transmissivity.value, self.nbar.value)[0]

    @property
    def Y_matrix(self):
        return gaussian.loss_XYd(self.transmissivity.value, self.nbar.value)[1]


class Amplifier(Parametrized, Transformation):
    r"""The noisy amplifier channel.

    It corresponds to mixing with a thermal environment and applying a two-mode squeezing gate.

    .. code:: python

        X = sqrt(gain) * I
        Y = (gain-1) * (2*nbar + 1) * (hbar / 2) * I

    If ``len(modes) > 1`` the gate is applied in parallel to all of the modes provided.
    If ``gain`` is a single float, the parallel instances of the gate share that parameter.
    To apply mode-specific values use a list of floats.
    One can optionally set bounds for ``gain``, which the optimizer will respect.

    Args:
        gain (float or List[float]): the list of gains (must be > 1)
        nbar (float): the average number of photons in the thermal state
        nbar_trainable (bool): whether nbar is a trainable variable
        gain_trainable (bool): whether gain is a trainable variable
        gain_bounds (float, float): bounds for the gain
        nbar_bounds (float, float): bounds for the average number of photons in the thermal state
        modes (optional, List[int]): the list of modes this gate is applied to
    """

    def __init__(
        self,
        gain: Union[Optional[float], Optional[List[float]]] = 1.0,
        nbar: float = 0.0,
        gain_trainable: bool = False,
        nbar_trainable: bool = False,
        gain_bounds: Tuple[Optional[float], Optional[float]] = (1.0, None),
        nbar_bounds: Tuple[Optional[float], Optional[float]] = (0.0, None),
        modes: Optional[List[int]] = None,
    ):
        super().__init__(
            gain=gain,
            gain_trainable=gain_trainable,
            gain_bounds=gain_bounds,
            nbar=nbar,
            nbar_trainable=nbar_trainable,
            nbar_bounds=nbar_bounds,
        )
        self._modes = modes
        self.is_unitary = False
        self.is_gaussian = True
        self.short_name = "Amp"

    @property
    def X_matrix(self):
        return gaussian.amp_XYd(self.gain.value, self.nbar.value)[0]

    @property
    def Y_matrix(self):
        return gaussian.amp_XYd(self.gain.value, self.nbar.value)[1]


# pylint: disable=no-member
class AdditiveNoise(Parametrized, Transformation):
    r"""The additive noise channel.

    Equivalent to an amplifier followed by an attenuator. E.g.,

    .. code-block::

        na,nb = np.random.uniform(size=2)
        tr = np.random.uniform()
        Amplifier(1/tr, nb) >> Attenuator(tr, na) == AdditiveNoise(2*(1-tr)*(1+na+nb)) # evaluates to True

    or equivalent to an attenuator followed by an amplifier:

    .. code-block::

        na,nb = np.random.uniform(size=2)
        amp = 1.0 + np.random.uniform()
        Attenuator(1/amp, nb) >> Amplifier(amp, na) == AdditiveNoise(2*(amp-1)*(1+na+nb))

    Args:
        noise (float or List[float]): the added noise in units of hbar/2
        noise_trainable (bool): whether noise is a trainable variable
        noise_bounds (float, float): bounds for the noise
        modes (optional, List[int]): the list of modes this gate is applied to
    """

    def __init__(
        self,
        noise: Union[Optional[float], Optional[List[float]]] = 0.0,
        noise_trainable: bool = False,
        noise_bounds: Tuple[Optional[float], Optional[float]] = (0.0, None),
        modes: Optional[List[int]] = None,
    ):
        super().__init__(
            noise=noise,
            noise_trainable=noise_trainable,
            noise_bounds=noise_bounds,
        )
        self._modes = modes
        self.is_unitary = False
        self.is_gaussian = True
        self.short_name = "Add"

    @property
    def Y_matrix(self):
        return gaussian.noise_Y(self.noise.value)


class PhaseNoise(Parametrized, Transformation):
    r"""The phase noise channel.

    The phase noise channel is a non-Gaussian transformation that is equivalent to
    a random phase rotation.

    Args:
        phase_stdev (float or List[float]): the standard deviation of the (wrapped) normal
            distribution for the angle of the rotation
        modes (optional, list(int)): the single mode this gate is applied to (default [0])
    """

    def __init__(
        self,
        phase_stdev: Union[Optional[float], Optional[List[float]]] = 0.0,
        phase_stdev_trainable: bool = False,
        phase_stdev_bounds: Tuple[Optional[float], Optional[float]] = (0.0, None),
        modes: Optional[List[int]] = None,
    ):
        super().__init__(
            phase_stdev=phase_stdev,
            phase_stdev_trainable=phase_stdev_trainable,
            phase_stdev_bounds=phase_stdev_bounds,
        )
        self._modes = modes or [0]
        self.is_unitary = False
        self.is_gaussian = False
        self.short_name = "P~"

    # need to override primal because of the unconventional way
    # the channel is defined in Fock representation
    def primal(self, state):
        idx = state.modes.index(self.modes[0])
        if state.is_pure:
            ket = state.ket()
            dm = fock.ket_to_dm(ket)
        else:
            dm = state.dm()

        # transpose dm so that the modes of interest are at the end
        M = state.num_modes
        indices = list(range(2 * M))
        indices.remove(idx)
        indices.remove(idx + M)
        indices += [idx, idx + M]
        dm = math.transpose(dm, indices)

        coeff = math.cast(
            math.exp(
                -0.5
                * self.phase_stdev.value**2
                * math.arange(-dm.shape[-2] + 1, dm.shape[-1]) ** 2
            ),
            dm.dtype,
        )

        for k in range(-dm.shape[-2] + 1, dm.shape[-1]):
            diagonal = math.diag_part(dm, k=k)
            diagonal *= coeff[k + dm.shape[-2] - 1]
            dm = math.set_diag(dm, diagonal, k=k)

        # transpose dm back to the original order
        return State(dm=math.transpose(dm, np.argsort(indices)), modes=state.modes)
