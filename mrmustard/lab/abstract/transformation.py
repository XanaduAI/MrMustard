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

"""This module contains the implementation of the :class:`Transformation` class."""


# pylint: disable = missing-function-docstring

from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

from mrmustard import math, settings
from mrmustard.math.parameter_set import ParameterSet
from mrmustard.math.parameters import Constant, Variable
from mrmustard.math.tensor_networks import Tensor
from mrmustard.physics import bargmann, fock, gaussian
from mrmustard.utils.typing import RealMatrix, RealVector

from .state import State


class Transformation(Tensor):
    r"""
    Base class for all Transformations.
    """

    def __init__(
        self,
        name: str,
        modes_in_ket: Optional[list[int]] = None,
        modes_out_ket: Optional[list[int]] = None,
        modes_in_bra: Optional[list[int]] = None,
        modes_out_bra: Optional[list[int]] = None,
    ):
        super().__init__(
            name=name,
            modes_in_ket=modes_in_ket,
            modes_out_ket=modes_out_ket,
            modes_in_bra=modes_in_bra,
            modes_out_bra=modes_out_bra,
        )
        self._parameter_set = ParameterSet()

    def _add_parameter(self, parameter: Union[Constant, Variable]):
        r"""
        Adds a parameter to a transformation.

        Args:
            parameter: The parameter to add.
        """
        self.parameter_set.add_parameter(parameter)
        self.__dict__[parameter.name] = parameter

    @property
    def parameter_set(self):
        r"""
        The set of parameters for this transformation.
        """
        return self._parameter_set

    def primal(self, state: State) -> State:
        r"""Applies this transformation to the given ``state`` and returns the transformed state.

        Args:
            state (State): the state to transform

        Returns:
            State: the transformed state
        """
        if state.is_gaussian:
            new_state = self._transform_gaussian(state, dual=False)
        else:
            new_state = self._transform_fock(state, dual=False)
        return new_state

    def dual(self, state: State) -> State:
        r"""Applies the dual of this transformation to the given ``state`` and returns the transformed state.

        Args:
            state (State): the state to transform

        Returns:
            State: the transformed state
        """
        if state.is_gaussian:
            new_state = self._transform_gaussian(state, dual=True)
        else:
            new_state = self._transform_fock(state, dual=True)
        return new_state

    def _transform_gaussian(self, state: State, dual: bool) -> State:
        r"""Transforms a Gaussian state into a Gaussian state.

        Args:
            state (State): the state to transform
            dual (bool): whether to apply the dual channel

        Returns:
            State: the transformed state
        """
        X, Y, d = self.XYd(allow_none=False) if not dual else self.XYd_dual(allow_none=False)
        cov, means = gaussian.CPTP(
            math.astensor(state.cov), math.astensor(state.means), X, Y, d, state.modes, self.modes
        )
        new_state = State(
            cov=cov, means=means, modes=state.modes, _norm=state.norm
        )  # NOTE: assumes modes don't change
        return new_state

    @property
    def num_modes(self) -> int:
        r"""The number of modes on which the transformation acts."""
        return len(self.modes)

    def _validate_modes(self, modes):
        pass

    @property
    def X_matrix(self) -> Optional[RealMatrix]:
        return None

    @property
    def Y_matrix(self) -> Optional[RealMatrix]:
        return None

    @property
    def d_vector(self) -> Optional[RealVector]:
        return None

    @property
    def X_matrix_dual(self) -> Optional[RealMatrix]:
        if (X := self.X_matrix) is None:
            return None
        return gaussian.math.inv(X)

    @property
    def Y_matrix_dual(self) -> Optional[RealMatrix]:
        if (Y := self.Y_matrix) is None:
            return None
        if (Xdual := self.X_matrix_dual) is None:
            return Y
        return math.matmul(math.matmul(Xdual, Y), math.transpose(Xdual))

    @property
    def d_vector_dual(self) -> Optional[RealVector]:
        if (d := self.d_vector) is None:
            return None
        if (Xdual := self.X_matrix_dual) is None:
            return -d
        return -math.matmul(Xdual, d)

    def bargmann(self, numpy=False):
        X, Y, d = self.XYd(allow_none=False)
        if self.is_unitary:
            A, B, C = bargmann.wigner_to_bargmann_U(X, d)
        else:
            A, B, C = bargmann.wigner_to_bargmann_Choi(X, Y, d)
        if numpy:
            return math.asnumpy(A), math.asnumpy(B), math.asnumpy(C)
        return A, B, C

    def choi(
        self,
        cutoffs: Optional[Sequence[int]] = None,
        shape: Optional[Sequence[int]] = None,
        dual: bool = False,
    ):
        r"""Returns the Choi representation of the transformation.

        If specified, ``shape`` takes precedence over ``cutoffs``.
        The ``shape`` is in the order ``(out_L, in_L, out_R, in_R)``.

        Args:
            cutoffs: the cutoffs of the input and output modes
            shape: the shape of the Choi matrix
            dual: whether to return the dual Choi
        """
        N = self.num_modes
        if cutoffs is None:
            pass
        elif len(cutoffs) != N:
            raise ValueError(f"len(cutoffs) must be {self.num_modes} (got {len(cutoffs)})")

        shape = shape or tuple(cutoffs) * 4

        if self.is_unitary:
            shape = shape[: 2 * self.num_modes]
            U = self.U(shape[: self.num_modes])
            Udual = self.U(shape[self.num_modes :])
            if dual:
                return fock.U_to_choi(U=Udual, Udual=U)
            return fock.U_to_choi(U=U, Udual=Udual)

        X, Y, d = self.XYd(allow_none=False)
        choi = fock.wigner_to_fock_Choi(X, Y, d, shape=shape)
        if dual:
            n = len(shape) // 4
            N0 = list(range(0, n))
            N1 = list(range(n, 2 * n))
            N2 = list(range(2 * n, 3 * n))
            N3 = list(range(3 * n, 4 * n))
            choi = math.conj(math.transpose(choi, N1 + N0 + N3 + N2))  # if dual we flip out-in
        return choi

    def XYd(
        self, allow_none: bool = True
    ) -> Tuple[Optional[RealMatrix], Optional[RealMatrix], Optional[RealVector]]:
        r"""Returns the ```(X, Y, d)``` triple.

        Override in subclasses if computing ``X``, ``Y`` and ``d`` together is more efficient.
        """
        if allow_none:
            return self.X_matrix, self.Y_matrix, self.d_vector
        X = math.eye(2 * self.num_modes) if self.X_matrix is None else self.X_matrix
        Y = math.zeros_like(X) if self.Y_matrix is None else self.Y_matrix
        d = math.zeros_like(X[:, 0]) if self.d_vector is None else self.d_vector
        return X, Y, d

    def XYd_dual(
        self, allow_none: bool = True
    ) -> tuple[Optional[RealMatrix], Optional[RealMatrix], Optional[RealVector]]:
        r"""Returns the ```(X, Y, d)``` triple of the dual of the current transformation.

        Override in subclasses if computing ``Xdual``, ``Ydual`` and ``ddual`` together is more efficient.
        """
        if allow_none:
            return self.X_matrix_dual, self.Y_matrix_dual, self.d_vector_dual
        Xdual = math.eye(2 * self.num_modes) if self.X_matrix_dual is None else self.X_matrix_dual
        Ydual = math.zeros_like(Xdual) if self.Y_matrix_dual is None else self.Y_matrix_dual
        ddual = math.zeros_like(Xdual[:, 0]) if self.d_vector_dual is None else self.d_vector_dual
        return Xdual, Ydual, ddual

    def __getitem__(self, items) -> Callable:
        r"""Sets the modes on which the transformation acts.

        Allows transformations to be used as: ``output = transf[0,1](input)``,  e.g. acting on
        modes 0 and 1.
        """
        #  TODO: this won't work when we want to reuse the same op for different modes in a circuit.
        # i.e. `psi = op[0](psi); psi = op[1](psi)` is ok, but `circ = Circuit([op[0], op[1]])` won't work.
        if isinstance(items, int):
            modes = [items]
        elif isinstance(items, slice):
            modes = list(range(items.start, items.stop, items.step))
        elif isinstance(items, (Sequence, Iterable)):
            modes = list(items)
        else:
            raise ValueError(f"{items} is not a valid slice or list of modes.")
        if self.is_unitary:
            self.change_modes(modes, modes)
        else:
            self.change_modes(modes, modes, modes, modes)
        return self

    def __rshift__(self, other: Transformation):
        r"""Concatenates self with other (other after self).

        If any of the two is a circuit, all the ops in it migrate to the new circuit that is returned.
        E.g., ``circ = Sgate(1.0)[0,1] >> Dgate(0.2)[0] >> BSgate(np.pi/4)[0,1]``

        Args:
            other: another transformation

        Returns:
            Circuit: A circuit that concatenates self with other
        """
        from mrmustard.lab.circuit import Circuit  # pylint: disable=import-outside-toplevel

        ops1 = self._ops if isinstance(self, Circuit) else [self]
        ops2 = other._ops if isinstance(other, Circuit) else [other]
        return Circuit(ops1 + ops2)

    def __lshift__(self, other: Union[State, Transformation]):
        r"""Applies the dual of self to other.

        If other is a state, the dual of self is applied to the state.
        If other is a transformation, the dual of self is concatenated after other (in the dual sense).

        E.g.
        .. code-block::

            Sgate(0.1) << Coherent(0.5)   # state
            Sgate(0.1) << Dgate(0.2)      # transformation

        Args:
            other: a state or a transformation

        Returns:
            State: the state transformed via the dual transformation or the transformation
            concatenated after other
        """
        if isinstance(other, State):
            return self.dual(other)
        if isinstance(other, Transformation):
            return self >> other  # so that the dual is self.dual(other.dual(x))
        raise ValueError(
            f"{other} of type {other.__class__} is not a valid state or transformation."
        )

    def __repr__(self):
        class_name = self.name
        modes = self.modes

        parameters = {k: v for k, v in self.__dict__.items() if isinstance(v, (Constant, Variable))}
        param_str_rep = [
            f"{name}={repr(math.asnumpy(par.value))}" for name, par in parameters.items()
        ]

        params_str = ", ".join(sorted(param_str_rep))

        return f"{class_name}({params_str}, modes = {modes})".replace("\n", "")

    def __str__(self):
        class_name = self.name
        modes = self.modes
        return f"<{class_name} object at {hex(id(self))} acting on modes {modes}>"

    def _repr_markdown_(self):
        header = (
            f"##### {self.__class__.__qualname__} on modes {self.modes}\n"
            "|Parameters|dtype|Value|Bounds|Shape|Trainable|\n"
            "| :-:      | :-: | :-: | :-:  | :-: | :-:     |\n"
        )

        body = ""
        with np.printoptions(precision=6, suppress=True):
            parameters = {
                k: v for k, v in self.__dict__.items() if isinstance(v, (Constant, Variable))
            }
            for name, par in parameters.items():
                par_value = repr(math.asnumpy(par.value)).replace("\n", "<br>")
                body += (
                    f"| {name}"
                    f"| {par.value.dtype.name}"
                    f"| {par_value}"
                    f"| {str(getattr(par.value, 'bounds', 'None'))}"
                    f"| {par.value.shape}"
                    f"| {str(math.is_trainable(par.value))}"
                    "|\n"
                )

        return header + body


class Unitary(Transformation):
    r"""
    A unitary transformation.

    Args:
        name: The name of this unitary.
        modes: The modes that this unitary acts on.
    """

    def __init__(self, name: str, modes: list[int]):
        super().__init__(name=name, modes_in_ket=modes, modes_out_ket=modes)
        self.is_unitary = True

    def value(self, shape: Tuple[int]):
        return self.U(shape=shape)

    def _transform_fock(self, state: State, dual=False) -> State:
        op_idx = [state.modes.index(m) for m in self.modes]
        U = self.U(cutoffs=[state.cutoffs[i] for i in op_idx])
        if state.is_hilbert_vector:
            return State(ket=fock.apply_kraus_to_ket(U, state.ket(), op_idx), modes=state.modes)
        return State(dm=fock.apply_kraus_to_dm(U, state.dm(), op_idx), modes=state.modes)

    def U(
        self,
        cutoffs: Optional[Sequence[int]] = None,
        shape: Optional[Sequence[int]] = None,
    ):
        r"""Returns the unitary representation of the transformation.

        If specified, ``shape`` takes precedence over ``cutoffs``.
        ``shape`` is in the order ``(out, in)``.

        Note that for a unitary transformation on N modes, ``len(cutoffs)`` is ``N``
        and ``len(shape)`` is ``2N``.

        Args:
            cutoffs: the cutoffs of the input and output modes
            shape: the shape of the unitary matrix

        Returns:
            ComplexTensor: the unitary matrix in Fock representation
        """
        if cutoffs is None:
            pass
        elif len(cutoffs) != self.num_modes:
            raise ValueError(f"len(cutoffs) must be {self.num_modes} (got {len(cutoffs)})")
        shape = shape or tuple(cutoffs) * 2
        X, _, d = self.XYd(allow_none=False)
        return fock.wigner_to_fock_U(X, d, shape=shape)

    def __eq__(self, other):
        r"""Returns ``True`` if the two transformations are equal."""
        if not isinstance(other, Unitary):
            return False
        if not (self.is_gaussian and other.is_gaussian):
            return np.allclose(
                self.U(cutoffs=[settings.EQ_TRANSFORMATION_CUTOFF] * 2 * self.num_modes),
                other.U(cutoffs=[settings.EQ_TRANSFORMATION_CUTOFF] * 2 * self.num_modes),
                rtol=settings.EQ_TRANSFORMATION_RTOL_FOCK,
            )
        sX, sY, sd = self.XYd(allow_none=False)
        oX, oY, od = other.XYd(allow_none=False)
        assert np.isclose(np.linalg.norm(sY), 0)
        assert np.isclose(np.linalg.norm(oY), 0)
        return np.allclose(sX, oX) and np.allclose(sd, od)


class Channel(Transformation):
    r"""
    A quantum channel.

    Args:
        name: The name of this channel.
        modes: The modes that this channel acts on.
    """

    def __init__(self, name: str, modes: list[int]):
        super().__init__(
            name=name,
            modes_in_ket=modes,
            modes_out_ket=modes,
            modes_in_bra=modes,
            modes_out_bra=modes,
        )
        self.is_unitary = False

    def _transform_fock(self, state: State, dual: bool = False) -> State:
        op_idx = [state.modes.index(m) for m in self.modes]
        choi = self.choi(cutoffs=[state.cutoffs[i] for i in op_idx], dual=dual)
        if state.is_hilbert_vector:
            return State(dm=fock.apply_choi_to_ket(choi, state.ket(), op_idx), modes=state.modes)
        return State(dm=fock.apply_choi_to_dm(choi, state.dm(), op_idx), modes=state.modes)

    def value(self, shape: Tuple[int]):
        return self.choi(shape=shape)

    def __eq__(self, other):
        r"""Returns ``True`` if the two transformations are equal."""
        if not isinstance(other, Channel):
            return False
        if not (self.is_gaussian and other.is_gaussian):
            return np.allclose(
                self.choi(cutoffs=[settings.EQ_TRANSFORMATION_CUTOFF] * 4 * self.num_modes),
                other.choi(cutoffs=[settings.EQ_TRANSFORMATION_CUTOFF] * 4 * self.num_modes),
                rtol=settings.EQ_TRANSFORMATION_RTOL_FOCK,
            )
        sX, sY, sd = self.XYd(allow_none=False)
        oX, oY, od = other.XYd(allow_none=False)
        return np.allclose(sX, oX) and np.allclose(sY, oY) and np.allclose(sd, od)
