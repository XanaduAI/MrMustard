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

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from rich.table import Table
from rich import print as rprint

from mrmustard.physics import gaussian, fock
from mrmustard.lab.abstract.state import State
from mrmustard.types import *
from mrmustard.utils import graphics
from mrmustard.utils.parametrized import Parametrized
from mrmustard import settings
from mrmustard.math import Math

math = Math()


class Transformation:
    r"""Base class for all Transformations."""
    _bell = None  # single-mode TMSV state for gaussian-to-fock conversion
    is_unitary = True  # whether the transformation is unitary (True by default)

    def primal(self, state: State) -> State:
        r"""Applies ``self`` (a ``Transformation``) to other (a ``State``) and returns the transformed state.

        Args:
            state (State): the state to transform

        Returns:
            State: the transformed state
        """
        if state.is_gaussian:
            new_state = self.transform_gaussian(state, dual=False)
        else:
            new_state = self.transform_fock(state, dual=False)
        return new_state

    def dual(self, state: State) -> State:
        r"""Applies the dual of self (dual of a ``Transformation``) to other (a ``State``) and returns the transformed state.

        Args:
            state (State): the state to transform

        Returns:
            State: the transformed state
        """
        if state.is_gaussian:
            new_state = self.transform_gaussian(state, dual=True)
        else:
            new_state = self.transform_fock(state, dual=True)
        return new_state

    @property
    def bell(self):
        r"""The N-mode two-mode squeezed vacuum for the choi-jamiolkowksi isomorphism."""
        if self._bell is None:
            cov = gaussian.two_mode_squeezed_vacuum_cov(
                r=settings.CHOI_R, phi=0.0, hbar=settings.HBAR
            )
            means = gaussian.vacuum_means(num_modes=2, hbar=settings.HBAR)
            bell = bell_single = State(cov=cov, means=means)
            for _ in range(self.num_modes - 1):
                bell = bell & bell_single
            tot = 2 * self.num_modes
            order = tuple(range(0, tot, 2)) + tuple(range(1, tot, 2))
            self._bell = bell.get_modes(order)
        return self._bell[self.modes + [m + self.num_modes for m in self.modes]]

    def transform_gaussian(self, state: State, dual: bool) -> State:
        r"""Transforms a Gaussian state into a Gaussian state.

        Args:
            state (State): the state to transform
            dual (bool): whether to apply the dual channel

        Returns:
            State: the transformed state
        """
        X, Y, d = self.XYd if not dual else self.XYd_dual
        cov, means = gaussian.CPTP(state.cov, state.means, X, Y, d, state.modes, self.modes)
        new_state = State(
            cov=cov, means=means, modes=state.modes
        )  # NOTE: assumes modes don't change
        return new_state

    def transform_fock(self, state: State, dual: bool) -> State:
        r"""Transforms a state in Fock representation.

        Args:
            state (State): the state to transform
            dual (bool): whether to apply the dual channel

        Returns:
            State: the transformed state
        """
        if self.is_unitary:
            U = self.U(cutoffs=state.cutoffs)
            transformation = fock.math.dagger(U) if dual else U
        else:
            transformation = self.choi(cutoffs=state.cutoffs)
            if dual:
                n = len(state.cutoffs)
                N0 = list(range(0, n))
                N1 = list(range(n, 2 * n))
                N2 = list(range(2 * n, 3 * n))
                N3 = list(range(3 * n, 4 * n))
                transformation = fock.math.transpose(transformation, N3 + N0 + N1 + N2)
        new_fock = fock.CPTP(
            transformation=transformation,
            fock_state=state.ket(state.cutoffs) if state.is_pure else state.dm(state.cutoffs),
            transformation_is_unitary=self.is_unitary,
            state_is_dm=state.is_mixed,
        )
        if state.is_mixed or not self.is_unitary:
            return State(dm=new_fock, modes=state.modes)
        else:
            return State(ket=new_fock, modes=state.modes)

    @property
    def modes(self) -> Sequence[int]:
        if self._modes in (None, []):
            X, Y, d = self.XYd
            if d is not None:
                self._modes = list(range(d.shape[-1] // 2))
            elif X is not None:
                self._modes = list(range(X.shape[-1] // 2))
            elif Y is not None:
                self._modes = list(range(Y.shape[-1] // 2))
        return self._modes

    @modes.setter
    def modes(self, modes: List[int]):
        r"""Sets the modes on which the transformation acts."""
        self._validate_modes(modes)
        self._modes = modes

    @property
    def num_modes(self) -> int:
        r"""The number of modes on which the transformation acts."""
        return len(self.modes)

    def _validate_modes(self, modes):
        pass

    @property
    def X_matrix(self) -> Optional[Matrix]:
        return None

    @property
    def Y_matrix(self) -> Optional[Matrix]:
        return None

    @property
    def d_vector(self) -> Optional[Vector]:
        return None

    @property
    def X_matrix_dual(self) -> Optional[Matrix]:
        if (X := self.X_matrix) is None:
            return None
        return gaussian.math.inv(X)

    @property
    def Y_matrix_dual(self) -> Optional[Matrix]:
        if (Y := self.Y_matrix) is None:
            return None
        elif (Xdual := self.X_matrix_dual) is None:
            return Y
        else:
            return math.matmul(math.matmul(Xdual, Y), math.transpose(Xdual))

    @property
    def d_vector_dual(self) -> Optional[Vector]:
        if (d := self.d_vector) is None:
            return None
        elif (Xdual := self.X_matrix_dual) is None:
            return d
        else:
            return math.matmul(Xdual, d)

    @property
    def XYd(self) -> Tuple[Optional[Matrix], Optional[Matrix], Optional[Vector]]:
        r"""Returns the ```(X, Y, d)``` triple.

        Override in subclasses if computing ``X``, ``Y`` and ``d`` together is more efficient.
        """
        return self.X_matrix, self.Y_matrix, self.d_vector

    @property
    def XYd_dual(self) -> Tuple[Optional[Matrix], Optional[Matrix], Optional[Vector]]:
        r"""Returns the ```(X, Y, d)``` triple of the dual of the current transformation.

        Override in subclasses if computing ``Xdual``, ``Ydual`` and ``ddual`` together is more efficient.
        """
        return self.X_matrix_dual, self.Y_matrix_dual, self.d_vector_dual

    @property
    def is_phase_covariant(self) -> bool:
        X, Y, d = self.XYd
        if d is not None:
            return False
        if X is not None and not math.allclose(self.X, math.diag(math.diag_part(self.X))):
            return False
        if Y is not None and not math.allclose(self.Y, math.diag(math.diag_part(self.Y))):
            return False
        return True

    def U(self, cutoffs: Sequence[int]):
        r"""Returns the unitary representation of the transformation."""
        if not self.is_unitary:
            return None
        choi_state = self.bell >> self
        return fock.fock_representation(
            choi_state.cov,
            choi_state.means,
            shape=cutoffs * 2,
            return_unitary=True,
            choi_r=settings.CHOI_R,
        )

    def choi(self, cutoffs: Sequence[int]):
        r"""Returns the Choi representation of the transformation."""
        if self.is_unitary:
            U = self.U(cutoffs)
            return fock.U_to_choi(U)
        else:
            choi_state = self.bell >> self
            choi_op = fock.fock_representation(
                choi_state.cov,
                choi_state.means,
                shape=cutoffs * 4,
                return_unitary=False,
                choi_r=settings.CHOI_R,
            )
            return choi_op

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
        self.modes = modes
        return self

    # TODO: use __class_getitem__ for compiler stuff

    def __rshift__(self, other: Transformation):
        r"""Concatenates self with other (other after self).

        If any of the two is a circuit, all the ops in it migrate to the new circuit that is returned.
        E.g., ``circ = Sgate(1.0)[0,1] >> Dgate(0.2)[0] >> BSgate(np.pi/4)[0,1]``

        Args:
            other: another transformation

        Returns:
            Circuit: A circuit that concatenates self with other
        """
        from mrmustard.lab import Circuit  # this is called at runtime so it's ok

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
        elif isinstance(other, Transformation):
            return self >> other  # so that the dual is self.dual(other.dual(x))
        else:
            raise ValueError(f"{other} is not a valid state or transformation.")

    def __eq__(self, other):
        r"""Returns ``True`` if the two transformations are equal."""
        if not isinstance(other, Transformation):
            return False
        if self.is_gaussian and other.is_gaussian:
            sX, sY, sd = self.XYd
            oX, oY, od = other.XYd
            if sX is None:
                if oX is not None:
                    if not np.allclose(
                        oX, np.eye(oX.shape[0]), rtol=settings.EQ_TRANSFORMATION_RTOL_GAUSS
                    ):
                        return False
            if oX is None:
                if sX is not None:
                    if not np.allclose(
                        sX, np.eye(sX.shape[0]), rtol=settings.EQ_TRANSFORMATION_RTOL_GAUSS
                    ):
                        return False
            if sX is not None and oX is not None:
                if not np.allclose(sX, oX):
                    return False
            if sY is None:
                if oY is not None:
                    if not np.allclose(
                        oY, np.zeros_like(oY), rtol=settings.EQ_TRANSFORMATION_RTOL_GAUSS
                    ):
                        return False
            if oY is None:
                if sY is not None:
                    if not np.allclose(
                        sY, np.zeros_like(sY), rtol=settings.EQ_TRANSFORMATION_RTOL_GAUSS
                    ):
                        return False
            if sY is not None and oY is not None:
                if not np.allclose(sY, oY):
                    return False
            if sd is None:
                if od is not None:
                    if not np.allclose(
                        sd, np.zeros_like(sd), rtol=settings.EQ_TRANSFORMATION_RTOL_GAUSS
                    ):
                        return False
            if od is None:
                if sd is not None:
                    if not np.allclose(
                        sd, np.zeros_like(sd), rtol=settings.EQ_TRANSFORMATION_RTOL_GAUSS
                    ):
                        return False
            if sd is not None and od is not None:
                if not np.allclose(sd, od):
                    return False
            return True
        else:
            return np.allclose(
                self.choi(cutoffs=[settings.EQ_TRANSFORMATION_CUTOFF] * self.num_modes),
                other.choi(cutoffs=[settings.EQ_TRANSFORMATION_CUTOFF] * self.num_modes),
                rtol=settings.EQ_TRANSFORMATION_RTOL_FOCK,
            )

    def __repr__(self):
        table = Table(title=f"{self.__class__.__qualname__}")
        table.add_column("Parameters")
        table.add_column("dtype")
        table.add_column("Value")
        table.add_column("Bounds")
        table.add_column("Shape")
        table.add_column("Trainable")
        with np.printoptions(precision=6, suppress=True):
            for name in self._param_names:
                par = self.__dict__[name]
                table.add_row(
                    name,
                    par.dtype.name,
                    f"{np.array(par)}",
                    str(self.__dict__["_" + name + "_bounds"]),
                    f"{par.shape}",
                    str(self.__dict__["_" + name + "_trainable"]),
                )
        rprint(table)
        return ""
