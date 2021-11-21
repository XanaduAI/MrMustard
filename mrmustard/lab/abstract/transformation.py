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
from mrmustard.utils.types import *
from mrmustard.utils import graphics
from mrmustard import settings


class Transformation:
    r"""
    Base class for all transformations.
    Note that measurements are CP but not TP, so they have their own abstract class.
    Transformations include:
        * unitary transformations
        * non-unitary CPTP channels
    """

    _bell = None  # single-mode TMSV state for gaussian to fock transformation
    is_unitary = True  # whether the transformation is unitary

    def __call__(self, state: State) -> State:
        return self.transform_gaussian(state) if state.is_gaussian else self.transform_fock(state)

    @property
    def bell(self):
        r"""The N-mode two-mode squeezed vacuum for the choi-jamiolkowksi isomorphism"""
        if self._bell is None:
            cov = gaussian.two_mode_squeezed_vacuum_cov(
                r=settings.CHOI_R, phi=0.0, hbar=settings.HBAR
            )
            means = gaussian.vacuum_means(num_modes=2, hbar=settings.HBAR)
            bell = bell_single = State(cov=cov, means=means, is_mixed=False)
            for _ in self.modes[1:]:
                bell = bell & bell_single
            tot = 2 * len(self.modes)
            order = tuple(range(0, tot, 2)) + tuple(range(1, tot, 2))
            self._bell = bell[order]
        return self._bell

    def transform_gaussian(self, state: State) -> State:
        r"""
        Transforms a state in Gaussian representation.
        """
        cov, means = gaussian.CPTP(state.cov, state.means, *self.XYd, self.modes)
        return State(cov=cov, means=means, is_mixed=state.is_mixed or not self.is_unitary)

    def transform_fock(self, state: State) -> State:
        r"""
        Transforms a state in Fock representation.
        """
        if self.is_unitary:
            transformation = self.U(cutoffs=state.cutoffs)
        else:
            transformation = self.choi(cutoffs=state.cutoffs)
        new_state = fock.CPTP(
            transformation=transformation,
            fock_state=state._fock,
            transformation_is_unitary=self.is_unitary,
            state_is_mixed=state.is_mixed,
        )
        return State(
            fock=new_state, is_mixed=not self.is_unitary or state.is_mixed
        )  # TODO: is_mixed is not computed correctly (non-unitary maps can return pure states)

    def __repr__(self):
        table = Table(title=f"{self.__class__.__qualname__}")
        table.add_column("Parameters")
        table.add_column("dtype")
        table.add_column("Value")
        table.add_column("Shape")
        table.add_column("Trainable")
        with np.printoptions(precision=6, suppress=True):
            for name in self.param_names:
                par = self.__dict__[name]
                table.add_row(
                    name,
                    par.dtype.name,
                    f"{np.array(par)}",
                    f"{par.shape}",
                    str(self.__dict__["_" + name + "_trainable"]),
                )
            lst = [
                f"{name}={np.array(np.atleast_1d(self.__dict__[name]))}"
                for name in self.param_names
            ]
            repr_string = f"{self.__class__.__qualname__}({', '.join(lst)})" + (
                f"[{self._modes}]" if self._modes is not None else ""
            )
        rprint(table)
        return repr_string

    @property
    def modes(self) -> Sequence[int]:
        if self._modes in (None, []):
            X,Y,d = self.XYd
            if d is not None:
                self._modes = list(range(d.shape[-1] // 2))
            elif X is not None:
                self._modes = list(range(X.shape[-1] // 2))
            elif Y is not None:
                self._modes = list(range(Y.shape[-1] // 2))
        return self._modes

    @modes.setter
    def modes(self, modes: List[int]):
        r"""
        Sets the modes on which the transformation acts.
        """
        self._validate_modes(modes)
        self._modes = modes

    def _validate_modes(self, modes):
        pass  # override in subclasses as needed

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
    def XYd(self) -> Tuple[Optional[Matrix], Optional[Matrix], Optional[Vector]]:
        return [self.X_matrix, self.Y_matrix, self.d_vector]

    @property
    def XYd_dual(self) -> Tuple[Optional[Matrix], Optional[Matrix], Optional[Vector]]:
        r"""
        Returns the (X, Y, d) triple of the dual of the current transformation.
        """
        X, Y, d = self.XYd
        X_dual = math.inv(X) if X is not None else None
        Y_dual = Y
        if Y is not None:
            Y_dual = math.matmul(X_dual, math.matmul(Y, math.transpose(X_dual))) if X_dual is not None else Y
        d_dual = math.matvec(X_dual, d)
        return X_dual, Y_dual, d_dual
            
    def U(self, cutoffs: Sequence[int]):
        "Returns the unitary representation of the transformation"
        if not self.is_unitary:
            return None
        choi_state = self(self.bell)
        return fock.fock_representation(
            choi_state.cov,
            choi_state.means,
            shape=cutoffs * 2,
            is_unitary=True,
            choi_r=settings.CHOI_R,
        )

    def choi(self, cutoffs: Sequence[int]):
        "Returns the Choi representation of the transformation"
        if self.is_unitary:
            U = self.U(cutoffs)
            return fock.U_to_choi(U)
        else:
            choi_state = self(self.bell)
            return fock.fock_representation(
                choi_state.cov,
                choi_state.means,
                shape=cutoffs * 4,
                is_unitary=False,
                choi_r=settings.CHOI_R,
            )

    def trainable_parameters(self) -> Dict[str, List[Trainable]]:
        return {"symplectic": [], "orthogonal": [], "euclidean": self._trainable_parameters}

    def __getitem__(self, items) -> Callable:
        r"""
        Allows transformations to be used as:
        output = op[0,1](input)  # e.g. acting on modes 0 and 1
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

    def __rshift__(self, other):
        r"""
        Concatenates self with other.
        E.g.
        `circ = Sgate[0,1] >> Dgate[0] >> BSgate[0,1] >> PNRdetector[0]`  # TODO: use __class_getitem__ for compiler stuff
        Arguments:
            other: another transformation
        Returns:
            A new transformation that is the concatenation of the two.
        """
        from mrmustard.lab import Circuit  # this is called at runtime so it's ok
        return Circuit([self, other])

    def __eq__(self, other):
        r"""
        Returns True if the two transformations are equal.
        """
        if not isinstance(other, Transformation):
            return False
        if self.is_gaussian and other.is_gaussian:
            for s,o in zip(self.XYd, other.XYd):
                if not np.allclose(s, o, rtol=settings.EQ_TRANSFORMATION_RTOL_GAUSS):
                    return False
            return True
        else:
            return np.allclose(self.choi(cutoffs=[settings.EQ_TRANSFORMATION_CUTOFF]*self.num_modes),
            other.choi(cutoffs=[settings.EQ_TRANSFORMATION_CUTOFF]*self.num_modes), rtol=settings.EQ_TRANSFORMATION_RTOL_FOCK)