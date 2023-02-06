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

from __future__ import annotations

import numpy as np

from mrmustard import settings
from mrmustard.math import Math
from mrmustard.physics import fock, gaussian
from mrmustard.training.parameter import Parameter
from mrmustard.types import (
    Callable,
    Iterable,
    List,
    Matrix,
    Optional,
    Sequence,
    Tuple,
    Union,
    Vector,
)

from .state import State

math = Math()


class Transformation:
    r"""Base class for all Transformations."""
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
            cov=cov, means=means, modes=state.modes, _norm=state.norm
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
        op_idx = [state.modes.index(m) for m in self.modes]
        if self.is_unitary:
            U = self.U(cutoffs=[state.cutoffs[i] for i in op_idx])
            U = math.dagger(U) if dual else U
            if state.is_pure:
                return State(ket=fock.apply_kraus_to_ket(U, state.ket(), op_idx), modes=state.modes)
            return State(dm=fock.apply_kraus_to_dm(U, state.dm(), op_idx), modes=state.modes)
        else:
            choi = self.choi(cutoffs=state.cutoffs)
            n = state.num_modes
            N0 = list(range(0, n))
            N1 = list(range(n, 2 * n))
            N2 = list(range(2 * n, 3 * n))
            N3 = list(range(3 * n, 4 * n))
            if dual:
                choi = math.transpose(choi, N1 + N0 + N3 + N2)  # we flip out-in

            if state.is_pure:
                return State(
                    dm=fock.apply_choi_to_ket(choi, state.ket(), op_idx), modes=state.modes
                )
            return State(dm=fock.apply_choi_to_dm(choi, state.dm(), op_idx), modes=state.modes)

    def label(self, decimals: Optional[int] = None) -> str:
        r"""Returns a string representation of the transformation for use in
        a circuit diagram.

        Args:
            decimals (int): the number of decimals to use for the parameters

        Returns:
            str: the string representation of the transformation

        Example:
            BSgate(0.54, 0.12) -> "BS(0.54, 0.12)"
            Ggate(2) -> "G(2)"
        """
        if decimals is None:
            return self.short_name
        label = self.short_name + "("
        for name in self.param_names.values():
            par = getattr(self, name).value
            show = False
            if math.asnumpy(par).ndim == 0:
                label += f"{math.asnumpy(par):.{decimals}g}, "
                show = True
            if not show:
                label += f"{len(self.modes)}, "
        return label[:-2] + ")"

    @property
    def modes(self) -> Sequence[int]:
        """Returns the list of modes on which the transformation acts on."""
        if self._modes in (None, []):
            for elem in self.XYd:
                if elem is not None:
                    self._modes = list(range(elem.shape[-1] // 2))
                    break
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
        if (Xdual := self.X_matrix_dual) is None:
            return Y
        return math.matmul(math.matmul(Xdual, Y), math.transpose(Xdual))

    @property
    def d_vector_dual(self) -> Optional[Vector]:
        if (d := self.d_vector) is None:
            return None
        if (Xdual := self.X_matrix_dual) is None:
            return d
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
        X, _, d = self.XYd
        return fock.wigner_to_fock_U(
            X if X is not None else math.eye(2 * self.num_modes),
            d if d is not None else math.zeros((2 * self.num_modes,)),
            shape=cutoffs * 2 if len(cutoffs) == self.num_modes else cutoffs,
        )

    def choi(self, cutoffs: Sequence[int]):
        r"""Returns the Choi representation of the transformation."""
        if self.is_unitary:
            U = self.U(cutoffs)
            return fock.U_to_choi(U)
        X, Y, d = self.XYd
        return fock.wigner_to_fock_Choi(
            X if X is not None else math.eye(2 * self.num_modes),
            Y if Y is not None else math.zeros((2 * self.num_modes, 2 * self.num_modes)),
            d if d is not None else math.zeros((2 * self.num_modes,)),
            shape=cutoffs * 4 if len(cutoffs) == self.num_modes else cutoffs,
        )

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

    # pylint: disable=import-outside-toplevel,cyclic-import
    def __rshift__(self, other: Transformation):
        r"""Concatenates self with other (other after self).

        If any of the two is a circuit, all the ops in it migrate to the new circuit that is returned.
        E.g., ``circ = Sgate(1.0)[0,1] >> Dgate(0.2)[0] >> BSgate(np.pi/4)[0,1]``

        Args:
            other: another transformation

        Returns:
            Circuit: A circuit that concatenates self with other
        """
        from ..circuit import (
            Circuit,
        )

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
        raise ValueError(f"{other} is not a valid state or transformation.")

    # pylint: disable=too-many-branches,too-many-return-statements
    def __eq__(self, other):
        r"""Returns ``True`` if the two transformations are equal."""
        if not isinstance(other, Transformation):
            return False
        if not (self.is_gaussian and other.is_gaussian):
            return np.allclose(
                self.choi(cutoffs=[settings.EQ_TRANSFORMATION_CUTOFF] * self.num_modes),
                other.choi(cutoffs=[settings.EQ_TRANSFORMATION_CUTOFF] * self.num_modes),
                rtol=settings.EQ_TRANSFORMATION_RTOL_FOCK,
            )

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

    def __repr__(self):
        class_name = self.__class__.__name__
        modes = self.modes

        parameters = {k: v for k, v in self.__dict__.items() if isinstance(v, Parameter)}
        param_str_rep = [
            f"{name}={repr(math.asnumpy(par.value))}" for name, par in parameters.items()
        ]

        params_str = ", ".join(sorted(param_str_rep))

        return f"{class_name}({params_str}, modes = {modes})".replace("\n", "")

    def __str__(self):
        class_name = self.__class__.__name__
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
            parameters = {k: v for k, v in self.__dict__.items() if isinstance(v, Parameter)}
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
