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
from mrmustard.types import *
from abc import ABC, abstractmethod, abstractproperty
from itertools import product
import numpy as np
from mrmustard.math import Math

math = Math()


class XPTensor(ABC):
    r"""A representation of Matrices and Vectors in phase space.

    Tensors in phase space have a ``(2n, 2n)`` or ``(2n,)`` shape where n is the number of modes.

    There are two main orderings:
        - xxpp: matrix is a `2\times 2` block matrix where each block is an `xx`, `xp`, `px`, `pp` block on all modes.
        - xpxp: matrix is a `n\times n` block matrix of `2\times 2` blocks each corresponding to a mode or a coherence between modes
    This creates some difficulties when we need to work in a mode-wise fashion, especially whith coherences.
    We solve this problem by reshaping the matrices to `(n,m,2,2)` and vectors to `(n,2)`.

    We call `n` the outmodes and `m` the inmodes.
    Off-diagonal matrices like coherences have all the outmodes different than the inmodes.
    Diagonal matrices like coviariances and symplectic transformations have the same outmodes as the inmodes.
    Vectors have only outmodes.

    XPTensor objects are sparse, in the sense that they support implicit operations between modes where one or more tensors are undefined.
    There are two types of behaviour:
        * like_0 (default): in modes where the tensor is undefined, it's like having a zero (a zero matrix)
        * like_1: in modes where the tensor is undefined, it's like having a one (an identity matrix)

    For example, in the expression :math:`X @ means + d` where `X` is a symplectic matrix and `d` is a displacement vector,
    if `X` is undefined it's like having the identity and the matrix product simply returns means, while in the expression
    `means + d` if `d` is undefined it simply returns `means`. In this example no operation was actually computed.
    Thanks to sparsity we can represent graph states and transformations on graph states as XPTensor objects.

    Args:
        tensor: The tensor in (n,m,2,2) or (n,2) order.
        modes: a list of modes for a diagonal matrix or a vector and a tuple of two lists for a coherence (not optional for a coherence)
        like_0: Whether the null tensor behaves like 0 under addition (e.g. the noise matrix Y)
        like_1: Whether the null tensor behaves like 1 under multiplication (e.g. a symplectic transformation matrix)
    """

    @abstractmethod  # so that XPTensor can't be instantiated directly
    def __init__(
        self,
        tensor: Optional[Tensor],
        like_0: bool,
        isVector: bool,
        modes: Tuple[List[int], List[int]],
    ):

        self.like_0 = like_0
        self.shape = (
            None if tensor is None else tensor.shape[: len(tensor.shape) // 2]
        )  # only (N,M) or (N,)
        self.ndim = None if tensor is None else len(self.shape)
        self.isVector = isVector
        if self.ndim == 1 and not self.isVector:
            raise ValueError(
                f"tensor shape incompatible with isVector={isVector} (expected len(tensor.shape)==4, got {len(tensor.shape)})"
            )
        if self.ndim == 2 and self.isVector:
            raise ValueError(
                f"tensor shape incompatible with isVector={isVector} (expected len(tensor.shape)==2, got {len(tensor.shape)})"
            )
        if self.isVector and self.like_1:
            raise ValueError("vectors should be like_0")
        self.tensor = tensor
        if not (set(modes[0]) == set(modes[1]) or set(modes[0]).isdisjoint(modes[1])):
            raise ValueError(
                "The inmodes and outmodes should either contain the same modes or be disjoint"
            )
        self.modes = modes

    @property
    def dtype(self):
        return None if self.tensor is None else self.tensor.dtype

    @property
    def outmodes(self) -> List[int]:
        return self.modes[0]

    @property
    def inmodes(self) -> List[int]:
        return self.modes[1]

    @property
    def num_modes(self) -> int:
        # TODO: raise warning for coherence blocks?
        return len(self.outmodes)

    @property
    def isMatrix(self) -> Optional[bool]:
        return not self.isVector

    @property
    def isCoherence(self) -> Optional[bool]:
        return self.isMatrix and self.outmodes != self.inmodes

    @property
    def like_1(self) -> bool:
        return not self.like_0

    @property
    def T(self) -> XPMatrix:
        if self.isVector:
            raise ValueError("Cannot transpose a vector")
        if self.tensor is None:
            return self
        return XPMatrix(
            math.transpose(self.tensor, (1, 0, 3, 2)),
            self.like_0,
            self.like_1,
            (self.inmodes, self.outmodes),
        )

    def to_xpxp(self) -> Optional[Union[Matrix, Vector]]:
        if self.tensor is None:
            return None
        tensor = math.transpose(
            self.tensor, (0, 2, 1, 3) if self.isMatrix else (0, 1)
        )  # from NN22 to N2N2 or from N2 to N2
        return math.reshape(tensor, [2 * s for s in self.shape])

    def to_xxpp(self) -> Optional[Union[Matrix, Vector]]:
        if self.tensor is None:
            return None
        tensor = math.transpose(
            self.tensor, (2, 0, 3, 1) if self.isMatrix else (1, 0)
        )  # from NN22 to 2N2N or from N2 to 2N
        return math.reshape(tensor, [2 * s for s in self.shape])

    def __array__(self):
        return self.to_xxpp()

    def modes_first(self) -> Optional[Tensor]:
        return self.tensor

    def modes_last(self) -> Optional[Tensor]:
        if self.tensor is None:
            return None
        return math.transpose(self.tensor, (2, 3, 0, 1) if self.isMatrix else (0, 1))  # 22NM or 2N

    def clone(self, times: int, modes=None) -> XPtensor:
        r"""Create a new XPTensor made by cloning the system a given number of times
        (the modes are reset by default unless specified).
        """
        if self.tensor is None:
            return self
        if times == 1:
            return self
        if self.isMatrix:
            tensor = math.expand_dims(self.modes_last(), axis=4)  # shape = [2,2,N,N,1]
            tensor = math.tile(tensor, (1, 1, 1, 1, times))  # shape = [2,2,N,N,T]
            tensor = math.diag(tensor)  # shape = [2,2,N,N,T,T]
            tensor = math.transpose(tensor, (0, 1, 2, 4, 3, 5))  # shape = [2,2,N,T,N,T]
            tensor = math.reshape(
                tensor, (2, 2, tensor.shape[2] * times, tensor.shape[4] * times)
            )  # shape = [2,2,NT,NT] = [2,2,O,O]
            tensor = math.transpose(tensor, (2, 3, 0, 1))  # shape = [NT,NT,2,2]
            return XPMatrix(tensor, self.like_0, self.like_1, ([], []) if modes is None else modes)
        else:
            tensor = math.tile(
                self.expand_dims(self.modes_last(), axis=2), (1, 1, times)
            )  # shape = [2,N,T]
            tensor = math.reshape(tensor, (2, -1))  # shape = [2,NT] = [2,O]
            tensor = math.transpose(tensor, (1, 0))  # shape = [NT,2] = [O,2]
            return XPVector(tensor, [] if modes is None else modes)

    def clone_like(self, other: XPTensor):
        r"""Create a new XPTensor with the same shape and modes as other.

        The new tensor has the same content as self, cloned as many times as necessary to match the
        shape and modes of other. The other properties are kept as is.

        Args:
            other: The tensor to be cloned.

        Returns:
            Tensor: A new XPTensor with the same shape and modes as other.
        """
        if other.shape == self.shape:
            return self
        if self.isCoherence:
            raise ValueError("Cannot clone a coherence block")
        if bool(other.num_modes % self.num_modes):
            raise ValueError(
                f"No integer multiple of {self.num_modes} modes fits into {other.num_modes} modes"
            )
        times = other.num_modes // self.num_modes
        if self.isVector == other.isVector:
            tensor = self.clone(times, modes=other.modes).tensor
        else:
            raise ValueError("Cannot clone a vector into a matrix or viceversa")
        if self.isMatrix:
            return XPMatrix(tensor, self.like_0, self.like_1, (other.outmodes, other.inmodes))
        else:
            return XPVector(tensor, other.outmodes)

    ####################################################################################################################
    # Operators
    ####################################################################################################################

    def __rmul__(self, other: Scalar) -> XPTensor:
        "Implements the operation ``self * other``"
        if self.tensor is None:
            if self.like_1:
                raise NotImplementedError("Cannot multiply a scalar and a like_1 null tensor yet")
            else:
                return self
        self.tensor = other * self.tensor
        return self

    def __mul__(self, other: Scalar) -> Optional[XPTensor]:
        return other * self

    def __matmul__(self, other: Union[XPMatrix, XPVector]) -> Union[XPMatrix, XPVector, Scalar]:
        if not isinstance(other, (XPMatrix, XPVector)):
            raise TypeError(
                f"Unsupported operand type(s) for @: '{self.__class__.__qualname__}' and '{other.__class__.__qualname__}'"
            )
        # TODO: move mode-check at beginning?

        # both are None
        if self.tensor is None and other.tensor is None:
            if self.isMatrix and other.isMatrix:
                return XPMatrix(None, like_1=self.like_1 and other.like_1)
            elif self.isVector or other.isVector:
                return XPVector(None)
        # either is None
        if self.tensor is None:
            return self if self.like_0 else other
        if other.tensor is None:
            return other if other.like_0 else self
        # Now neither self nor other is None
        if self.isMatrix and other.isMatrix:
            tensor, modes = self._mode_aware_matmul(other)
            return XPMatrix(tensor, like_1=self.like_1 and other.like_1, modes=modes)
        elif self.isMatrix and other.isVector:
            tensor, modes = self._mode_aware_matmul(other)
            return XPVector(
                tensor, modes[0]
            )  # TODO: check if we can output modes as a list in _mode_aware_matmul
        elif self.isVector and other.isMatrix:
            tensor, modes = other.T._mode_aware_matmul(self)
            return XPVector(tensor, modes[0])
        else:  # self.isVector and other.isVector:
            return self._mode_aware_vecvec(other)  # NOTE: this is a scalar, not an XPTensor

    def _mode_aware_matmul(
        self, other: Union[XPMatrix, XPVector]
    ) -> Tuple[Tensor, Tuple[List[int], List[int]]]:
        r"""Performs matrix multiplication only on the necessary modes and
        takes care of keeping only the modes that are needed, in case of mismatch.

        See documentation for a visual explanation with blocks.  #TODO: add link to figure
        """
        if list(self.inmodes) == list(other.outmodes):  # NOTE: they match including the ordering
            prod = math.tensordot(
                self.tensor, other.tensor, ((1, 3), (0, 2)) if other.isMatrix else ((1, 3), (0, 1))
            )
            return math.transpose(prod, (0, 2, 1, 3) if other.isMatrix else (0, 1)), (
                self.outmodes,
                other.inmodes,
            )
        contracted = [i for i in self.inmodes if i in other.outmodes]
        uncontracted_self = [i for i in self.inmodes if i not in contracted]
        uncontracted_other = [o for o in other.outmodes if o not in contracted]
        if not (
            set(self.outmodes).isdisjoint(uncontracted_other)
            and set(other.inmodes).isdisjoint(uncontracted_self)
        ):
            raise ValueError("Invalid modes")
        bulk = None
        copied_rows = None
        copied_cols = None
        if len(contracted) > 0:
            subtensor1 = math.gather(
                self.tensor, [self.inmodes.index(m) for m in contracted], axis=1
            )
            subtensor2 = math.gather(
                other.tensor, [other.outmodes.index(m) for m in contracted], axis=0
            )
            if other.isMatrix:
                bulk = math.tensordot(subtensor1, subtensor2, ((1, 3), (0, 2)))
                bulk = math.transpose(bulk, (0, 2, 1, 3))
            else:
                bulk = math.tensordot(subtensor1, subtensor2, ((1, 3), (0, 1)))
        if self.like_1 and len(uncontracted_other) > 0:
            copied_rows = math.gather(
                other.tensor, [other.outmodes.index(m) for m in uncontracted_other], axis=0
            )
        if other.like_1 and len(uncontracted_self) > 0:
            copied_cols = math.gather(
                self.tensor, [self.inmodes.index(m) for m in uncontracted_self], axis=1
            )
        if copied_rows is not None and copied_cols is not None:
            if bulk is None:
                bulk = math.zeros(
                    (copied_cols.shape[0], copied_rows.shape[1], 2, 2), dtype=copied_cols.dtype
                )
            empty = math.zeros(
                (copied_rows.shape[0], copied_cols.shape[1], 2, 2), dtype=copied_cols.dtype
            )
            final = math.block([[copied_cols, bulk], [empty, copied_rows]], axes=[0, 1])
        elif copied_cols is None and copied_rows is not None:
            if bulk is None:
                final = copied_rows
            else:
                final = math.block([[bulk], [copied_rows]], axes=[0, 1])
        elif copied_rows is None and copied_cols is not None:
            if bulk is None:
                final = copied_cols
            else:
                final = math.block([[copied_cols, bulk]], axes=[0, 1])
        else:  # copied_rows and copied_cols are both None
            final = bulk  # NOTE: could be None

        outmodes = self.outmodes + uncontracted_other
        if other.like_0 and len(contracted) == 0:
            outmodes = uncontracted_other
        if self.like_0:
            outmodes = [m for m in outmodes if m in self.outmodes]

        inmodes = uncontracted_self + other.inmodes
        if self.like_0 and len(contracted) == 0:
            inmodes = uncontracted_self
        if other.like_0:
            inmodes = [m for m in inmodes if m in other.inmodes]

        if final is not None:
            final = math.gather(final, [outmodes.index(o) for o in sorted(outmodes)], axis=0)
            if other.isMatrix:
                final = math.gather(final, [inmodes.index(i) for i in sorted(inmodes)], axis=1)
        return final, (sorted(outmodes), sorted(inmodes))

    def _mode_aware_vecvec(self, other: XPVector) -> Scalar:
        if list(self.outmodes) == list(other.outmodes):
            return math.sum(self.tensor * other.tensor)
        common = list(
            set(self.outmodes) & set(other.outmodes)
        )  # only the common modes (the others are like 0)
        return math.sum(self.tensor[common] * other.tensor[common])

    def __add__(self, other: Union[XPMatrix, XPVector]) -> Union[XPMatrix, XPVector]:
        if not isinstance(other, (XPMatrix, XPVector)):
            raise TypeError(
                f"unsupported operand type(s) for +: '{self.__class__.__qualname__}' and '{other.__class__.__qualname__}'"
            )
        if self.isVector != other.isVector:
            raise ValueError("Cannot add a vector and a matrix")
        if self.isCoherence != other.isCoherence:
            raise ValueError("Cannot add a coherence block and a diagonal block")
        if self.tensor is None and other.tensor is None:  # both are None
            if self.like_1 and other.like_1:
                raise ValueError("Cannot add two like_1 null tensors yet")  # because 1+1 = 2
            if self.isMatrix and other.isMatrix:
                return XPMatrix(like_0=self.like_0 and other.like_0)
            else:
                return XPVector()
        if self.tensor is None:  # only self is None
            if self.like_0:
                return other
            elif (
                self.like_1
            ):  # other must be a matrix because self is like_1, so it must be a matrix and we can't add a vector to a matrix
                indices = [
                    [i, i] for i in range(other.num_modes)
                ]  # TODO: check if this is always correct
                updates = math.tile(
                    math.expand_dims(math.eye(2, dtype=other.dtype), 0), (other.num_modes, 1, 1)
                )
                other.tensor = math.update_add_tensor(other.tensor, indices, updates)
                return other
        if other.tensor is None:  # only other is None
            return other + self
        # now neither is None
        modes_match = list(self.outmodes) == list(other.outmodes) and list(self.inmodes) == list(
            other.inmodes
        )
        if modes_match:
            self.tensor = self.tensor + other.tensor
            return self
        if not modes_match and self.like_1 and other.like_1:
            raise ValueError("Cannot add two like_1 tensors on different modes yet")
        outmodes = sorted(set(self.outmodes).union(other.outmodes))
        inmodes = sorted(set(self.inmodes).union(other.inmodes))
        self_contains_other = set(self.outmodes).issuperset(other.outmodes) and set(
            self.inmodes
        ).issuperset(other.inmodes)
        other_contains_self = set(other.outmodes).issuperset(self.outmodes) and set(
            other.inmodes
        ).issuperset(self.inmodes)
        if self_contains_other:
            to_update = self.tensor
            to_add = [other]
        elif other_contains_self:
            to_update = other.tensor
            to_add = [self]
        else:  # need to add both to a new empty tensor
            to_update = math.zeros(
                (len(outmodes), len(inmodes), 2, 2) if self.isMatrix else (len(outmodes), 2),
                dtype=self.tensor.dtype,
            )
            to_add = [self, other]
        for t in to_add:
            outmodes_indices = [outmodes.index(o) for o in t.outmodes]
            inmodes_indices = [inmodes.index(i) for i in t.inmodes]
            if (
                t.isMatrix
            ):  # e.g. outmodes of to_update are [self]+[other_new] = (e.g.) [9,1,2]+[0,20]
                indices = [[o, i] for o in outmodes_indices for i in inmodes_indices]
            else:
                indices = [[o] for o in outmodes_indices]
            to_update = math.update_add_tensor(
                to_update,
                indices,
                math.reshape(t.modes_first(), (-1, 2, 2) if self.isMatrix else (-1, 2)),
            )
        if self.isMatrix and other.isMatrix:
            return XPMatrix(
                to_update,
                like_0=self.like_0 and other.like_0,
                like_1=self.like_1 or other.like_1,
                modes=(outmodes, inmodes),
            )
        else:
            return XPVector(to_update, outmodes)

    def __sub__(self, other: Union[XPMatrix, XPVector]) -> Optional[XPTensor]:
        return self + (-1) * other

    def __truediv__(self, other: Scalar) -> Optional[XPTensor]:
        return (1 / other) * self

    def __getitem__(self, modes: Union[int, slice, List[int], Tuple]) -> Union[XPMatrix, XPVector]:
        r"""Returns modes or subsets of modes from the XPTensor or coherences between modes using an
        intuitive notation.

        We handle mode indices and we get the corresponding tensor indices handled correctly.

        Examples:

        .. code::

            T[N] ~ self.tensor[N,:,:,:]
            T[M,N] = the coherence between the modes M and N
            T[:,N] ~ self.tensor[:,N,:,:]
            T[[1,2,3],:] ~ self.tensor[[1,2,3],:,:,:] # i.e. the block with outmodes [1,2,3] and all inmodes
            T[[1,2,3],[4,5]] ~ self.tensor[[1,2,3],[4,5],:,:]  # i.e. the block with outmodes [1,2,3] and inmodes [4,5]
        """
        if self.isVector:
            if isinstance(modes, int):
                _modes = [modes]
            elif isinstance(modes, list) and all(isinstance(m, int) for m in modes):
                _modes = modes
            elif modes == slice(None, None, None):
                _modes = self.outmodes
            else:
                raise ValueError(f"Usage: V[1], V[[1,2,3]] or V[:]")
            rows = [self.outmodes.index(m) for m in modes]
            return XPVector(math.gather(self.tensor, rows, axis=0), modes)
        else:
            _modes = [None, None]
            if isinstance(modes, int):
                _modes = ([modes], slice(None, None, None))
            elif isinstance(modes, list) and all(isinstance(m, int) for m in modes):
                _modes = (modes, slice(None, None, None))
            elif modes == slice(None, None, None):
                _modes = (slice(None, None, None), slice(None, None, None))
            elif isinstance(modes, tuple) and len(modes) == 2:
                for i, M in enumerate(modes):
                    if isinstance(M, int):
                        _modes[i] = [M]
                    elif isinstance(M, list):
                        _modes[i] = M
                    elif M == slice(None, None, None):
                        _modes[i] = self.modes[i]
                    else:
                        raise ValueError(
                            f"Invalid modes: {M} from {modes} (tensor has modes {self.modes})"
                        )
            else:
                raise ValueError(f"Invalid modes: {modes} (tensor has modes {self.modes})")
            rows = [self.outmodes.index(m) for m in _modes[0]]
            columns = [self.inmodes.index(m) for m in _modes[1]]
            subtensor = math.gather(self.tensor, rows, axis=0)
            subtensor = math.gather(subtensor, columns, axis=1)
            return XPMatrix(
                subtensor,
                like_1=_modes[0] == _modes[1] if self.like_1 else False,
                modes=tuple(_modes),
            )


class XPMatrix(XPTensor):
    r"""A convenience class for a matrix in the XPTensor format.

    # TODO: write docstring
    """

    def __init__(
        self,
        tensor: Tensor = None,
        like_0: bool = None,
        like_1: bool = None,
        modes: Tuple[List[int], List[int]] = ([], []),
    ):
        if like_0 is None and like_1 is None:
            raise ValueError("At least one of like_0 or like_1 must be set")
        if like_0 == like_1:
            raise ValueError(f"like_0 and like_1 can't both be {like_0}")
        if not (
            isinstance(modes, tuple) and len(modes) == 2 and all(type(m) == list for m in modes)
        ):
            raise ValueError("modes should be a tuple containing two lists (outmodes and inmodes)")
        if len(modes[0]) == 0 and len(modes[1]) == 0 and tensor is not None:
            if (
                tensor.shape[0] != tensor.shape[1] and like_0
            ):  # NOTE: we can't catch square coherences if no modes are specified
                raise ValueError("Must specify the modes for a coherence block")
            modes = tuple(
                list(range(s)) for s in tensor.shape[:2]
            )  # NOTE assuming that it isn't a coherence block
        like_0 = like_0 if like_0 is not None else not like_1
        super().__init__(tensor, like_0, isVector=False, modes=modes)

    @classmethod
    def from_xxpp(
        cls,
        tensor: Optional[Union[Matrix, Vector]],
        like_0: Optional[bool] = None,
        like_1: Optional[bool] = None,
        modes: Tuple[List[int], List[int]] = ([], []),
    ) -> XPMatrix:
        if tensor is not None:
            tensor = math.reshape(tensor, [_ for n in tensor.shape for _ in (2, n // 2)])
            tensor = math.transpose(tensor, (1, 3, 0, 2))
        return XPMatrix(tensor, like_0, like_1, modes)

    @classmethod
    def from_xpxp(
        cls,
        tensor: Optional[Union[Matrix, Vector]],
        like_0: bool = None,
        like_1: bool = None,
        modes: Tuple[List[int], List[int]] = ([], []),
    ) -> XPMatrix:
        if tensor is not None:
            tensor = math.reshape(tensor, [_ for n in tensor.shape for _ in (n // 2, 2)])
            tensor = math.transpose(tensor, (0, 2, 1, 3))
        return XPMatrix(tensor, like_0, like_1, modes)

    def __repr__(self) -> str:
        return f"XPMatrix(like_0={self.like_0}, modes={self.modes}, tensor_xpxp=\n{self.to_xpxp()})"


class XPVector(XPTensor):
    r"""A convenience class for a vector in the XPTensor format."""

    def __init__(self, tensor: Tensor = None, modes: List[int] = []):
        if not (isinstance(modes, list) or all(type(m) == int for m in modes)):
            raise ValueError(f"the modes of an XPVector should be a list of ints")
        if len(modes) == 0 and tensor is not None:
            modes = list(range(tensor.shape[0]))
        super().__init__(tensor, like_0=True, isVector=True, modes=(modes, []))

    @classmethod
    def from_xxpp(
        cls,
        tensor: Optional[Union[Matrix, Vector]],
        modes: List[int] = [],
    ) -> XPMatrix:
        if tensor is not None:
            tensor = math.reshape(tensor, (2, -1))
            tensor = math.transpose(tensor, (1, 0))
        return XPVector(tensor, modes)

    @classmethod
    def from_xpxp(
        cls,
        tensor: Optional[Union[Matrix, Vector]],
        modes: List[int] = [],
    ) -> XPMatrix:
        if tensor is not None:
            tensor = math.reshape(tensor, (-1, 2))
        return XPVector(tensor, modes)

    def __repr__(self) -> str:
        return f"XPVector(modes={self.outmodes}, tensor_xpxp=\n{self.to_xpxp()})"
