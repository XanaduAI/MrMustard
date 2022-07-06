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

"""This module contains the classes for representing Matrices and Vectors in phase space."""

from __future__ import annotations
from abc import ABC, abstractmethod
from mrmustard.types import Optional, Union, Matrix, Vector, List, Tensor, Tuple, Scalar
from mrmustard.math import Math

math = Math()


class XPTensor(ABC):
    r"""A representation of Matrices and Vectors in phase space.

    Tensors in phase space have shape ``(2n, 2n)`` (e.g. symplectic matrices, covariance matrices) or shape ``(2n,)`` (e.g. d vector)
    where n is the number of modes. 

    There are two main orderings of rows and columns:
        - xxpp: matrix is a `2\times 2` block matrix in which each block is `n\times n` (the four blocks correspond to `xx`, `xp`, `px`, `pp`).
        - xpxp: matrix is an `n\times n` block matrix in which each block is `2\times 2` and corresponding to a sigle mode or a single coherence between modes.
    This creates some difficulties when we need to work in a mode-wise fashion, especially with off-diagonal coherences, or when we rearrange modes.
    We solve this problem by reshaping the matrices to shape `(b,n,m,2,2)` and vectors to shape `(b,n,2)` where `b` is a batch dimension. This creates a unified ordering.
    The idea is for a user to forget about the (2,2) extra dimensions and just work with the `n` modes when multiplying, adding, getting submatrices, etc.

    We call `n` the outmodes and `m` the inmodes. Covariance matrices have outmodes equal to the inmodes.
    For symplectic matrices they can differ because we include transformations that map a set of modes to a set of different modes.
    Off-diagonal matrices have all the outmodes different from the inmodes. Vectors have only outmodes.

    XPTensor objects are mode-wise sparse, in the sense that they support operations between modes where one or more of the tensors are undefined.
    There are two types of behaviour:
        * like_0: in modes where the tensor is undefined, it's like having a zero (a zero matrix)
        * like_1: in modes where the tensor is undefined, it's like having a one (an identity matrix)

    For example, in the expression :math:`X @ means + d` where `X` is a symplectic matrix and `d` is a displacement vector,
    if `X` is undefined it's like having the identity and the matrix product simply returns `means`, while in the expression
    :math:`means + d` if `d` is undefined it simply returns `means`. In these cases no operation is actually computed.
    Thanks to sparsity we can represent graph states and transformations on graph states using XPTensor objects.

    Thanks to the batch dimension we can represent non-gaussian objects as linear superpositions of gaussian objects.

    Args:
        tensor: The tensor in (b,n,m,2,2) or (b,n,2) order.
        outmodes: a list of output modes
        inmodes: a list of input modes
    """

    @abstractmethod  # to prevent XPTensor to be instantiated directly
    def __init__(
        self,
        tensor: Optional[Tensor],
        outmodes: List[int],
        inmodes: List[int],
    ):
        # NOTE: tensor is supposed to be rank 5 for matrices: batch + outmodes + inmodes + 2 + 2 or rank 3 for vectors: batch + outmodes + 2
        assert tensor is None or tensor.ndim == 5 or tensor.ndim == 3
        self.shape = tuple() if tensor is None else tensor.shape[1:1+(len(tensor.shape)-1)//2]  # NOTE: (N,M) or (N,) or (,)
        self.ndim = len(self.shape) # 0 if tensor is None
        self.batch_size = tensor.shape[0] if tensor is not None else 0
        self.tensor = tensor
        if not (set(outmodes) == set(inmodes) or set(outmodes).isdisjoint(inmodes)):
            raise ValueError("inmodes and outmodes should contain the same modes or be disjoint")
        self.outmodes = outmodes
        self.inmodes = inmodes

    @property
    def dtype(self):
        return None if self.tensor is None else self.tensor.dtype

    @property
    def like_1(self) -> bool:
        return not self.like_0

    @property
    def num_modes(self) -> int:
        if len(self.outmodes) != len(self.inmodes):
            raise ValueError("The number of outmodes and inmodes doesn't match")
        return len(self.outmodes)

    def to_xpxp(self) -> Optional[Union[Matrix, Vector]]:
        if self.tensor is None:
            return None
        tensor = math.transpose(
            self.tensor, (0, 1, 3, 2, 4) if self.ndim == 2 else (0, 1, 2)
        )  # from BNN22 to B(N2)(N2) or from BN2 to B(N2)
        return math.reshape(tensor, [self.batch_size] + [2 * s for s in self.shape])

    def to_xxpp(self) -> Optional[Union[Matrix, Vector]]:
        if self.tensor is None:
            return None
        tensor = math.transpose(
            self.tensor, (0, 3, 1, 4, 2) if self.ndim == 2 else (0, 2, 1)
        )  # from BNN22 to B(2N)(2N) or from BN2 to B(2N)
        return math.reshape(tensor, [self.batch_size] + [2 * s for s in self.shape]) 

    def __array__(self):
        return self.to_xxpp()

    def modes_first(self) -> Optional[Tensor]:
        return self.tensor

    def modes_last(self) -> Optional[Tensor]:
        if self.tensor is None:
            return None
        return math.transpose(self.tensor, (0, 3, 4, 1, 2) if self.ndim == 2 else (0, 2, 1))  # B22NM or B2N

    ####################################################################################################################
    # Operators
    ####################################################################################################################

    def __rmul__(self, other: Scalar) -> XPTensor:
        "Implements the operation ``self * other``"
        if self.tensor is None:
            if self.like_1:
                raise NotImplementedError("Cannot multiply a scalar and a like_1 null tensor yet")
            return self
        self.tensor = other * self.tensor
        return self

    def __mul__(self, other: Scalar) -> Optional[XPTensor]:
        return other * self

    # def __add__(self, other: Union[XPMatrix, XPVector]) -> Union[XPMatrix, XPVector]:
    #     if not isinstance(other, (XPMatrix, XPVector)):
    #         raise TypeError(
    #             f"unsupported operand type(s) for +: '{self.__class__.__qualname__}' and '{other.__class__.__qualname__}'"
    #         )
    #     if self.isVector != other.isVector:
    #         raise ValueError("Cannot add a vector and a matrix")
    #     if self.isCoherence != other.isCoherence:
    #         raise ValueError("Cannot add a coherence block and a diagonal block")
    #     if self.tensor is None and other.tensor is None:  # both are None
    #         if self.like_1 and other.like_1:
    #             raise ValueError("Cannot add two like_1 null tensors yet")  # because 1+1 = 2
    #         if self.isMatrix and other.isMatrix:
    #             return XPMatrix(like_0=self.like_0 and other.like_0)
    #         return XPVector()
    #     if self.tensor is None:  # only self is None
    #         if self.like_0:
    #             return other

    #         # other must be a matrix because self is like_1, so it must be a matrix and we can't add a vector to a matrix
    #         if self.like_1:
    #             indices = [
    #                 [i, i] for i in range(other.num_modes)
    #             ]  # TODO: check if this is always correct
    #             updates = math.tile(
    #                 math.expand_dims(math.eye(2, dtype=other.dtype), 0), (other.num_modes, 1, 1)
    #             )
    #             other.tensor = math.update_add_tensor(other.tensor, indices, updates)
    #             return other
    #     if other.tensor is None:  # only other is None
    #         return other + self
    #     # now neither is None
    #     modes_match = list(self.outmodes) == list(other.outmodes) and list(self.inmodes) == list(
    #         other.inmodes
    #     )
    #     if modes_match:
    #         self.tensor = self.tensor + other.tensor
    #         return self
    #     if not modes_match and self.like_1 and other.like_1:
    #         raise ValueError("Cannot add two like_1 tensors on different modes yet")
    #     outmodes = sorted(set(self.outmodes).union(other.outmodes))
    #     inmodes = sorted(set(self.inmodes).union(other.inmodes))
    #     self_contains_other = set(self.outmodes).issuperset(other.outmodes) and set(
    #         self.inmodes
    #     ).issuperset(other.inmodes)
    #     other_contains_self = set(other.outmodes).issuperset(self.outmodes) and set(
    #         other.inmodes
    #     ).issuperset(self.inmodes)
    #     if self_contains_other:
    #         to_update = self.tensor
    #         to_add = [other]
    #     elif other_contains_self:
    #         to_update = other.tensor
    #         to_add = [self]
    #     else:  # need to add both to a new empty tensor
    #         to_update = math.zeros(
    #             (len(outmodes), len(inmodes), 2, 2) if self.isMatrix else (len(outmodes), 2),
    #             dtype=self.tensor.dtype,
    #         )
    #         to_add = [self, other]
    #     for t in to_add:
    #         outmodes_indices = [outmodes.index(o) for o in t.outmodes]
    #         inmodes_indices = [inmodes.index(i) for i in t.inmodes]
    #         if (
    #             t.isMatrix
    #         ):  # e.g. outmodes of to_update are [self]+[other_new] = (e.g.) [9,1,2]+[0,20]
    #             indices = [[o, i] for o in outmodes_indices for i in inmodes_indices]
    #         else:
    #             indices = [[o] for o in outmodes_indices]
    #         to_update = math.update_add_tensor(
    #             to_update,
    #             indices,
    #             math.reshape(t.modes_first(), (-1, 2, 2) if self.isMatrix else (-1, 2)),
    #         )
    #     if self.isMatrix and other.isMatrix:
    #         return XPMatrix(
    #             to_update,
    #             like_0=self.like_0 and other.like_0,
    #             like_1=self.like_1 or other.like_1,
    #             modes=(outmodes, inmodes),
    #         )

    #     return XPVector(to_update, outmodes)

    def __sub__(self, other: Optional[XPTensor]) -> Optional[XPTensor]:
        return self + (-1) * other

    def __truediv__(self, other: Scalar) -> Optional[XPTensor]:
        return (1 / other) * self


class XPMatrix(XPTensor):  # TODO: should this be abstract too?
    r"""A convenience class for a matrix in the XPTensor format.

    # TODO: write docstring
    """

    def __init__(
        self,
        tensor: Optional[Union[Tensor, XPMatrix]],
        like_0: bool = None,
        outmodes: List[int] = [],
        inmodes: List[int] = [],
    ):
        if isinstance(tensor, XPMatrix):
            super().__init__(tensor.tensor, tensor.outmodes, tensor.inmodes)
            self.like_0 = tensor.like_0
        else:   
            super().__init__(tensor, outmodes, inmodes)
            if like_0 is None: raise ValueError("like_0 must be specified")
            self.like_0 = like_0

    @classmethod
    def from_xxpp(
        cls,
        tensor: Optional[Matrix],
        like_0: Optional[bool],
        outmodes: List[int] = [],
        inmodes: List[int] = [],
    ) -> XPMatrix:
        if tensor is not None:
            # batch if necessary
            if len(tensor.shape) == 2:
                tensor = math.expand_dims(tensor, 0)
            # split 2n -> 2,n for both input dimensions
            tensor = math.reshape(tensor, [tensor.shape[0]] + [_ for n in tensor.shape[1:] for _ in (2, n // 2)])
            # transpose so that the index order is b,n,m,2,2
            tensor = math.transpose(tensor, (0, 2, 4, 1, 3))
        return XPMatrix(tensor, like_0, outmodes, inmodes)  # NOTE we use cls so that subclasses won't need to reimplement this method

    @classmethod
    def from_xpxp(
        cls,
        tensor: Optional[Matrix],
        like_0: Optional[bool],
        outmodes: List[int] = [],
        inmodes: List[int] = [],
    ) -> XPMatrix:
        if tensor is not None:
            # batch if necessary
            if len(tensor.shape) == 2:
                tensor = math.expand_dims(tensor, 0)
            # split 2n -> 2,n for both input dimensions
            tensor = math.reshape(tensor, [tensor.shape[0]] + [_ for n in tensor.shape[1:] for _ in (n // 2, 2)])
            # transpose so that the index order is b,n,m,2,2
            tensor = math.transpose(tensor, (0, 1, 3, 2, 4))
        return XPMatrix(tensor, like_0, outmodes, inmodes) #note cls is so that subclasses don't need to reimplement this method

    def clone(self, times: int, outmodes: Optional[List[int]] = None, inmodes: Optional[List[int]] = None) -> XPtensor:
        r"""Create a new XPTensor by cloning the system a given number of times
        (the modes are reset by default unless specified).
        """
        if self.tensor is None or times == 1:
            return self
        tensor = math.expand_dims(self.modes_last(), axis=-1)  # shape = [B,2,2,N,N,1]
        tensor = math.tile(tensor, (1, 1, 1, 1, 1, times))  # shape = [B,2,2,N,N,T]
        tensor = math.diag(tensor)  # shape = [B,2,2,N,N,T,T]
        tensor = math.transpose(tensor, (0, 1, 2, 3, 5, 4, 6))  # shape = [B,2,2,N,T,N,T]
        tensor = math.reshape(tensor, (self.batch_size, 2, 2, tensor.shape[-4] * times, tensor.shape[-2] * times))  # shape = [B,2,2,NT,NT]
        tensor = math.transpose(tensor, (0, 3, 4, 1, 2))  # shape = [B,NT,NT,2,2]
        if outmodes is None:
            outmodes = list(range(times*len(self.outmodes))) # NOTE: is this what we want? e.g. outmodes = [3,10] -> [3,10,13,20]. Should we reset the modes instead (i.e. [3,10]->[0,1,2,3])?
        if inmodes is None:
            inmodes = list(range(times*len(self.inmodes)))
        return XPMatrix(tensor, self.like_0, outmodes, inmodes)

    # def clone_like(self, other: XPMatrix) -> XPMatrix:
    #     r"""Create a new XPMatrix with the same shape and modes as other.

    #     The new tensor has the same content as self, cloned as many times as necessary to match the
    #     shape and modes of other. The other properties are kept as is.

    #     Args:
    #         other: the XPMatrix to clone like.

    #     Returns:
    #         Tensor: A new XPMatrix with the same shape and modes as other.
    #     """
    #     if other.shape == self.shape:
    #         return self
    #     if bool(other.num_modes % self.num_modes):
    #         raise ValueError(
    #             f"No integer multiple of {self.num_modes} modes fits into {other.num_modes} modes"
    #         )
    #     times = other.num_modes // self.num_modes
    #     return XPVector(self.clone(times, modes=other.modes).tensor, other.outmodes)

    @property
    def T(self) -> XPMatrix:
        if self.tensor is None:
            return self
        return XPMatrix(
            math.transpose(self.tensor, (0, 2, 1, 4, 3)),
            self.like_0,
            self.outmodes,
            self.inmodes,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(tensor=..., like_0={self.like_0}, outmodes={self.outmodes}, inmodes={self.inmodes})"

    def __matmul__(self, other: Union[XPMatrix, XPVector]) -> Union[XPMatrix, XPVector]:
        if not isinstance(other, (XPMatrix, XPVector)):
            raise TypeError(f"Unsupported operand type(s) for @: 'XPMatrix' and '{other.__class__.__qualname__}'")
        if other.tensor is None:
            return self if other.like_1 else other
        if self.tensor is None: # NOTE: other is not None now
            return other if self.like_1 else self
        if isinstance(other, XPMatrix):
            tensor = math.sparse_matmul(self.tensor, other.tensor, self.outmodes, other.outmodes, self.like_0, other.like_0)
            modes, *_ = math.sparse_matmul_data(self.tensor, other.tensor,self.outmodes, other.outmodes, self.like_0, other.like_0)
            return XPMatrix(tensor, self.like_0 or other.like_0, outmodes=modes, inmodes=modes)  # TODO: generalize to handle different in/out modes
        if isinstance(other, XPVector):
            tensor = math.sparse_matvec(self.tensor, other.tensor, self.outmodes, other.outmodes, self.like_0)
            return XPVector(tensor, modes=other.outmodes)

    def __add__(self, other: XPMatrix) -> XPMatrix:
        if not isinstance(other, XPMatrix):
            raise TypeError(f"Unsupported operand type(s) for +: 'XPMatrix' and '{other.__class__.__qualname__}'. Only an XPMatrix can be added to an XPMatrix.")
        if other.tensor is None and other.like_0:
            return self
        if self.tensor is None and self.like_0:
            return other
        if self.tensor is not None and other.tensor is not None:
            match_exactly = self.outmodes == other.outmodes and self.inmodes == other.inmodes
            if match_exactly:
                return XPMatrix(self.tensor + other.tensor, self.like_0 or other.like_0, self.outmodes, self.inmodes)
            return math.sparse_mat_add(self.tensor, other.tensor, self.outmodes, other.outmodes, self.like_0, other.like_0)
        raise ValueError(f"Can't add a like_1 null XPMatrix and an XPMatrix that is not null and like_0.")

    def __getitem__(self, modes: Union[int, slice, List[int], Tuple]) -> XPMatrix:
        r"""Returns modes or subsets of modes using an intuitive notation.

        We handle mode indices and we get the corresponding tensor indices handled correctly.

        Examples:

        .. code::

            T[N] ~ self.tensor[:,N,:,:,:]
            T[M,N] = the coherence between the modes M and N
            T[:,N] ~ self.tensor[:,:,N,:,:]
            T[[1,2,3],:] ~ self.tensor[:,[1,2,3],:,:,:] # i.e. the block with outmodes [1,2,3] and all inmodes
            T[[1,2,3],[4,5]] ~ self.tensor[:,[1,2,3],[4,5],:,:]  # i.e. the block with outmodes [1,2,3] and inmodes [4,5]
        """
        outmodes, inmodes = [None, None]
        if isinstance(modes, int):
            outmodes = modes
            inmodes = slice(None, None, None)
        elif isinstance(modes, list) and all(isinstance(m, int) for m in modes):
            outmodes = modes
            inmodes = slice(None, None, None)
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
        rows = [self.outmodes.index(m) for m in outmodes]
        columns = [self.inmodes.index(m) for m in inmodes]
        subtensor = math.gather(self.tensor, rows, axis=-4)
        subtensor = math.gather(subtensor, columns, axis=-3)
        return XPMatrix(
            subtensor,
            like_0=self.like_0 or set(outmodes).isdisjoint(inmodes),
            outmodes=outmodes,
            inmodes=inmodes,
        )

class XPVector(XPTensor):
    r"""A convenience class for a vector in the XPTensor format."""

    def __init__(self, tensor: Tensor = None, modes: List[int] = []):
        if isinstance(tensor, XPVector):
            super().__init__(tensor.tensor, tensor.outmodes, tensor.inmodes)
            self.like_0 = True
        else:
            if modes == [] and tensor is not None:
                modes = list(range(tensor.shape[-2]))
            super().__init__(tensor, outmodes = modes, inmodes = [])

    @classmethod
    def from_xxpp(
        cls,
        tensor: Optional[Vector],
        modes: List[int] = [],
    ) -> XPVector:
        if tensor is not None:
            if len(tensor.shape) == 1:
                tensor = math.expand_dims(tensor, 0)
            tensor = math.reshape(tensor, (tensor.shape[0], 2, -1))
            tensor = math.transpose(tensor, (0, 2, 1))
        return cls(XPVector(tensor, modes))

    @classmethod
    def from_xpxp(
        cls,
        tensor: Optional[Vector],
        modes: List[int] = [],
    ) -> XPVector:
        if tensor is not None:
            if len(tensor.shape) == 1:
                tensor = math.expand_dims(tensor, 0)
            tensor = math.reshape(tensor, (tensor.shape[0], -1, 2))
        return cls(XPVector(tensor, modes))

    def clone(self, times: int, modes=None) -> XPVector:
        r"""Create a new XPTensor by cloning the system a given number of times
        (the modes are reset by default unless specified).
        """
        if self.tensor is None or times == 1:
            return self
        tensor = math.tile(
            self.expand_dims(self.modes_last(), axis=2), (1,1, 1, times)
        )  # shape = [B,2,N,T]
        tensor = math.reshape(tensor, (self.batch_size, 2, -1))  # shape = [B,2,NT]
        tensor = math.transpose(tensor, (0,2,1))  # shape = [B,NT,2]
        return XPVector(tensor, [] if modes is None else modes)

    # def clone_like(self, other: XPVector) -> XPVector:
    #     r"""Create a new XPTensor with the same shape and modes as other.

    #     The new tensor has the same content as self, cloned as many times as necessary to match the
    #     shape and modes of other. The other properties are kept as is.

    #     Args:
    #         other: the XPVector to clone like.

    #     Returns:
    #         Tensor: A new XPVector with the same shape and modes as other.
    #     """
    #     if other.shape == self.shape:
    #         return self
    #     if bool(other.num_modes % self.num_modes):
    #         raise ValueError(
    #             f"No integer multiple of {self.num_modes} modes fits into {other.num_modes} modes"
    #         )
    #     times = other.num_modes // self.num_modes
    #     return XPVector(self.clone(times, modes=other.modes).tensor, other.outmodes)

    def __repr__(self) -> str:
        return f"XPVector(modes={self.outmodes}, tensor_xpxp=\n{self.to_xpxp()})"

    def __matmul__(self, other: Union[XPMatrix, XPVector]) -> Union[XPMatrix, Scalar]:
        if not isinstance(other, (XPMatrix, XPVector)):
            raise TypeError(f"Unsupported operand type(s) for @: 'XPVector' and '{other.__class__.__qualname__}'. Only XPMatrix and XPVector are supported.")
        if isinstance(other, XPMatrix):
            return other.T @ self
        if self.tensor is not None and other.tensor is not None:
            if list(self.outmodes) == list(other.outmodes):
                return math.sum(self.tensor * other.tensor)
            common = list(set(self.outmodes) & set(other.outmodes))
            return math.sum(self[common].tensor * other[common].tensor) # TODO: this is not batched in the right way
        return 0.0

    def __add__(self, other: XPVector) -> XPVector:
        if not isinstance(other, XPVector):
            raise TypeError(f"Unsupported operand type(s) for +: 'XPVector' and '{other.__class__.__qualname__}'. Only an XPVector can be added to an XPVector.")
        if other.tensor is None:
            return self
        if self.tensor is None:
            return other
        if self.tensor is not None and other.tensor is not None:
            return math.sparse_vec_add(self.to_xxpp(), other.to_xxpp(), self.outmodes, other.outmodes)

    def __getitem__(self, modes: Union[int, slice, List[int], Tuple]) -> XPVector:
        r"""Returns modes or subsets of modes from the XPMatrix or coherences between modes using an
        intuitive notation.

        We handle mode indices and we get the corresponding tensor indices handled correctly.

        Examples:

        .. code::

            V[N] ~ self.tensor[:,N,:]
            V[[1,2,3]] ~ self.tensor[:,[1,2,3],:] 
            V[1:3] ~ self.tensor[:,1:3,:]
        """
        if isinstance(modes, int):
            _modes = [modes]
        elif isinstance(modes, list) and all(isinstance(m, int) for m in modes):
            _modes = modes
        elif modes == slice(None, None, None):
            _modes = self.outmodes
        else:
            raise ValueError("Usage: V[1], V[[1,2,3]] or V[:]")
        rows = [self.outmodes.index(m) for m in _modes]
        return XPVector(math.gather(self.tensor, rows, axis=-2), _modes)


class Symplectic(XPMatrix):
    r"""A convenience class for a symplectic matrix in the XPTensor format.

    # TODO: write docstring
    """

    def __init__(
        self,
        tensor: Union[XPMatrix, Tensor] = None,
        modes: List[int] = [],
    ):
        super().__init__(tensor, False, modes, modes)
        self.like_0 = False

    @property
    def inverse(self) -> Symplectic:  # TODO: batch
        [[A,B],[C,D]] = self.modes_last()
        return Symplectic(math.astensor([[math.transpose(D),-math.transpose(B)],
                                        [-math.transpose(C),math.transpose(A)]]), self.modes)

    def __repr__(self) -> str:
        return f"Symplectic matrix in Sp({2*len(self.modes)},R) on modes {self.modes}."