from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty
from mrmustard import Backend
backend = Backend()
from mrmustard._typing import *


class XPTensor:
    r"""A representation of tensors in phase space.
    Matrices are stored as a (2*nmodes_out, 2*nmodes_in)-tensor in xxpp ordering, but internally we utilize the
    (2, nmodes_out, 2, nmodes_in) representation to simplify several operations. For a Matrix, modes_in and modes_out can differ
    (the matrix maps between two vector spaces).
    Vectors are stored as a (2*nmodes)-vector, but we use the (2, nmodes) representation to perform simplified operations. For
    a Vector, modes_in and modes_out must be equal (there is only one vector space the vector belongs to).
    
    TODO: implement a sparse version for representing graph states (collection of 2x2 matrices + 2x2 coherences to other states)

    Arguments:
        tensor: The tensor to be represented.
        modes: input-output modes if matrix, just modes if vector
        additive: Whether the null tensor behaves like 0 for addition
        multiplicative: Whether the null tensor behaves like 1 for multiplication
    """

    def __init__(self,
                tensor: Optional[Union[Matrix, Vector]] = None,
                modes: Union[Tuple[List[int], List[int]], List[int]]] = None,
                additive: Optional[bool] = None,
                multiplicative: Optional[bool] = None):
        self.additive = bool(additive) or not bool(multiplicative)  # I love python
        self.shape = None if tensor is None else [t // 2 for t in tensor.shape]
        self.ndim = None if tensor is None else tensor.ndim
        self.isVector = None if tensor is None else self.ndim == 1
        self.isMatrix = None if tensor is None else self.ndim == 2
        if modes is None and tensor is not None:
            if self.isVector:
                modes = [m for m in range(tensor.shape[-1] // 2)]
            if self.isMatrix:
                modes = [m for m in range(tensor.shape[-2] // 2)], [m for m in range(tensor.shape[-1] // 2)]
        self.modes = modes
        self._tensor = tensor

    @property
    def multiplicative(self) -> bool:
        return not bool(self.additive)

    @property
    def tensor(self):
        if self._tensor is None:
            return None
        return backend.reshape(self._tensor, [k for n in self.shape for k in (2, n)])  # [2,n] or [2,n,2,n] or [2,n,2,m]

    def valid(self, tensor: Optional[Tensor], modes: List[int]) -> bool:
        if tensor is None:
            return True
        if len(tensor.shape) > 2:
            raise ValueError(f"Tensor must be 1D or 2D, got {tensor.ndim}D")
        if len(modes) > tensor.shape[-1] // 2:
            raise ValueError(f"Too many modes ({len(modes)}) for tensor with {tensor.shape[-1]//2} modes")
        if len(modes) < tensor.shape[-1] // 2:
            raise ValueError(f"Too few modes ({len(modes)}) for tensor with {tensor.shape[-1]//2} modes")
        return True

    @classmethod
    def from_xxpp(cls, matrix: Matrix, modes: Optional[Tuple[List[int],List[int]]], additive: bool = None, multiplicative: bool = None) -> XPMatrix:
        return XPMatrix(matrix, modes, additive, multiplicative)

    @classmethod
    def from_xpxp(cls, matrix: Matrix, modes: Optional[Tuple[List[int], List[int]]], additive: bool = None, multiplicative: bool = None) -> XPTensor:
        if matrix is not None:
            matrix = backend.reshape(matrix, [k for n in matrix.shape for k in (n, 2)])
            matrix = backend.transpose(matrix, (1, 0, 3, 2)[: 2 * self.ndim])
            matrix = backend.reshape(matrix, [2 * s for s in matrix.shape])
        return cls(matrix, modes, additive, multiplicative)

    def to_xpxp(self) -> Optional[Matrix]:
        if self._tensor is None:
            return None
        tensor = backend.transpose(self.tensor, (1, 0, 3, 2)[: 2 * self.ndim])
        return backend.reshape(tensor, [2 * s for s in self.shape])

    def to_xxpp(self) -> Optional[Tensor]:
        return self._tensor

    def clone(self, modes: Sequence[int], times: int):
        if self._tensor is None:
            pass
        to_clone = self[list(modes)]  # [2,m] or [2,m,2,m]
        if self.isMatrix:
            coherences = self[:, list(modes)] # [2,n,2,m]
        if self.isVector:
            return XPTensor(backend.concat([self.tensor]+[to_clone]*times, axis=-1))
        if self.isMatrix:
            rows = backend.concat([coherences]*times, axis=-1)

    def __array__(self):
        return self._tensor

    def __rmul__(self, other: Scalar) -> XPTensor:
        if self._tensor is None:
            if self.multiplicative:
                raise NotImplementedError("Cannot multiply a scalar by a multiplicative null tensor yet")
            else:
                return XPTensor(None, self.modes, self.additive, self.multiplicative)
        return XPTensor(other * self._tensor, self.modes, self.additive, self.multiplicative)

    def __mul__(self, other: Scalar) -> Optional[XPTensor]:
        return other * self if self._tensor is not None else None

    def __matmul__(self, other: XPTensor) -> Optional[XPTensor]:
        if self._tensor is None and other._tensor is None:
            return other # XPTensor(additive=self.additive or other.additive)
        if self._tensor is None:  # only self is None
            if self.additive:
                return self
            return other
        if other._tensor is None:
            if other.additive:
                return other
            return self
        if self.isMatrix and other.isVector:
            xxpp, modes = self.matvec_at_modes(self.to_xxpp(), other.to_xxpp(), modes_a=self.modes, modes_b=other.modes, additive_a=self.additive, additive_b=other.additive)
        elif self.isVector and other.isMatrix:
            xxpp, modes = self.matvec_at_modes(other.T.to_xxpp(), self.to_xxpp(), modes_a=other.modes, modes_b=self.modes, additive_a=other.additive, additive_b=self.additive)
        elif self.isMatrix and other.isMatrix:
            xxpp, modes = self.matmat_at_modes(self.to_xxpp(), other.to_xxpp(), modes_a=self.modes, modes_b=other.modes, additive_a=self.additive, additive_b=other.additive)
        else:
            xxpp, modes = self.vecvec_at_modes(self.to_xxpp(), other.to_xxpp(), modes_a=self.modes, modes_b=other.modes, additive_a=self.additive, additive_b=other.additive)
            if not hasattr(xxpp, 'shape') or len(xxpp.shape) == 1:
                return xxpp
        return XPTensor.from_xxpp(xxpp, modes, additive=self.additive or other.additive)

    def _join_modes(modes_a, modes_b, additive_a: bool, additive_b: bool):
        contracted = set(modes_a[1]) & set(modes_b[0])
        uncontracted_b = set(modes_b[0]) - contracted
        uncontracted_a = set(modes_a[1]) - contracted
        modes_out = set(modes_a[0]) if additive_a else set(modes_a[0]) | uncontracted_b
        modes_in = set(modes_b[1]) if additive_b else set(modes_b[1]) | uncontracted_a
        return list(modes_out), contracted, list(modes_in)

    def _gather_inmodes(self, modes: List[int]):
        if self.isMatrix:
            return backend.reshape(backend.gather(self.tensor, modes, axis=3), [2*self.shape[0]]+[2*len(modes)])
        
    def _gather_outmodes(self, modes: List[int]):
        if self.isMatrix:
            return backend.reshape(backend.gather(self.tensor, modes, axis=1), 2*len(modes)]+[2*self.shape[1]])

    def matmat_at_modes(self,
                        xxpp_a: Matrix,
                        xxpp_b: Matrix,
                        modes_a: Tuple[List[int],List[int]],
                        modes_b: Tuple[List[int],List[int]],
                        additive_a: bool,
                        additive_b: bool) -> Matrix, Sequence:
        if modes_a[1] == modes_b[0]:
            return backend.matmul(xxpp_a, xxpp_b)
        modes_out, contracted, modes_in = self._join_modes(modes_a, modes_b, additive_a, additive_b)
        out = backend.matmul(self._gather_inmodes(xxpp_a, some_modes_a) self._gather_outmodes(xxpp_b, some_modes_b))  # modes_out is the same as modes_in
        # TODO: propagate appropriate unconctracted modes for each matrix
        return out, modes

    def matvec_at_modes(self, xxpp_a: Matrix, xxpp_b: Vector, modes_a: List[int], modes_b: List[int]) -> Vector:
        if modes_a == modes_b:
            return backend.matvec(xxpp_a, xxpp_b)
        modes = set(modes_a) | set(modes_b)
        out = backend.zeros(2 * len(modes), dtype=xxpp_b.dtype)
        out = backend.add_at_modes(xxpp_b, out, modes_b)
        out = backend.matvec_at_modes(xxpp_a, out, modes_a)
        return out, modes

    def vecvec_at_modes(self, xxpp_a: Vector, xxpp_b: Vector, modes_a: List[int], modes_b: List[int]) -> Scalar:
        if modes_a == modes_b:
            return backend.sum(xxpp_a * xxpp_b)
        modes = set(modes_a) & set(modes_b)  # only the common modes
        out = backend.vecvec_at_modes(xxpp_a, out, modes)
        return out, modes

    def __add__(self, other: XPTensor) -> Optional[XPTensor]:
        if self._tensor is None and other._tensor is None:
            return XPTensor(None, [], self.additive and other.additive)  # NOTE: this says 1+1 = 1
        if self._tensor is None:  # only self is None
            if self.additive:
                return other
            return ValueError("0+1 not implemented 🥸")
        if other._tensor is None:  # only other is None
            if other.additive:
                return self
            return ValueError("1+0 not implemented 🥸")
        xxpp = self.add_at_modes(self.to_xxpp(), other.to_xxpp(), modes_a=self.modes, modes_b=other.modes)
        return XPTensor.from_xxpp(xxpp, sorted(list(set(self.modes) | set(other.modes))), self.additive and other.additive)

    def add_at_modes(self, xxpp_a: Matrix, xxpp_b: Matrix, modes_a: List[int], modes_b: List[int]) -> Matrix:
        if modes_a == modes_b:
            return xxpp_a + xxpp_b
        modes = set(modes_a) | set(modes_b)
        out = backend.zeros((2 * len(modes), 2 * len(modes)), dtype=xxpp_a.dtype)
        out = backend.add_at_modes(out, xxpp_a, modes_a)
        out = backend.add_at_modes(out, xxpp_b, modes_b)
        return out

    def __sub__(self, other: XPTensor) -> Optional[XPTensor]:
        return self + (-1) * other

    def __repr__(self) -> str:
        return f"XPTensor(modes={self.modes}, additive={self.additive}, _tensor={self._tensor})"

    # TODO: write a tensor wrapper to use __setitem__ with TF (it's already possible with pytorch)
    # (must be differentiable!)
    # def __setitem__(self, key: Union[int, slice, List[int]], value: XPTensor) -> None:
    #     if isinstance(key, int):
    #         self._tensor[:, key, :, key] = value._tensor
    #     elif isinstance(key, slice):
    #         self._tensor[:, key, :, key] = value._tensor
    #     elif isinstance(key, List):
    #         self._tensor[:, key, :, key] = value._tensor
    #     else:
    #         raise TypeError("Invalid index type")

    def __getitem_list(self, item):
        if isinstance(item, int):
            lst = [item]
        elif isinstance(item, slice):
            lst = list(range(item.start or 0, item.stop or self.nmodes, item.step))
        elif isinstance(item, List):
            lst = item
        else:
            lst = None  # is this right?
        return lst

    @property
    def T(self) -> XPTensor:
        if self.isVector:
            raise ValueError("Cannot transpose a vector")
        if self._tensor is None:
            return XPTensor(None, [], self.additive)
        return XPTensor(backend.transpose(self._tensor), self.modes)












class XPMatrix(XPTensor):
    r"""Concrete class for an XPTensor that is a matrix.
    """
    def __init__(self, matrix: Optional[Matrix]=None, modes:Optional[Tuple[List[int], List[int]]] = None, additive:bool=None, multiplicative: bool=None):
        self.additive = bool(additive) or not bool(multiplicative)  # I love python
        self.shape = None if matrix is None else [t // 2 for t in matrix.shape]
        self.ndim = 0 if matrix is None else matrix.ndim
        self.isVector = False
        self.isMatrix = True   # TODO: remove None case?
        if modes is None and matrix is not None:
            modes = [m for m in range(matrix.shape[-2] // 2)], [m for m in range(matrix.shape[-1] // 2)]
        self.modes = modes
        self._tensor = matrix

    @classmethod
    def from_xxpp(cls, matrix: Matrix, modes: Optional[Tuple[List[int],List[int]]]=None, additive: bool = None, multiplicative: bool = None) -> XPMatrix:
        return XPMatrix(matrix, modes, additive, multiplicative)

    @classmethod
    def from_xpxp(cls, matrix: Matrix, modes: Tuple[List[int], List[int]]=([],[]), additive: bool = None, multiplicative: bool = None) -> XPMatrix:
        if matrix is not None:
            matrix = backend.reshape(matrix, [k for n in matrix.shape for k in (n, 2)])
            matrix = backend.transpose(matrix, (1, 0, 3, 2))
            matrix = backend.reshape(matrix, [2 * s for s in matrix.shape])
        return cls(matrix, modes, additive, multiplicative)

    def clone(self, modes: Sequence[int], times: int):
        if self._tensor is None:
            pass
        to_clone = self[list(modes)]  # [2,m] or [2,m,2,m]
        if self.isMatrix:
            coherences = self[:, list(modes)] # [2,n,2,m]
        if self.isVector:
            return XPTensor(backend.concat([self.tensor]+[to_clone]*times, axis=-1))
        if self.isMatrix:
            rows = backend.concat([coherences]*times, axis=-1)

    def __array__(self):
        return self._tensor

    def __rmul__(self, other: Scalar) -> XPMatrix:  # assumes other is a scalar
        if self._tensor is None:
            raise NotImplementedError("Cannot multiply a scalar by a None matrix yet")
        return XPMatrix(other * self._tensor, self.modes, self.additive, self.multiplicative)

    def __mul__(self, other: Scalar) -> XPMatrix:
        return other * self

    def __matmul__(self, other: Union[XPMatrix, XPVector]) -> Optional[Union[XPMatrix, XPVector]]:
        if self._tensor is None and other._tensor is None:
            return other
        if self._tensor is None:  # only self is None
            if self.additive:
                return self
            return other
        if other._tensor is None:
            if other.additive:
                return other
            return self
        if other.isVector:
            xxpp = self.matvec_at_modes(self.to_xxpp(), other.to_xxpp(), modes_a=self.modes, modes_b=other.modes)
            modes = 
        else: # other is XPMatrix:
            xxpp = self.matmat_at_modes(self.to_xxpp(), other.to_xxpp(), modes_a=self.modes, modes_b=other.modes)
        return XPMatrix.from_xxpp(xxpp, sorted(list(set(self.modes[0]) | set(other.modes[0]))), self.additive or other.additive)

    def matmat_at_modes(self, xxpp_a: Matrix, xxpp_b: Matrix, modes_a: Tuple[List[int],List[int]], modes_b: Tuple[List[int],List[int]]) -> Matrix:
        if modes_a == modes_b:
            return backend.matmul(xxpp_a, xxpp_b)
        modes = set(modes_a) | set(modes_b)
        self._get_inmode_subset(xxpp_a, modes)
        out = backend.eye(2 * len(modes), dtype=xxpp_a.dtype)
        out = backend.left_matmul_at_modes(xxpp_b, out, modes_b)
        out = backend.left_matmul_at_modes(xxpp_a, out, modes_a)
        return out

    def matvec_at_modes(self, xxpp_a: Matrix, xxpp_b: Vector, modes_a: Tuple[List[int], List[int]], modes_b: List[int]) -> Vector:
        if modes_a[1] == modes_b:
            return backend.matvec(xxpp_a, xxpp_b)
        modes = set(modes_a) | set(modes_b)
        out = backend.zeros(2 * len(modes), dtype=xxpp_b.dtype)
        out = backend.add_at_modes(xxpp_b, out, modes_b)
        out = backend.matvec_at_modes(xxpp_a, out, modes_a)
        return out

    def vecvec_at_modes(self, xxpp_a: Vector, xxpp_b: Vector, modes_a: List[int], modes_b: List[int]) -> Scalar:
        if modes_a == modes_b:
            return backend.sum(xxpp_a * xxpp_b)
        modes = set(modes_a) & set(modes_b)  # only the common modes
        out = backend.vecvec_at_modes(xxpp_a, out, modes)
        return out

    def __add__(self, other: XPTensor) -> Optional[XPTensor]:
        if self._tensor is None and other._tensor is None:
            return XPTensor(None, [], self.additive and other.additive)  # NOTE: this says 1+1 = 1
        if self._tensor is None:  # only self is None
            if self.additive:
                return other
            return ValueError("0+1 not implemented 🥸")
        if other._tensor is None:  # only other is None
            if other.additive:
                return self
            return ValueError("1+0 not implemented 🥸")
        xxpp = self.add_at_modes(self.to_xxpp(), other.to_xxpp(), modes_a=self.modes, modes_b=other.modes)
        return XPTensor.from_xxpp(xxpp, sorted(list(set(self.modes) | set(other.modes))), self.additive and other.additive)

    def add_at_modes(self, xxpp_a: Matrix, xxpp_b: Matrix, modes_a: List[int], modes_b: List[int]) -> Matrix:
        if modes_a == modes_b:
            return xxpp_a + xxpp_b
        modes = set(modes_a) | set(modes_b)
        out = backend.zeros((2 * len(modes), 2 * len(modes)), dtype=xxpp_a.dtype)
        out = backend.add_at_modes(out, xxpp_a, modes_a)
        out = backend.add_at_modes(out, xxpp_b, modes_b)
        return out

    def __sub__(self, other: XPTensor) -> Optional[XPTensor]:
        return self + (-1) * other

    def __repr__(self) -> str:
        return f"XPTensor(modes={self.modes}, additive={self.additive}, _tensor={self._tensor})"

    def __getitem__(self, item: Union[int, slice, List[int]]) -> XPTensor:
        r"""
        Returns modes or subsets of modes from the XPTensor, or coherences between modes.
        Examples:
        >>> T[0]  # returns mode 0
        >>> T[0:3]  # returns modes 0, 1, 2
        >>> T[[0, 2, 12]]  # returns modes 0, 2 and 12
        >>> T[0:3, [0, 10]]  # returns the coherence between modes 0,1,2 and 0,10 (rectangular block)
        >>> T[[0,1,2], [0, 10]]  # equivalent to T[0:3, 0:10]
        """
        if self._tensor is None:
            return XPTensor(None, self.__getitem_list(item), self.additive)
        lst1 = self.__getitem_list(item)
        lst2 = lst1
        if isinstance(item, tuple) and len(item) == 2:
            if self.ndim == 1:
                raise ValueError("XPTensor is a vector")
            lst1 = self.__getitem_list(item[0])
            lst2 = self.__getitem_list(item[1])
        gather = self.backend.gather(self.tensor, lst1, axis=1)
        if self.ndim == 2:
            gather = (self.backend.gather(gather, lst2, axis=3),)
        return gather  # self.backend.reshape(gather, (2*len(lst1), 2*len(lst2))[:self.ndim])
    # TODO: write a tensor wrapper to use __setitem__ with TF (it's already possible with pytorch)
    # (must be differentiable!)
    # def __setitem__(self, key: Union[int, slice, List[int]], value: XPTensor) -> None:
    #     if isinstance(key, int):
    #         self._tensor[:, key, :, key] = value._tensor
    #     elif isinstance(key, slice):
    #         self._tensor[:, key, :, key] = value._tensor
    #     elif isinstance(key, List):
    #         self._tensor[:, key, :, key] = value._tensor
    #     else:
    #         raise TypeError("Invalid index type")

    def __getitem_list(self, item):
        if isinstance(item, int):
            lst = [item]
        elif isinstance(item, slice):
            lst = list(range(item.start or 0, item.stop or self.nmodes, item.step))
        elif isinstance(item, List):
            lst = item
        else:
            lst = None  # is this right?
        return lst

    @property
    def T(self) -> XPTensor:
        if self.isVector:
            raise ValueError("Cannot transpose a vector")
        if self._tensor is None:
            return XPTensor(None, [], self.additive)
        return XPTensor(backend.transpose(self._tensor), self.modes)

