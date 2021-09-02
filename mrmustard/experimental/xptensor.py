from __future__ import annotations
from mrmustard import Backend
backend = Backend()
from mrmustard._typing import *


class XPTensor:
    r"""A representation of tensors in phase space.
    A cov tensor is stored as a matrix of shape (2*nmodes_out, 2*nmodes_in) in xxpp ordering, but internally we utilize a
    (2, nmodes_out, 2, nmodes_in) representation to simplify several operations.
    A means vector is stored as a vector of shape (2*nmodes), but analogously to the cov case,
    we use the (2, nmodes) representation to perform simplified operations.
    
    TODO: implement a sparse version for representing graph states.

    Arguments:
        tensor: The tensor to be represented.
        modes: The modes to be represented.
        additive: Whether the tensor behaves like 0 for addition
        multiplicative: Whether the tensor behaves like 1 for multiplication
    """

    def __init__(self, tensor: Optional[Tensor] = None, modes=[], additive=None, multiplicative=None) -> None:
        self.validate(tensor, modes)
        self.additive = bool(additive) or not bool(multiplicative)  # I love python
        self.isVector = None if tensor is None else tensor.ndim == 1
        self.shape = None if tensor is None else [t // 2 for t in tensor.shape]
        self.ndim = None if tensor is None else tensor.ndim
        self._modes = modes
        self._tensor = tensor

    @property
    def dtype(self):
        if self._tensor is None:
            return None
        return self._tensor.dtype

    @property
    def modes(self) -> List[int]:
        if self._modes == []:
            return list(range(self.ndim))
        return self._modes

    @property
    def multiplicative(self) -> bool:
        return not self.additive

    @property
    def isMatrix(self) -> bool:
        return not self.isVector

    @property
    def tensor(self):
        if self._tensor is None:
            return None
        return backend.reshape(self._tensor, [k for n in self.shape for k in (2, n)])  # [2,n] or [2,n,2,n]

    def validate(self, tensor: Optional[Tensor], modes: List[int]) -> None:
        if tensor is None:
            return
        if len(tensor.shape) > 2:
            raise ValueError(f"Tensor must be at most 2D, got {tensor.ndim}D")
        if len(modes) > tensor.shape[-1] // 2:
            raise ValueError(f"Too many modes ({len(modes)}) for tensor with {tensor.shape[-1]//2} modes")
        if len(modes) < tensor.shape[-1] // 2:
            raise ValueError(f"Too few modes ({len(modes)}) for tensor with {tensor.shape[-1]//2} modes")

    @classmethod
    def from_xxpp(cls, tensor: Union[Matrix, Vector], modes: List[int], additive: bool = None, multiplicative: bool = None) -> XPTensor:
        return XPTensor(tensor, modes, additive, multiplicative)

    @classmethod
    def from_xpxp(cls, tensor: Union[Matrix, Vector], modes: List[int], additive: bool = None, multiplicative: bool = None) -> XPTensor:
        if tensor is not None:
            tensor = cls.backend.reshape(tensor, [k for n in tensor.shape for k in (n, 2)])
            tensor = cls.backend.transpose(tensor, (1, 0, 3, 2)[: 2 * tensor.ndim])
            tensor = cls.backend.reshape(tensor, [2 * s for s in tensor.shape])
        return cls(tensor, modes, additive, multiplicative)

    def to_xpxp(self) -> Optional[Matrix]:
        if self._tensor is None:
            return None
        tensor = backend.transpose(self.tensor, (1, 0, 3, 2)[: 2 * self.ndim])
        return backend.reshape(tensor, [2 * s for s in self.shape])

    def to_xxpp(self) -> Optional[Matrix]:
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


    def __mul__(self, other: XPTensor) -> Optional[XPTensor]:
        if self._tensor is None and other._tensor is None:
            return XPTensor(None, self.additive or other.additive)
        if self._tensor is None:  # only self is None
            if self.additive:
                return self
            return other
        if other._tensor is None:
            if other.additive:
                return other
            return self
        xxpp = self.mul_at_modes(self.to_xxpp(), other.to_xxpp(), modes_a=self.modes, modes_b=other.modes)
        return XPTensor.from_xxpp(xxpp, sorted(list(set(self.modes) | set(other.modes))), self.additive or other.additive)

    def mul_at_modes(self, xxpp_a: Matrix, xxpp_b: Matrix, modes_a: List[int], modes_b: List[int]) -> Matrix:
        if modes_a == modes_b:
            return xxpp_a * xxpp_b
        modes = set(modes_a) | set(modes_b)
        out = backend.eye(2 * len(modes), dtype=xxpp_a.dtype)
        out = backend.left_mul_at_modes(xxpp_b, out, modes_b)  # TODO: implement this in backend
        out = backend.left_mul_at_modes(xxpp_a, out, modes_a)
        return out

    def __matmul__(self, other: XPTensor) -> Optional[XPTensor]:
        if self._tensor is None and other._tensor is None:
            return XPTensor(None, [], self.additive or other.additive)
        if self._tensor is None:  # only self is None
            if self.additive:
                return self
            return other
        if other._tensor is None:
            if other.additive:
                return other
            return self
        if self.isMatrix and other.isVector:
            xxpp = self.matvec_at_modes(self.to_xxpp(), other.to_xxpp(), modes_a=self.modes, modes_b=other.modes)
        elif self.isVector and other.isMatrix:
            xxpp = self.matvec_at_modes(other.T.to_xxpp(), self.to_xxpp(), modes_a=other.modes, modes_b=self.modes)
        elif self.isMatrix and other.isMatrix:
            xxpp = self.matmat_at_modes(self.to_xxpp(), other.to_xxpp(), modes_a=self.modes, modes_b=other.modes)
        else:
            xxpp = self.vecvec_at_modes(self.to_xxpp(), other.to_xxpp(), modes_a=self.modes, modes_b=other.modes)
            if not hasattr(xxpp, 'shape') or len(xxpp.shape) == 1:
                return xxpp
        return XPTensor.from_xxpp(xxpp, sorted(list(set(self.modes) | set(other.modes))), self.additive or other.additive)

    def matmat_at_modes(self, xxpp_a: Matrix, xxpp_b: Matrix, modes_a: List[int], modes_b: List[int]) -> Matrix:
        if modes_a == modes_b:
            return backend.matmul(xxpp_a, xxpp_b)
        modes = set(modes_a) | set(modes_b)
        out = backend.eye(2 * len(modes), dtype=xxpp_a.dtype)
        out = backend.left_matmul_at_modes(xxpp_b, out, modes_b)
        out = backend.left_matmul_at_modes(xxpp_a, out, modes_a)
        return out

    def matvec_at_modes(self, xxpp_a: Matrix, xxpp_b: Vector, modes_a: List[int], modes_b: List[int]) -> Vector:
        if modes_a == modes_b:
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

    def __repr__(self) -> str:
        return f"XPTensor(modes={self.modes}, additive={self.additive}, _tensor={self._tensor})"

    def __getitem__(self, item: Union[int, slice, List[int], Tuple]) -> XPTensor:
        r"""
        Returns modes or subsets of modes from the XPTensor when item is an int or a slice,
        or coherences between modes when item is a tuple.
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
        gather = backend.gather(self.tensor, lst1, axis=1)
        if self.ndim == 2:
            gather = (backend.gather(gather, lst2, axis=3),)
        return gather  # backend.reshape(gather, (2*len(lst1), 2*len(lst2))[:self.ndim])

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
