from __future__ import annotations
from mrmustard import Backend
from mrmustard._typing import *


class XPTensor:
    r"""A representation of tensors in phase space.
    A cov tensor is stored as a matrix of shape (2*nmodes, 2*nmodes) in xxpp ordering, but internally we heavily utilize a
    (2, nmodes, 2, nmodes) representation to simplify several operations.
    A means vector is stored as a vector of shape (2*nmodes), but analogously to the cov case,
    we use the (2, nmodes) representation to perform simplified operations.
    """

    backend = Backend()

    def __init__(self, tensor: Optional[Tensor] = None, modes=[], additive=None, multiplicative=None) -> None:
        self.validate(tensor, modes)
        self.additive = bool(additive) or not bool(multiplicative)  # I love python
        self.isVector = None if tensor is None else tensor.ndim == 1
        self.shape = None if tensor is None else [t // 2 for t in tensor.shape]
        self.ndim = None if tensor is None else tensor.ndim
        self.modes = modes
        self._tensor = tensor

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
        return self.backend.reshape(self._tensor, [k for n in self.shape for k in (2, n)])  # [2,n] or [2,n,2,n]

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
        tensor = self.backend.transpose(self.tensor, (1, 0, 3, 2)[: 2 * self.ndim])
        return self.backend.reshape(tensor, [2 * s for s in self.shape])

    def to_xxpp(self) -> Optional[Matrix]:
        return self._tensor

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
        out = self.backend.eye(2 * len(modes), dtype=xxpp_a.dtype)
        out = self.backend.left_mul_at_modes(xxpp_b, out, modes_b)
        out = self.backend.left_mul_at_modes(xxpp_a, out, modes_a)
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
        xxpp = self.matmul_at_modes(self.to_xxpp(), other.to_xxpp(), modes_a=self.modes, modes_b=other.modes)
        return XPTensor.from_xxpp(xxpp, sorted(list(set(self.modes) | set(other.modes))), self.additive or other.additive)

    def matmul_at_modes(self, xxpp_a: Matrix, xxpp_b: Matrix, modes_a: List[int], modes_b: List[int]) -> Matrix:
        if modes_a == modes_b:
            return self.backend.matmul(xxpp_a, xxpp_b)
        modes = set(modes_a) | set(modes_b)
        out = self.backend.eye(2 * len(modes), dtype=xxpp_a.dtype)
        out = self.backend.left_matmul_at_modes(xxpp_b, out, modes_b)
        out = self.backend.left_matmul_at_modes(xxpp_a, out, modes_a)
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
        out = self.backend.zeros((2 * len(modes), 2 * len(modes)), dtype=xxpp_a.dtype)
        out = self.backend.add_at_modes(out, xxpp_a, modes_a)
        out = self.backend.add_at_modes(out, xxpp_b, modes_b)
        return out

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

    # TODO: write a tensor wrapper to use the method here below with TF as well (it's already possible with pytorch)
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
        if self._tensor is None:
            return XPTensor(None, [], self.additive)
        return XPTensor(self.backend.transpose(self._tensor), self.modes)
