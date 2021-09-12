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
        tensor: The tensor to be represented, in xxpp ordering. Use XPTensor.from_xpxp() to initialize from an xpxp-ordered tensor.
        modes: input-output modes if matrix, just modes if vector
        additive: Whether the null tensor behaves like 0 for addition
        multiplicative: Whether the null tensor behaves like 1 for multiplication
    """

    def __init__(self,
                tensor: Optional[Union[Matrix, Vector]] = None,
                modes: Union[Tuple[List[int], List[int]], List[int]] = None,
                additive=None, multiplicative=None
                ):
        self.additive = bool(additive) or not bool(multiplicative)  # I love python
        self.shape = None if tensor is None else [t // 2 for t in tensor.shape]
        self.ndim = None if tensor is None else tensor.ndim
        self.isVector = None if tensor is None else self.ndim == 1
        self.isMatrix = None if tensor is None else self.ndim == 2
        if modes is None and tensor is not None:
            if self.isVector:
                modes = [m for m in range(tensor.shape[0] // 2)], []
            if self.isMatrix:
                modes = [m for m in range(tensor.shape[0] // 2)], [m for m in range(tensor.shape[1] // 2)]
        assert set(modes[0]).isdisjoint(modes[1]) or set(modes[0]) == set(modes[1])  # either a coherence or a diagonal block
        self.modes = modes
        self.tensor = None if tensor is None else backend.reshape(tensor, [k for n in tensor.shape for k in (2, n)])

    @property
    def isCoherence(self):
        return self.isMatrix and self.modes[0] != self.modes[1]

    @property
    def multiplicative(self) -> bool:
        return not bool(self.additive)

    @property
    def outmodes(self) -> List[int]:
        return self.modes[0]

    @property
    def inmodes(self) -> List[int]:
        return self.modes[1]

    @classmethod
    def from_xxpp(cls, tensor: Union[Matrix, Vector], modes: Optional[Tuple[List[int],List[int]]], additive: bool = None, multiplicative: bool = None) -> XPTensor:
        return XPTensor(tensor, modes, additive, multiplicative)

    @classmethod
    def from_xpxp(cls, tensor: Union[Matrix, Vector], modes: Optional[Tuple[List[int], List[int]]], additive: bool = None, multiplicative: bool = None) -> XPTensor:
        if tensor is not None:
            tensor = backend.reshape(tensor, [k for n in matrix.shape for k in (n, 2)])
            tensor = backend.transpose(tensor, (1, 0, 3, 2)[: 2 * tensor.ndim])
            tensor = backend.reshape(tensor, [2 * s for s in matrix.shape])
        return cls(tensor, modes, additive, multiplicative)

    def to_xpxp(self) -> Optional[Matrix]:
        if self.tensor is None:
            return None
        tensor = backend.transpose(self.tensor, (1, 0, 3, 2)[: 2 * self.ndim])
        return backend.reshape(tensor, [2 * s for s in self.shape])

    def to_xxpp(self) -> Optional[Tensor]:
        if self.tensor is None:
            return None
        return backend.reshape(self.tensor, [2 * s for s in matrix.shape])

    def clone(self, modes: Sequence[int], times: int):  # TODO: finish this
        r"""Artificially clones the given modes a given number of times and includes them as new modes."""
        if self.tensor is None:
            pass
        if self.isCoherence:
            raise NotImplementedError("Cloning of pure coherences is not yet implemented.")
        if self.isVector:
            return XPTensor(backend.concat([self.tensor] + ([to_clone] for _ in range(times)), axis=1), (self.outmodes + outmodes*times, self.inmodes), self.additive, self.multiplicative)
        else:
            pass

    def __array__(self):
        return self.tensor

    def __rmul__(self, other: Scalar) -> XPTensor:
        if self.tensor is None:
            if self.multiplicative:
                raise NotImplementedError("Cannot multiply a scalar by a multiplicative null tensor yet")
            else:
                return XPTensor(None, self.modes, self.additive, self.multiplicative)
        return XPTensor(other * self.tensor, self.modes, self.additive, self.multiplicative)

    def __mul__(self, other: Scalar) -> Optional[XPTensor]:
        return other * self if self.tensor is not None else None

    def __matmul__(self, other: XPTensor) -> Optional[XPTensor]:
        # both None
        if self.tensor is None and other.tensor is None:
            return XPTensor(multiplicative=self.multiplicative and other.multiplicative)
        # either None
        if self.tensor is None:
            return self if self.additive else other
        if other.tensor is None:
            return other if other.additive else self
        # Now neither self nor other is None
        if self.isMatrix and other.isVector:
            xxpp, modes = self.mode_aware_matvec(other)
        elif self.isVector and other.isMatrix:
            xxpp, modes = other.T.mode_aware_matvec(self)
        elif self.isMatrix and other.isMatrix:
            xxpp, modes = self.mode_aware_matmul(other)
        else: # self.isVector and other.isVector:
            xxpp, modes = self.mode_aware_vecvec(other)
        return XPTensor(xxpp, modes, multiplicative=self.multiplicative and other.multiplicative)

    def _join_modes_prod_right(other: XPTensor) -> Tuple[List[int], List[int], List[int]]:
        contracted = [i for i in self.inmodes if i in other.outmodes]
        uncontracted_other = [o for o in other.outmodes if o not in contracted]
        if set(self.outmodes).intersection(uncontracted_other) is not None:
            raise ValueError("outmodes are not disjoint")
        uncontracted_self = [i for i in self.inmodes if i not in contracted]
        if set(other.inmodes).intersection(uncontracted_self) is not None:
            raise ValueError("inmodes are not disjoint")
        return uncontracted_self, contracted, uncontracted_other

    def _mode_aware_matmul(self, other:XPTensor) -> Tuple[Matrix, Tuple[List[int], List[int]]]:
        if self.inmodes == other.outmodes:
            return backend.tensordot(self.tensor, other.tensor, ((2,3),(0,1))), (self.outmodes, other.inmodes)
        # otherwise, there's a mismatch
        uncontracted_self, contracted, uncontracted_other = self._join_modes_prod_right(other)
        if set(self.inmodes).issubset(other.outmodes): # other is bigger than us and we plug entirely into it
            if self.additive: # we trow away lots of stuff from other
                return backend.tensordot(self.tensor[:,:,:,contracted], other.tensor[:,contracted], ((2,3),(0,1))), (modes_out, modes_in)
        
        product = backend.tensordot(self.tensor[:,:,:,contracted], other[:,contracted,:,:], ((2,3),(0,1))) # result without the uncontracted modes
        if self.additive:
            if other.additive:
                return product, (modes_out, modes_in)
            else:
                other.tensor[:,modes_out] = product
                return other, (modes_out, modes_in)
            self.tensor[:,modes_out,:,contracted] = product[:,:,:,contracted]
        return self.to_xxpp(), (modes_out, modes_in)
        else:
            self.tensor[:,contracted,:,:] = backend.reshape(product, (2, len(contracted), 2, len(contracted)))
        return other.to_xxpp(), (modes_out, modes_in)

    def mode_aware_matvec(self, xxpp_a: Matrix, xxpp_b: Vector, modes_a: List[int], modes_b: List[int]) -> Vector:
        if modes_a == modes_b:
            return backend.matvec(xxpp_a, xxpp_b)
        modes = set(modes_a) | set(modes_b)
        out = backend.zeros(2 * len(modes), dtype=xxpp_b.dtype)
        out = backend.add_at_modes(xxpp_b, out, modes_b)
        out = backend.matvec_at_modes(xxpp_a, out, modes_a)
        return out, modes

    def mode_aware_vecvec(self, xxpp_a: Vector, xxpp_b: Vector, modes_a: List[int], modes_b: List[int]) -> Scalar:
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

    def __getitem__(self, item: Union[int, slice, List[int]]) -> XPTensor:
        r"""
        Returns modes or subsets of modes from the XPTensor, or coherences between modes.
        Examples:
        >>> T[0]  # returns obly outmode 0 (result has only inmodes, i.e. it's a phase space covector)
        >>> T[:, 0]  # returns only inmode 0 (result has only outmodes, i.e. it's a phase space vector)
        >>> T[0, 1]  # returns coherence between modes 0 and 1
        >>> T[:, 0:3]  # returns block with all outmodes and inmodes 0, 1, 2
        >>> T[[0, 2, 12]]  # returns block with outmodes 0, 2, 12 and all inmodes
        >>> T[0:3, [0, 10]]  # returns block with outmodes 0, 1, 2 and inmodes 0, 10
        
        """
        if self._tensor is None:
            return XPTensor(None, self.__getitem_list(item), self.additive)
        if isinstance(item, tuple) and len(item) == 2:
            if self.isVector:
                raise ValueError("Cannot index a vector with 2 indices")
            return XPTensor(backend.getitem(self.tensor, (slice(None), item[0], slice(None), item[1])), self.modes, self.additive)
        if isinstance(item, int):
            return XPTensor(backend.getitem(self.tensor, (slice(None), item, slice(None))[:2+self.isMatrix]), self.modes, self.additive)
        if isinstance(item, slice):
            return XPTensor(backend.getitem(self.tensor, (item, slice(
        # the right indices (don't exceed 2 or 1 index)

        # lst1 = self.__getitem_list(item)
        # lst2 = lst1
        # if isinstance(item, tuple) and len(item) == 2:
        #     if self.ndim == 1:
        #         raise ValueError("XPTensor is a vector")
        #     lst1 = self.__getitem_list(item[0])
        #     lst2 = self.__getitem_list(item[1])
        # gather = self.backend.gather(self.tensor, lst1, axis=1)
        # if self.ndim == 2:
        #     gather = (self.backend.gather(gather, lst2, axis=3),)
        # return gather  # self.backend.reshape(gather, (2*len(lst1), 2*len(lst2))[:self.ndim])

    def __setitem__(self, key, value: XPTensor):
        if self.isMatrix:
            self._tensor = backend.setitem(self.tensor, (slice(), key[0], slice(), key[1]], value.tensor[:, key[0], :, key[1]])
        else:
            self._tensor = backend.setitem(self.tensor, (slice(), key), value.tensor[:, key])

    def __getitem_list(self, item=None):
        if isinstance(item, int):
            lst = [item]
        elif isinstance(item, slice):
            lst = list(range(item.start or 0, item.stop or self.nmodes, item.step))
        elif isinstance(item, List):
            lst = np.array(item)
        elif item is None:
            lst = slice(None)
        else:
            raise ValueError(f"Invalid item: {item}")
        return lst

    @property
    def T(self) -> XPTensor:
        if self.isVector:
            raise ValueError("Cannot transpose a vector")
        if self._tensor is None:
            return XPTensor(None, [], self.additive)
        return XPTensor(backend.transpose(self._tensor), self.modes)
