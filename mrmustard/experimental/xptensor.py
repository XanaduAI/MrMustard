from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty
from itertools import product
from mrmustard import Backend
import numpy as np
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
                modes: Tuple[Sequence[int], Sequence[int]] = ([], []),
                additive=None, multiplicative=None
                ):
        self.additive = bool(additive) or not bool(multiplicative)  # I love python
        if isinstance(tensor, XPTensor) and False:  # remove False when you're sure you are not abusing of this functionality
            self.shape = tensor.shape
            self.ndim = tensor.ndim
            self.isVector = tensor.isVector
            self.tensor = tensor.tensor
        else:
            self.shape = None if tensor is None else tuple([t // 2 for t in tensor.shape])
            self.ndim = None if tensor is None else len(self.shape)
            self.isVector = None if tensor is None else self.ndim == 1
            self.tensor = None if tensor is None else backend.reshape(tensor, [_ for n in self.shape for _ in (2, n)])

        if len(modes[0]) == 0 and len(modes[1]) == 0 and self.tensor is not None:
            modes = tuple(list(range(s)) for s in (self.shape+(0,) if self.isVector else self.shape))
        assert set(modes[0]).isdisjoint(modes[1]) or set(modes[0]) == set(modes[1])
        self.modes = modes

    @property
    def dtype(self):
        return None if self.tensor is None else self.tensor.dtype

    @property
    def num_modes(self) -> int:
        return len(self.outmodes)

    @property
    def isMatrix(self) -> Optional[bool]:
        return None if self.tensor is None else not self.isVector

    @property
    def isCoherence(self) -> Optional[bool]:
        return None if self.tensor is None else self.isMatrix and self.modes[0] != self.modes[1]

    @property
    def multiplicative(self) -> bool:
        return not bool(self.additive)

    @property
    def T(self) -> XPTensor:
        if self.isVector:
            raise ValueError("Cannot transpose a vector")
        if self.tensor is None:
            return self
        return XPTensor.from_tensor(backend.transpose(self.tensor, (0,3,2,1)), (self.modes[1], self.modes[0]), self.additive, self.multiplicative)

    @property
    def outmodes(self) -> List[int]:
        return self.modes[0]

    @property
    def inmodes(self) -> List[int]:
        return self.modes[1]

    @classmethod
    def from_tensor(cls,
    tensor: Union[Matrix, Vector],
    modes: Tuple[Sequence[int], Sequence[int]] = ([], []),
    additive=None, multiplicative=None) -> XPTensor:
        xptensor = cls()
        xptensor.additive = bool(additive) or not bool(multiplicative)
        xptensor.shape = tensor.shape[1::2]
        xptensor.ndim = tensor.ndim // 2
        xptensor.isVector = xptensor.ndim == 1
        if len(modes[0])==0 and len(modes[1])==0:
            modes = [list(range(s)) for s in xptensor.shape+(0,)*xptensor.isVector]
        assert set(modes[0]).isdisjoint(modes[1]) or set(modes[0]) == set(modes[1])  # either a coherence or a diagonal block
        xptensor.modes = modes
        xptensor.tensor = tensor
        return xptensor

    @classmethod
    def from_xxpp(cls,
    tensor: Union[Matrix, Vector],
    modes: Optional[Tuple[List[int],List[int]]],
    additive: bool = None,
    multiplicative: bool = None) -> XPTensor:
        return XPTensor(tensor, modes, additive, multiplicative)

    @classmethod
    def from_xpxp(cls,
    tensor: Union[Matrix, Vector],
    modes: Optional[Tuple[List[int], List[int]]],
    additive: bool = None,
    multiplicative: bool = None) -> XPTensor:
        if tensor is not None:
            tensor = backend.reshape(tensor, [_ for n in matrix.shape for _ in (n, 2)])
            tensor = backend.transpose(tensor, (1, 0, 3, 2) if self.isMatrix else (1, 0))
        return cls.from_tensor(tensor, modes, additive, multiplicative)

    def to_xpxp(self) -> Optional[Matrix]:
        if self.tensor is None:
            return None
        tensor = backend.transpose(self.tensor, (1, 0, 3, 2) if self.isMatrix else (1, 0))
        return backend.reshape(tensor, [2 * s for s in self.shape])

    def to_xxpp(self) -> Optional[Tensor]:
        if self.tensor is None:
            return None
        return backend.reshape(self.tensor, [2 * s for s in self.shape])

    def __array__(self):
        return self.to_xxpp()

    def modes_first(self) -> Optional[Tensor]:
        if self.tensor is None:
            return None
        return backend.transpose(self.tensor, (1,3,0,2) if self.isMatrix else (1,0))  # NM22 or N2
    
    def modes_last(self) -> Optional[Tensor]:
        if self.tensor is None:
            return None
        return backend.transpose(self.tensor, (0,2,1,3) if self.isMatrix else (0,1))  # 22NM or 2N

    def clone_like(self, other:XPTensor):
        r"""
        Create a new XPTensor with the same shape and modes as other. The new tensor
        has the same content as self, cloned as many times as necessary to match the shape of other. 
        """
        if other.shape == self.shape:
            return self
        if self.isCoherence:
            raise ValueError("Cannot clone a coherence block")
        if any(o % s != 0 for o, s in zip(other.shape, self.shape)):
            raise ValueError(f"Cannot clone tensor of shape {self.shape} to tensor of shape {other.shape}")
        times = other.shape[0]//self.shape[0]
        if self.isMatrix and other.isMatrix:
            tensor = backend.expand_dims(self.modes_last(), axis=4)  # shape = [2,2,N,N,1]
            tensor = backend.tile(tensor, (1, 1, 1, 1, times))  # shape = [2,2,N,N,T]
            tensor = backend.diag(tensor) # shape = [2,2,N,N,T,T]
            tensor = backend.transpose(tensor, (0, 1, 2, 4, 3, 5))  # shape = [2,2,N,T,N,T]
            tensor = backend.reshape(tensor, (2,2,times*self.shape[0],times*self.shape[1]))  # shape = [2,2,NT,NT]
        if self.isVector and other.isVector:
            tensor = backend.tile(self.expand_dims(self.tensor, axis=2), (1, 1, times))  # shape = [2,N,T]
            tensor = backend.reshape(tensor, (2, -1))  # shape = [2,NT]
        else:
            raise ValueError("Cannot clone a vector to a matrix or viceversa")
        return XPTensor.from_tensor(tensor, (other.modes[0], other.modes[1]), self.additive, self.multiplicative)

    def __rmul__(self, other: Scalar) -> XPTensor:
        if self.tensor is None:
            if self.multiplicative:
                raise NotImplementedError("Cannot multiply a scalar by a multiplicative null tensor yet")
            else:
                return XPTensor(None, self.modes, self.additive, self.multiplicative)
        self.tensor = other * self.tensor
        return self

    def __mul__(self, other: Scalar) -> Optional[XPTensor]:
        return other * self if self.tensor is not None else None

    def _join_modes_by_product(self, other: XPTensor) -> Tuple[List[int], List[int], List[int]]:
        contracted = [i for i in self.inmodes if i in other.outmodes]
        uncontracted_other = [o for o in other.outmodes if o not in contracted]
        uncontracted_self = [i for i in self.inmodes if i not in contracted]
        return contracted, uncontracted_other, uncontracted_self

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
        if self.isMatrix:
            tensor, modes = self._mode_aware_matmul(other)
        elif self.isVector and other.isMatrix:
            tensor, modes = other.T.mode_aware_matmul(self)
        else: # self.isVector and other.isVector:
            tensor, modes = self._mode_aware_vecvec(other)
        return XPTensor.from_tensor(tensor, modes, multiplicative=self.multiplicative and other.multiplicative)

    def sameModes(self, other: XPTensor) -> bool:
        return list(self.outmodes) == list(other.outmodes) and list(self.inmodes) == list(other.inmodes)

    def _mode_aware_matmul(self, other:XPTensor) -> Tuple[Tensor, Tuple[List[int], List[int]]]:
        r"""Performs matrix multiplication only on the necessary modes and
        takes care of keeping only the modes that are needed, in case of mismatch.
        See documentation for a visual explanation with coloured blocks.  #TODO: add link
        """
        if self.sameModes(other):
            return backend.tensordot(self.tensor, other.tensor, ((2,3),(0,1))), (self.outmodes, other.inmodes)
        contracted, uncontracted_other, uncontracted_self = self._join_modes_by_product(other)
        outmodes = self.outmodes + uncontracted_other if self.multiplicative else self.outmodes
        inmodes = other.inmodes + uncontracted_self if other.multiplicative else other.inmodes
        if len(set(outmodes)) != len(outmodes) or len(set(inmodes)) != len(inmodes):
            raise ValueError("modes are not disjoint")
        blue = green = purple = white = None
        if len(contracted) > 0:
            blue = backend.tensordot(backend.gather(self.tensor, contracted, axis=3), backend.gather(other.tensor, contracted, axis=1), ((2,3),(0,1)))
        if self.multiplicative and len(uncontracted_other) > 0:
            green = backend.gather(other.tensor, uncontracted_other, axis=1)
        if other.multiplicative and len(uncontracted_self) > 0:
            purple = backend.gather(self.tensor, uncontracted_self, axis=3)
        if self.multiplicative and other.multiplicative and green is not None and purple is not None and blue is not None:
            white = backend.zeros((2, green.shape[0], 2, purple.shape[1]), dtype=blue.dtype)
        if green is not None and purple is not None:
            final = backend.concat((backend.concatenate((blue, green), axis=1), backend.concatenate((purple, white), axis=1)), axis=3)
        elif green is not None and purple is None:
            final = backend.concat((blue, green), axis=1)
        elif green is None and purple is not None:
            final = backend.concat((blue, purple), axis=3)
        else:
            final = blue
        tr_out = [outmodes.index(o) for o in sorted(outmodes)]
        tr_in = [inmodes.index(i) for i in sorted(inmodes)]
        final = backend.gather(final, tr_out, axis=1)
        if self.isMatrix and len(tr_in) > 0:
            final = backend.gather(final, tr_in, axis=3)
        return final, (list(sorted(outmodes)), list(sorted(inmodes)))

    def _mode_aware_vecvec(self, other: XPTensor) -> Scalar:
        if self.sameModes(other):
            return backend.sum(self.tensor * other.tensor)
        contracted = list(set(self.outmodes) & set(other.outmodes))  # only the common modes
        return backend.sum(backend.gather(self.tensor, contracted, axis=1) * backend.gather(other.tensor, contracted, axis=1))

    def __add__(self, other: XPTensor) -> Optional[XPTensor]:
        if self.tensor is None and other.tensor is None: # both are none
            if self.multiplicative and other.multiplicative:
                raise ValueError("Cannot add two multiplicative null tensors yet")
            return XPTensor(additive = self.additive and other.additive) # 0+0 = 0
        if self.tensor is None:  # only self is None
            if self.additive:
                return other
            return ValueError("1+other not implemented 🥸")
        if other.tensor is None:  # only other is None
            if other.additive:
                return self
            return ValueError("self+1 not implemented 🥸")
        if self.isVector != other.isVector:
            raise ValueError("Cannot add a matrix and a vector")
        if self.isCoherence != other.isCoherence:
            raise ValueError("Cannot add a coherence block and a diagonal block")
        tensor, modes = self._mode_aware_add(other)
        return XPTensor.from_tensor(tensor, modes, self.additive and other.additive, self.multiplicative or other.multiplicative)
    
    def contains(self, other: XPTensor) -> bool:
        return set(self.outmodes).issuperset(other.outmodes) and set(self.inmodes).issuperset(other.inmodes)

    def _mode_aware_add(self, other: XPTensor) -> Tuple[Tensor, Tuple[List[int], List[int]]]:
        if self.sameModes(other):
            return self.tensor + other.tensor, self.modes
        outmodes = list(set(self.outmodes).union(other.outmodes))  # NOTE: beware of mode ordering
        inmodes = list(set(self.inmodes).union(other.inmodes))
        if self.contains(other):
            to_update = self.modes_first()
            to_add = [other]
        elif other.contains(self):
            to_update = other.modes_first()
            to_add = [self]
        else:  # need to add to an empty tensor
            to_update = backend.zeros((len(outmodes), len(inmodes), 2, 2) if self.isMatrix else (len(outmodes), 2), dtype=self.tensor.dtype)
            to_add = [self, other]
        for t in to_add:
            if t.isMatrix:
                indices = [t.outmodes.index(o) for o in t.outmodes], [t.inmodes.index(i) for i in t.inmodes]
                indices = list(product(*indices))
            else:
                indices = [[t.outmodes.index(o)] for o in t.outmodes]
            to_update = backend.update_add_tensor(to_update, indices, backend.reshape(t.modes_first(),(-1,2,2) if self.isMatrix else (-1,2)))
        return backend.transpose(to_update, (2,0,3,1) if self.isMatrix else (1,0)), (outmodes, inmodes)

    def __sub__(self, other: XPTensor) -> Optional[XPTensor]:
        return self + (-1) * other

    def __truediv__(self, other: Scalar) -> Optional[XPTensor]:
        return (1/other) * self

    def __repr__(self) -> str:
        return f"XPTensor(modes={self.modes}, additive={self.additive}, tensor_xpxp={self.to_xpxp()})"

    def __getitem__(self, item: Union[int, slice, List[int]]) -> Tensor:
        r"""
        Returns modes or subsets of modes from the XPTensor, or coherences between modes using an intuitive notation
        Examples:
            T[M] = self.tensor[:,M,...]
            T[M,N] = self.tensor[:,M,:,N]
            T[:,N] = self.tensor[:,:,:,N]
            T[[1,2,3],:] = self.tensor[:,[1,2,3],:,N]
            T[[1,2,3],[4,5]] = self.tensor[:,[1,2,3],:,[4,5]]  # i.e. the rows [1,2,3] and columns [4,5]
        """

        _all = slice(None)
        if isinstance(item, int):
            return XPTensor.from_tensor(backend.expand_dims(self.tensor[:,item,...], axis=1), modes=(self.modes[0][item], self.modes[1]), additive=self.additive)
        if self.tensor is None:
            return XPTensor(additive=self.additive)
        if isinstance(item, tuple) and len(item) == 2:
            if self.isVector:
                raise ValueError("Cannot index a vector with 2 indices")
        # the right indices (don't exceed 2 or 1 index)



    def __setitem__(self, key, value: XPTensor):
        if self.isMatrix:
            self._tensor = backend.setitem(self.tensor, (slice(), key[0], slice(), key[1]), value.tensor[:, key[0], :, key[1]])
        else:
            self._tensor = backend.setitem(self.tensor, (slice(), key), value.tensor[:, key])

    def __getitem_tuple(self, item=None):
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


