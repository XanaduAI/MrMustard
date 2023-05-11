# Copyright 2022 Xanadu Quantum Technologies Inc.

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

from mrmustard.math import Math
from mrmustard.math.datatypes import AutoData
from mrmustard.physics.representations import Glauber, Husimi, Wigner
from mrmustard.types import Number

math = Math()


class Representation:
    r"""Abstract representation class, no implementations in here, just
    the right calls to the appropriate repA -> repB methods with default
    routing via Q representation.

    The Representation class enables functionality like:
    >>> wigner = Wigner(cov = ..., means = ...)
    >>> stellar = Stellar(wigner)

    In this example it automatically understands that cov and means are gaussian data,
    and in the conversion to Stellar they become A,B,C i.e. quadratic poly data.

    Args:
        repr (Representation): Representation to be converted
        **kwargs: Keyword arguments to be passed to the data class
    """

    def __init__(self, repr: Representation = None, /, **kwargs):
        if repr is not None and len(kwargs) > 0:
            raise TypeError(
                "Either pass a representation (positional only) or keyword arguments, not both"
            )
        if isinstance(repr, Representation):
            self = self.from_representation(repr)  # triggers default sequence of transformations
        elif repr is None:
            self.data = AutoData(**kwargs)
        else:
            raise TypeError(f"Cannot initialize from {repr.__class__.__qualname__}")

    def __getattr__(self, name):
        # Intercept access to non-existent attributes of Representation like 'ket',
        # 'cov', etc and pass it to self.data.
        # This way we can access the data attributes directly from the representation
        try:
            return getattr(self.data, name)
        except AttributeError:
            raise AttributeError(f"{self.__class__.__qualname__}'s data has no attribute {name}")

    def from_representation(self, repr):
        # If the representation is already of the right type, return it.
        if isinstance(repr, self.__class__):
            return repr
        # Otherwise, call the appropriate from_xyz method
        return getattr(self, f"from_{repr.__class__.__qualname__.lower()}")(
            repr
        )  # calls the first transformation in the sequence
    
    # Shortest trajectory in the MVP graph
    ## From leaves = from branch
    def from_charp(self, charP):
        return self.from_glauber(Glauber(charP))

    def from_charq(self, charQ):
        return self.from_husimi(Husimi(charQ))

    def from_charw(self, charW):
        return self.from_wigner(Wigner(charW))

    def from_wavefunctionp(self, wavefunctionP):
        return self.from_wavefunctionx(WavefunctionX(wavefunctionP))

    ## From branches = from trunk
    def from_stellar(self, stellar):
        return self.from_husimi(Husimi(stellar))
    
    def from_fock(self, fock):
        return self.from_stellar(Stellar(fock))

    def from_wavefunctionx(self, wavefunctionX):
        return self.from_fock(Fock(wavefunctionX))

    def from_stellar(self, stellar):
        return self.from_husimi(Husimi(stellar))

    def from_wigner(self, wigner):
        return self.from_husimi(Husimi(wigner))

    def from_glauber(self, glauber):
        return self.from_husimi(Husimi(glauber))

    def _typecheck(self, other, operation: str):
        if self.__class__ != other.__class__:
            raise TypeError(
                f"Cannot perform {operation} between {self.__class__} and {other.__class__}"
            )

    # The implementation of these ops between representations depends on the underlying data
    def __add__(self, other):
        self._typecheck(other, "addition")
        # NOTE: self and other are in the same repr, so data.preferred exists and is the same
        return self.__class__(self.data.preferred + other.data.preferred)

    def __sub__(self, other):
        self._typecheck(other, "subtraction")
        return self.__class__(self.data.preferred - other.preferred_data)

    def __rmul__(self, other):
        return self * other

    def __mul__(self, other):
        if isinstance(other, Number):
            return self.__class__(self.data.preferred * other)
        elif isinstance(other, Representation):
            self._typecheck(other, "multiplication")
            return self.__class__(self.data.preferred * other.data.preferred)

    def __truediv__(self, other):
        if isinstance(other, Number):
            return self.__class__(self.data.preferred / other)
        elif isinstance(other, Representation):
            self._typecheck(other, "division")
        return self.__class__(self.data.preferred / other.data.preferred)
