# abstract representation class
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from mrmustard import settings
from mrmustard.math import Math
from mrmustard.types import Number

math = Math()


class Data(ABC):
    r"""Data is a class that holds the information that is necessary to define a
    state/operation/measurement in any representation. It enables algebraic operations
    for all types of data (gaussian, array, samples, symbolic, etc).
    Algebraic operations act on the Hilbert space vectors or on ensembles
    of Hilbert vectors (which form a convex structure via the Giri monad).
    The sopecific implementation depends on both the representation and the kind of data used by it.

    These are the supported data types (as either Hilbert vectors or convex
    combinations of Hilbert vectors):
    - Gaussian generally stacks cov, mean, coeff along a batch dimension.
    - QuadraticPoly is a representation of the Gaussian as a quadratic polynomial.
    - Array (ket/dm) operates with the data arrays themselves.
    - Samples (samples) operates on the (x,f(x)) pairs with interpolation.
    - Symbolic (symbolic) operates on the symbolic expression via sympy.

    This class is abstract and Gaussian, Array, Samples, Symbolic inherit from it.
    """

    @property
    def preferred(self):
        for data_type in settings.PREFERRED_DATA_ORDER:
            if hasattr(self, data_type):
                return getattr(self, data_type)

    @abstractmethod
    def __add__(self, other):
        pass

    @abstractmethod
    def __mul__(self, other):
        pass

    @abstractmethod
    def __and__(self, other):  # tensor product
        pass

    def __sub__(self, other):
        return self.__add__(-1 * other)

    def __neg__(self):
        return -1 * self

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.__mul__(1 / other)  # this numerically naughty


class MatVecData(Data):
    def __init__(self, mat, vec, coeff):
        self.mat = math.atleast_3d(mat)
        self.vec = math.atleast_2d(vec)
        self.coeff = math.atleast_1d(coeff)

    def __add__(self, other: MatVecData):
        if np.allclose(self.mat, other.mat) and np.allclose(self.vec, other.vec):
            return MatVecData(self.mat, self.vec, self.coeff + other.coeff)
        return MatVecData(
            math.concat([self.mat, other.mat], axis=0),
            math.concat([self.vec, other.vec], axis=0),
            math.concat([self.coeff, other.coeff], axis=0),
        )

    def simplify(self):
        to_check = set(range(len(self.mat)))
        removed = set()
        while to_check:
            i = to_check.pop()
            for j in to_check.copy():
                if np.allclose(self.mat[i], self.mat[j]) and np.allclose(self.vec[i], self.vec[j]):
                    self.coeff[i] += self.coeff[j]
                    to_check.remove(j)
                    removed.add(j)
        to_keep = [i for i in range(len(self.mat)) if i not in removed]
        self.mat = self.mat[to_keep]
        self.vec = self.vec[to_keep]
        self.coeff = self.coeff[to_keep]

    def __eq__(self, other):
        return (
            np.allclose(self.mat, other.mat)
            and np.allclose(self.vec, other.vec)
            and np.allclose(self.coeff, other.coeff)
        )

    def __and__(self, other):
        mat = []
        vec = []
        coeff = []
        for c1 in self.mat:
            for c2 in other.mat:
                mat.append(math.block_diag([c1, c2]))
        for m1 in self.mean:
            for m2 in other.mean:
                vec.append(math.concat([m1, m2], axis=-1))
        for c1 in self.coeff:
            for c2 in other.coeff:
                coeff.append(c1 * c2)
        mat = math.astensor(mat)
        vec = math.astensor(vec)
        coeff = math.astensor(coeff)
        return self.__class__(mat, vec, coeff)


class GaussianData(MatVecData):
    def __init__(self, cov=None, mean=None, coeff=None):
        r"""
        Gaussian data: covariance, mean, coefficient.
        Each of these has a batch dimension, and the length of the
        batch dimension is the same for all three.
        These are the parameters of a linear combination of Gaussians,
        which is Gaussian if there is only one contribution for each.
        Each contribution parametrizes the Gaussian function:
        `coeff * exp(-0.5*(x-mean)^T cov^-1 (x-mean))`.
        Args:
            cov (batch, dim, dim): covariance matrices (real symmetric)
            mean  (batch, dim): means (real)
            coeff (batch): coefficients (complex)
        """
        # TODO handle missing data
        # TODO switch to data/kwargs?
        if isinstance(cov, QuadraticPolyData):  # enables GaussianData(quadraticdata)
            poly = cov  # for readability
            inv_A = math.inv(poly.A)
            cov = 2 * inv_A
            mean = 2 * math.solve(poly.A, poly.b)
            coeff = poly.c * math.cast(
                math.exp(0.5 * math.einsum("bca,bcd,bde->bae", mean, cov, mean)), poly.c.dtype
            )
        else:
            super().__init__(cov, mean, coeff)

    @property
    def cov(self):
        return self.mat

    @cov.setter
    def cov(self, value):
        self.mat = value

    @property
    def mean(self):
        return self.vec

    @mean.setter
    def mean(self, value):
        self.vec = value

    def __mul__(self, other):
        if isinstance(other, Number):
            return GaussianData(
                self.cov, self.mean, self.coeff * math.cast(other, self.coeff.dtype)
            )
        elif isinstance(other, GaussianData):
            # cov matrices: c1 (c1 + c2)^-1 c2 for each pair of cov matrices in the batch
            covs = []
            for c1 in self.cov:
                for c2 in other.cov:
                    covs.append(math.matmul(c1, math.solve(c1 + c2, c2)))
            # means: c1 (c1 + c2)^-1 m2 + c2 (c1 + c2)^-1 m1 for each pair of cov matrices in the batch
            means = []
            for c1, m1 in zip(self.cov, self.mean):
                for c2, m2 in zip(other.cov, other.mean):
                    means.append(
                        math.matvec(c1, math.solve(c1 + c2, m2))
                        + math.matvec(c2, math.solve(c1 + c2, m1))
                    )
            cov = math.astensor(covs)
            mean = math.astensor(means)
            coeffs = []
            for c1, m1, c2, m2, c3, m3, co1, co2 in zip(
                self.cov, self.mean, other.cov, other.mean, cov, mean, self.coeff, other.coeff
            ):
                coeffs.append(
                    co1
                    * co2
                    * math.exp(
                        0.5 * math.sum(m1 * math.solve(c1, m1), axes=-1)
                        + 0.5 * math.sum(m2 * math.solve(c2, m2), axes=-1)
                        - 0.5 * math.sum(m3 * math.solve(c3, m3), axes=-1)
                    )
                )

            coeff = math.astensor(coeffs)
            return GaussianData(cov, mean, coeff)
        else:
            raise TypeError(f"Cannot multiply GaussianData with {other.__class__.__qualname__}")


class QuadraticPolyData(MatVecData):
    def __init__(self, A=None, b=None, c=None):
        r"""
        Quadratic Gaussian data: quadratic coefficients, linear coefficients, constant.
        Each of these has a batch dimension, and the batch dimension is the same for all of them.
        They are the parameters of a Gaussian expressed as `c * exp(-x^T A x + x^T b)`.
        Args:
            A (batch, dim, dim): quadratic coefficients
            b (batch, dim): linear coefficients
            c (batch): constant
        """
        if isinstance(A, GaussianData):
            A = -math.inv(A.cov)
            b = math.inv(A.cov) @ A.mean
            c = A.coeff * np.einsum("bca,bcd,bde->bae", A.mean, math.inv(A.cov), A.mean)
        super().__init__(A, b, c)

    @property
    def A(self):
        return self.mat

    @A.setter
    def A(self, value):
        self.mat = value

    @property
    def b(self):
        return self.vec

    @b.setter
    def b(self, value):
        self.vec = value

    def __mul__(self, other):
        if isinstance(
            other, Number
        ):  # TODO: this seems to deal only with the case of self and other being a single gaussian
            return QuadraticPolyData(self.A, self.b, self.c * other)
        elif isinstance(other, QuadraticPolyData):
            return QuadraticPolyData(
                self.A + other.A, self.b + other.b, self.c * other.c
            )  # TODO: invert decomposed covs instead
        else:
            raise TypeError(
                f"Cannot multiply QuadraticPolyData with {other.__class__.__qualname__}"
            )


def AutoData(**kwargs):
    r"""Automatically choose the data type based on the arguments.
    If the arguments contain any combination of 'cov', 'mean', 'coeff' then it is GaussianData.
    If the arguments contain any combination of 'A', 'b', 'c' then it is QuadraticPolyData.
    If the arguments contain 'mesh' then it is MeshData.

    """
    if "cov" in kwargs or "mean" in kwargs or "coeff" in kwargs:
        return GaussianData(**kwargs)
    elif "A" in kwargs or "b" in kwargs or "c" in kwargs:
        return QuadraticPolyData(**kwargs)
    elif "mesh" in kwargs:
        return Mesh(**kwargs)
    else:
        raise TypeError("Cannot automatically choose data type from the given arguments")


class Representation:
    r"""Abstract representation class, no implementations in here, just
    the right calls to the appropriate repA -> repB methods with default
    routing via Q representation.
    """

    def __init__(self, representation: Representation = None, **kwargs):
        if representation is not None and len(kwargs) > 0:
            raise TypeError("Either pass a single representation or keyword arguments, not both")
        if isinstance(representation, Representation):
            self = self.from_representation(
                representation
            )  # default sequence of transformations via Q
        elif representation is None:
            self.data = AutoData(**kwargs)
        else:
            raise TypeError(
                f"Cannot initialize representation from {representation.__class__.__qualname__}"
            )

    def __getattr__(self, name):
        # Intercept access to non-existent attributes of Representation like 'ket', 'cov' and route it to data.
        # This way we can access the data directly from the representation
        try:
            return getattr(self.data, name)
        except AttributeError:
            return self.data.__getattr__(name)

    def from_representation(self, representation):
        # If the representation is already of the right type, return it
        # Otherwise, call the appropriate from_xyz method
        if issubclass(representation.__class__, self.__class__):
            return representation
        return getattr(self, f"from_{representation.__class__.__qualname__.lower()}")(
            representation
        )

    # From leaves = from branch
    def from_charp(self, charP):
        return self.from_glauber(Glauber(charP))

    def from_charq(self, charQ):
        return self.from_husimi(Husimi(charQ))

    def from_charw(self, charW):
        return self.from_wigner(Wigner(charW))

    def from_wavefunctionp(self, wavefunctionP):
        return self.from_husimi(Husimi(wavefunctionP))

    # From branches = from trunk
    def from_stellar(self, stellar):
        return self.from_husimi(Husimi(stellar))

    def from_wavefunctionx(self, wavefunctionX):
        return self.from_husimi(Husimi(wavefunctionX))

    def from_wigner(self, wigner):
        return self.from_husimi(Husimi(wigner))

    def from_glauber(self, glauber):
        return self.from_husimi(Husimi(glauber))

    def _typecheck(self, other, operation: str):
        if self.__class__ != other.__class__:
            raise TypeError(
                f"Cannot perform {operation} between different representations ({self.__class__} and {other.__class__})"
            )

    # Operations between representations are defined in terms of operations between data
    def __add__(self, other):
        self._typecheck(other, "addition")
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


# TRUNK


# Now we implement the data-dependent methods.
# An advantage of collecting the code in this way is that instead of
# typechecking we try for the best case and catch the exception.
# It's also nice to keep the different versions of the transformation
# in different representations "side by side" (e.g. for learning).
class Husimi(Representation):
    def from_charq(self, charQ):
        try:
            return Husimi(
                GaussianData(math.inv(charQ.cov), charQ.mean, charQ.coeff)
            )  # TODO the mean is probably wrong
        except AttributeError:
            print("Fourier transform of charQ.ket/dm/mesh")

    def from_wigner(self, wigner):
        try:
            return Husimi(
                GaussianData(
                    math.qp_to_aadag(wigner.cov + math.eye_like(wigner.cov) / 2, axes=(-2, -1)),
                    math.qp_to_aadag(wigner.mean, axes=(-1,)),
                    wigner.coeff,
                )
            )
        except AttributeError:
            print("conv(wigner.dm, exp(|alpha|^2/2))")

    def from_glauber(self, glauber):
        try:
            return Husimi(GaussianData(glauber.cov + math.eye_like(glauber.cov), glauber.mean))
        except AttributeError:
            print("glauber.dm * exp(|alpha|^2)")

    def from_wavefunctionx(self, wavefunctionX):
        try:
            print("wavefunctionX.gaussian to Husimi...")
        except AttributeError:
            print("wavefunctionX.ket to Husimi...")

    def from_stellar(self, stellar):  # what if hilbert vector?
        try:
            math.Xmat(stellar.cov.shape[-1])
            Q = math.inv(math.eye_like(stellar.cov) - math.Xmat @ stellar.cov)
            return Husimi(GaussianData(Q, Q @ math.Xmat @ stellar.mean))
        except AttributeError:
            print("stellar.ket/dm to Husimi...")


# BRANCHES


class Wigner(Representation):
    def from_husimi(self, husimi):
        try:
            return Wigner(GaussianData(husimi.cov - math.eye_like(husimi.cov) / 2, husimi.mean))
        except AttributeError:
            print("husimi.ket * exp(-|alpha|^2/2)")
        except AttributeError:
            print("husimi.dm * exp(-|alpha|^2)")

    def from_charw(self, charw):
        try:
            return Wigner(GaussianData(math.inv(charw.cov), charw.mean))
        except AttributeError:
            print("Fourier transform of charw.ket")
        except AttributeError:
            print("Fourier transform of charw.dm")


class Glauber(Representation):
    def from_husimi(self, husimi):
        try:
            return Glauber(GaussianData(husimi.cov - math.eye_like(husimi.cov), husimi.mean))
        except AttributeError:
            print("husimi.dm * exp(-|alpha|^2)")

    def from_charp(self, charp):
        try:
            return Glauber(GaussianData(math.inv(charp.cov), charp.mean))
        except AttributeError:
            print("Fourier transform of charp.ket")
        except AttributeError:
            print("Fourier transform of charp.dm")


class WavefunctionX(Representation):
    def from_husimi(self, husimi):
        try:
            print("husimi.gaussian to wavefunctionX.gaussian")
        except AttributeError:
            print("husimi.ket to wavefunctionX.ket...")

    def from_wavefunctionp(self, wavefunctionP):
        try:
            return WavefunctionX(GaussianData(math.inv(wavefunctionP.cov), wavefunctionP.mean))
        except AttributeError:
            print("Fourier transform of wavefunctionP.ket")


class Stellar(Representation):
    # TODO: implement polynomial part (stellar rank > 0)
    def from_husimi(self, husimi):
        try:
            X = math.Xmat(husimi.cov.shape[-1])
            Qinv = math.inv(husimi.cov)
            A = X @ (math.eye_like(husimi.cov) - Qinv)
            return Stellar(
                GaussianData(A, X @ Qinv @ husimi.mean)
            )  # TODO: cov must be the inverse of A though
        except AttributeError:
            print("husimi.ket to stellar...")
        except AttributeError:
            print("husimi.dm to sellar...")

    @property
    def A(self):
        return math.inv(self.cov)

    @property
    def B(self):
        return math.inv(self.cov) @ self.mean

    @property
    def C(self):
        return self.coeff * math.exp(-self.mean.T @ math.inv(self.cov) @ self.mean)


# LEAVES


class CharP(Representation):
    def from_glauber(self, glauber):
        try:
            return CharP(GaussianData(math.inv(glauber.cov), glauber.mean))
        except AttributeError:
            print("Fourier transform of glauber.dm")

    def from_husimi(self, husimi):
        return self.from_glauber(Glauber(husimi))


class CharQ(Representation):
    def from_husimi(self, husimi):
        try:
            return CharQ(GaussianData(math.inv(husimi.cov), husimi.mean))
        except AttributeError:
            print("Fourier transform of husimi.ket")
        except AttributeError:
            print("Fourier transform of husimi.dm")


class CharW(Representation):
    def from_wigner(self, wigner):
        try:
            return CharW(GaussianData(math.inv(wigner.cov), wigner.mean))
        except AttributeError:
            print("Fourier transform of wigner.dm")

    def from_husimi(self, husimi):
        return self.from_wigner(Wigner(husimi))


class WavefunctionP(Representation):
    def from_wavefunctionx(self, wavefunctionX):
        try:
            return WavefunctionP(GaussianData(math.inv(wavefunctionX.cov), wavefunctionX.mean))
        except AttributeError:
            print("Fourier transform of wavefunctionX.ket")

    def from_husimi(self, husimi):
        self.from_wavefunctionx(WavefunctionX(husimi))


class Fock(Representation):
    def from_stellar(self, stellar):
        try:
            print("Recurrence relations")
        except AttributeError:
            print("stellar.ket to Fock...")
        except AttributeError:
            print("stellar.dm to Fock...")


# Data types
