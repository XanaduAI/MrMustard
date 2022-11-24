# abstract representation class
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Union, Protocol
from mrmustard.types import Array, Matrix, Vector, Number
import functools
from mrmustard.math import Math

math = Math()


class Data(ABC):
    r"""Supports algebraic operations for all kinds of data (gaussian, array, mesh, symbolic, etc).
    Algebra operations are intended between the Hilbert space vectors or between ensembles of Hilbert vectors.
    The implementation depends on both the representation and the kind of data. E.g. adding two gaussians in Wigner
    representation encoded as Gaussian data is implemented as stacking the two gaussian datas in the batch dimension.
    Adding the same two gaussians in Fock representation (e.g. with dm data) still means rho1+rho2,
    but in Fock representation we add the density matrices as arrays.

    For these operations (as either Hilbert vectors or convex combinations of Hilbert vectors):
    - Gaussian generally stacks cov, mean, coeff along a batch dimension.
    - Array (ket/dm) operates with the data arrays themselves.
    - Mesh (mesh) operates on the (x,f(x)) pairs with interpolation.
    - Symbolic (symbolic) operates on the symbolic expression.

    This class is abstract and Gaussian, Mesh, Symbolic inherit from it.
    There is no Ket/Dm class because they are just arrays.
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
        return self.__mul__(1 / other)  # is this numerically naughty?


class GaussianData(Data):
    def __init__(self, cov=None, mean=None, coeff=None):
        r"""
        Gaussian data: covariances, means, coefficients.
        Each of these has a batch dimension, and the batch dimension is the same for all of them.
        They are the parameters of a linear combination of Gaussians, which is Gaussian if there is only one contribution.
        Each contribution parametrizes the Gaussian function: `coeff * exp(-0.5*(x-mean)^T cov^-1 (x-mean))`.
        Args:
            cov (batch, dim, dim): covariance matrices
            mean  (batch, dim): means
            coeff (batch): coefficients
        """
        if isinstance(cov, QuadraticData):  # this captures the GaussianData(quaddata) syntax
            inv_A = math.inv(cov.A)
            self.cov = -2 * inv_A
            self.mean = inv_A @ cov.b
            self.coeff = cov.c * math.exp(0.5 * (self.mean.T @ cov.A @ self.mean))
        else:
            self.cov = math.atleast_1d(cov)
            self.mean = math.atleast_1d(mean)
            self.coeff = math.atleast_1d(coeff)

    def __add__(self, other: GaussianData):
        if np.allclose(self.cov, other.cov) and np.allclose(self.mean, other.mean):
            return GaussianData(
                self.cov, self.mean, self.coeff + other.coeff
            )  # TODO: what if coeff have a different batch dimension?
        return GaussianData(
            math.concat([self.cov, other.cov], axis=0),
            math.concat([self.mean, other.mean], axis=0),
            math.concat([self.coeff, other.coeff], axis=0),
        )

    def __mul__(self, other):
        if isinstance(
            other, Number
        ):  # TODO: this seems to deal only with the case of self and other being a single gaussian
            return GaussianData(self.cov, self.mean, self.coeff * other)
        elif isinstance(other, GaussianData):
            return GaussianData(
                math.inv(self.cov) + math.inv(other.cov),
                self.mean + other.mean,
                self.coeff * other.coeff,
            )  # TODO: invert decomposed covs instead
        else:
            raise TypeError(f"Cannot multiply GaussianData with {other.__class__.__qualname__}")

    def __and__(self, other: GaussianData):  # tensor product
        return GaussianData(
            [math.block_diag(self.cov, other.cov)],
            [math.concat([self.mean, other.mean], axis=0)],
            [self.coeff * other.coeff],
        )

    def __eq__(self, other):
        return (
            np.allclose(self.cov, other.cov)
            and np.allclose(self.mean, other.mean)
            and np.allclose(self.coeff, other.coeff)
        )


class QuadraticData(Data):
    def __init__(self, A=None, b=None, c=None):
        r"""
        Quadratic Gaussian data: quadratic coefficients, linear coefficients, constant.
        Each of these has a batch dimension, and the batch dimension is the same for all of them.
        They are the parameters of a Gaussian expressed as `c * exp(x^T A x + x^T b)`.
        Args:
            A (batch, dim, dim): quadratic coefficients
            b (batch, dim): linear coefficients
            c (batch): constant
        """
        if isinstance(A, GaussianData):
            self.A = -math.inv(A.cov)
            self.b = math.inv(A.cov) @ A.mean
            self.c = A.coeff * math.exp(-A.mean.T @ math.inv(A.cov) @ A.mean)
        else:
            self.A = math.atleast_1d(A)
            self.b = math.atleast_1d(b)
            self.c = math.atleast_1d(c)

    def __add__(self, other: QuadraticData):
        if np.allclose(self.A, other.A) and np.allclose(self.b, other.b):
            return QuadraticData(self.A, self.b, self.c + other.c)
        return QuadraticData(
            math.concat([self.A, other.A], axis=0),
            math.concat([self.b, other.b], axis=0),
            math.concat([self.c, other.c], axis=0),
        )

    def __mul__(self, other):
        if isinstance(
            other, Number
        ):  # TODO: this seems to deal only with the case of self and other being a single gaussian
            return QuadraticData(self.A, self.b, self.c * other)
        elif isinstance(other, QuadraticData):
            return QuadraticData(
                self.A + other.A, self.b + other.b, self.c * other.c
            )  # TODO: invert decomposed covs instead
        else:
            raise TypeError(f"Cannot multiply QuadraticData with {other.__class__.__qualname__}")

    def __and__(self, other: GaussianData):  # tensor product
        return GaussianData(
            [math.block_diag(self.A, other.A)],
            [math.concat([self.b, other.b], axis=0)],
            [self.c * other.c],
        )

    def __eq__(self, other):
        return (
            np.allclose(self.A, other.A)
            and np.allclose(self.b, other.b)
            and np.allclose(self.c, other.c)
        )


# array types are just arrays


class MeshData(Data):
    def __init__(self, mesh):
        self.mesh = mesh  # mesh is a dictionary of x:f(x) pairs

    def __add__(self, other):
        if isinstance(other, MeshData):
            # given x:f(x), if x:g(x) exists then x:f(x)+g(x)
            # given x:f(x), if y:g(y) with y=x does not exist then x:f(x)+gstar(x) where gstar(x) is the
            # interpolation of g(y) at x, performed using the following interpolation method:
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html
            # for scatteted data that does not follow a regular grid, we can use the following interpolation method:
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html
            # or the nearest neighbour interpolation method:
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.NearestNDInterpolator.html

            pass

    def __mul__(self, other):
        pass

    def __and__(self, other):
        pass


def AutoData(**kwargs):
    r"""Automatically choose the data type based on the arguments.
    If the arguments contain any combination of 'cov', 'mean', 'coeff' then it is GaussianData.
    If the arguments contain any combination of 'A', 'b', 'c' then it is QuadraticData.
    If the arguments contain 'mesh' then it is MeshData.

    """
    if "cov" in kwargs or "mean" in kwargs or "coeff" in kwargs:
        return GaussianData(**kwargs)
    elif "A" in kwargs or "b" in kwargs or "c" in kwargs:
        return QuadraticData(**kwargs)
    elif "mesh" in kwargs:
        return Mesh(**kwargs)
    else:
        raise TypeError(f"Cannot automatically choose data type from the given arguments")


class Representation:
    r"""Abstract representation class, no implementations in here, just
    the right calls to the appropriate repA -> repB methods with default
    routing via Q function.
    """

    def __init__(self, init: Union[Data, Representation] = None, **kwargs):
        if isinstance(init, Representation):
            self = self.from_representation(init)  # default sequence of transformations via Q
        elif isinstance(init, Data):
            self.data = init
        elif init is None:
            self.data = AutoData(**kwargs)
        else:
            raise TypeError(f"Cannot initialize representation with the given data")

    def __getattr__(self, name):
        # Intercept access to non-existent attribute like 'ket', 'cov', etc and route it to data.
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
            print(f"conv(wigner.dm, exp(|alpha|^2/2))")

    def from_glauber(self, glauber):
        try:
            return Husimi(GaussianData(glauber.cov + math.eye_like(glauber.cov), glauber.mean))
        except AttributeError:
            print(f"glauber.dm * exp(|alpha|^2)")

    def from_wavefunctionx(self, wavefunctionX):
        try:
            print(f"wavefunctionX.gaussian to Husimi...")
        except AttributeError:
            print(f"wavefunctionX.ket to Husimi...")

    def from_stellar(self, stellar):  # what if hilbert vector?
        try:
            X = math.Xmat(stellar.cov.shape[-1])
            Q = math.inv(math.eye_like(stellar.cov) - math.Xmat @ stellar.cov)
            return Husimi(GaussianData(Q, Q @ math.Xmat @ stellar.mean))
        except AttributeError:
            print(f"stellar.ket/dm to Husimi...")


# BRANCHES


class Wigner(Representation):
    def from_husimi(self, husimi):
        try:
            return Wigner(GaussianData(husimi.cov - math.eye_like(husimi.cov) / 2, husimi.mean))
        except AttributeError:
            print(f"husimi.ket * exp(-|alpha|^2/2)")
        except AttributeError:
            print(f"husimi.dm * exp(-|alpha|^2)")

    def from_charw(self, charw):
        try:
            return Wigner(GaussianData(math.inv(charw.cov), charw.mean))
        except AttributeError:
            print(f"Fourier transform of charw.ket")
        except AttributeError:
            print(f"Fourier transform of charw.dm")


class Glauber(Representation):
    def from_husimi(self, husimi):
        try:
            return Glauber(GaussianData(husimi.cov - math.eye_like(husimi.cov), husimi.mean))
        except AttributeError:
            print(f"husimi.dm * exp(-|alpha|^2)")

    def from_charp(self, charp):
        try:
            return Glauber(GaussianData(math.inv(charp.cov), charp.mean))
        except AttributeError:
            print(f"Fourier transform of charp.ket")
        except AttributeError:
            print(f"Fourier transform of charp.dm")


class WavefunctionX(Representation):
    def from_husimi(self, husimi):
        try:
            print(f"husimi.gaussian to wavefunctionX.gaussian")
        except AttributeError:
            print(f"husimi.ket to wavefunctionX.ket...")

    def from_wavefunctionp(self, wavefunctionP):
        try:
            return WavefunctionX(GaussianData(math.inv(wavefunctionP.cov), wavefunctionP.mean))
        except AttributeError:
            print(f"Fourier transform of wavefunctionP.ket")


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
            print(f"husimi.ket to stellar...")
        except AttributeError:
            print(f"husimi.dm to sellar...")

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
            print(f"Fourier transform of glauber.dm")

    def from_husimi(self, husimi):
        return self.from_glauber(Glauber(husimi))


class CharQ(Representation):
    def from_husimi(self, husimi):
        try:
            return CharQ(GaussianData(math.inv(husimi.cov), husimi.mean))
        except AttributeError:
            print(f"Fourier transform of husimi.ket")
        except AttributeError:
            print(f"Fourier transform of husimi.dm")


class CharW(Representation):
    def from_wigner(self, wigner):
        try:
            return CharW(GaussianData(math.inv(wigner.cov), wigner.mean))
        except AttributeError:
            print(f"Fourier transform of wigner.dm")

    def from_husimi(self, husimi):
        return self.from_wigner(Wigner(husimi))


class WavefunctionP(Representation):
    def from_wavefunctionx(self, wavefunctionX):
        try:
            return WavefunctionP(GaussianData(math.inv(wavefunctionX.cov), wavefunctionX.mean))
        except AttributeError:
            print(f"Fourier transform of wavefunctionX.ket")

    def from_husimi(self, husimi):
        self.from_wavefunctionx(WavefunctionX(husimi))


class Fock(Representation):
    def from_stellar(self, stellar):
        try:
            print(f"Recurrence relations")
        except AttributeError:
            print(f"stellar.ket to Fock...")
        except AttributeError:
            print(f"stellar.dm to Fock...")


# Data types
