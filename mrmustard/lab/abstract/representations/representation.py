# abstract representation class
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Union, Protocol
from mrmustard.types import Array, Matrix, Vector, Number


@dataclass
class Data:
    symbolic: Optional[str] = None
    gaussian: Optional[Tuple[Array]] = None # cov, mean, coeff
    array: Optional[Array] = None # ket or dm
    mesh: Optional[(Matrix, Vector)] = None # {x, f(x)}
    is_hilbert_vector: bool = False


class Representation:
    """Abstract representation class, no implementations in here, just
    the right calls to the appropriate repA -> repB methods with default
    routing via Q function.
    """

    def __init__(self, data: Union[Data, Representation]):
        if isinstance(data, Representation):
            return self.from_representation(data) # this calls the generic repA->repB transformation
        else:
            self.symbolic = data.symbolic
            self.pure_gaussian = Gaussian(data.pure_cov, data.mean, data.coeff)
            self.mixed_gaussian = Gaussian(data.mixed_cov, data.mean, data.coeff)
            self.ket = data.ket
            self.dm = data.dm
            self.mesh = data.mesh
            self.is_hilbert_vector = data.is_hilbert_vector

    @property
    def gaussian(self):
        return self.pure_gaussian or self.mixed_gaussian

    @property
    def array(self):
        return self.ket or self.dm
    
    @property
    def preferred_data(self):
        return self.symbolic or self.gaussian or self.array or self.mesh

    @property
    def cov(self):
        return self.gaussian.cov

    @property
    def mean(self):
        return self.gaussian.mean

    @property
    def coeff(self):
        return self.gaussian.coeff

    def from_representation(self, representation): # avoids having to implement all the combinations
        if isinstance(representation, self.__class__):
            return self
        return getattr(self, f'from_{representation.__class__.__qualname__.lower()}')(representation)

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
            raise TypeError(f"Cannot perform {operation} on representations of different types ({self.__class__} and {other.__class__})")

    def __add__(self, other):
        self._typecheck(other, 'addition')
        return self.__class__(self.preferred_data + other.preferred_data)
    
    def __sub__(self, other):
        self._typecheck(other, 'subtraction')
        return self.__class__(self.preferred_data - other.preferred_data)

    def __rmul__(self, other):
        return self * other

    def __mul__(self, other):
        if isinstance(other, Number):
            return self.__class__(self.preferred_data * other)

    def __truediv__(self, other):
        if isinstance(other, Number):
            return self.__class__(self.preferred_data / other)

# TRUNK

class Husimi(Representation):
    def from_charq(self, charQ):
        try:
            return Husimi(Gaussian(math.inv(charQ.cov), charQ.mean, charQ.coeff)) # TODO mean is probably wrong
        except AttributeError:
            print('Fourier transform of charQ.ket/dm/mesh')

    def from_wigner(self, wigner):
        try:
            return Husimi(Gaussian(math.qp_to_aadag(wigner.cov + math.eye_like(wigner.cov)/2, axes=(-2,-1)), math.qp_to_aadag(wigner.mean, axes=(-1,)), wigner.coeff))
        except AttributeError:
            print(f'conv(wigner.dm, exp(|alpha|^2/2))')

    def from_glauber(self, glauber):
        try:
            return Husimi(Gaussian(glauber.cov + math.eye_like(glauber.cov), glauber.mean))
        except AttributeError:
            print(f'glauber.dm * exp(|alpha|^2)')


    def from_wavefunctionx(self, wavefunctionX):
        try:
            print(f'wavefunctionX.gaussian to Husimi...')
        except AttributeError:
            print(f'wavefunctionX.ket to Husimi...')

    def from_stellar(self, stellar): # what if hilbert vector?
        try:
            X = math.Xmat(stellar.cov.shape[-1])
            Q = math.inv(math.eye_like(stellar.cov)-math.Xmat @ stellar.cov)
            return Husimi(Gaussian(Q, Q @ math.Xmat @ stellar.mean))
        except AttributeError:
            print(f'stellar.ket/dm to Husimi...')


# BRANCHES

class Wigner(Representation):
    def from_husimi(self, husimi):
        try:
            return Wigner(Gaussian(husimi.cov - math.eye_like(husimi.cov)/2, husimi.mean))
        except AttributeError:
            print(f'husimi.ket * exp(-|alpha|^2/2)')
        except AttributeError:
            print(f'husimi.dm * exp(-|alpha|^2)')

    def from_charw(self, charw):
        try:
            return Wigner(Gaussian(math.inv(charw.cov), charw.mean))
        except AttributeError:
            print(f'Fourier transform of charw.ket')
        except AttributeError:
            print(f'Fourier transform of charw.dm')

class Glauber(Representation):
    def from_husimi(self, husimi):
        try:
            return Glauber(Gaussian(husimi.cov - math.eye_like(husimi.cov), husimi.mean))
        except AttributeError:
            print(f'husimi.dm * exp(-|alpha|^2)')

    def from_charp(self, charp):
        try:
            return Glauber(Gaussian(math.inv(charp.cov), charp.mean))
        except AttributeError:
            print(f'Fourier transform of charp.ket')
        except AttributeError:
            print(f'Fourier transform of charp.dm')

class WavefunctionX(Representation):
    def from_husimi(self, husimi):
        try:
            print(f'husimi.gaussian to wavefunctionX.gaussian')
        except AttributeError:
            print(f'husimi.ket to wavefunctionX.ket...')

    def from_wavefunctionp(self, wavefunctionP):
        try:
            return WavefunctionX(Gaussian(math.inv(wavefunctionP.cov), wavefunctionP.mean))
        except AttributeError:
            print(f'Fourier transform of wavefunctionP.ket')

class Stellar(Representation):
    # TODO: implement polynomial part
    def from_husimi(self, husimi):
        try:
            X = math.Xmat(husimi.cov.shape[-1])
            Qinv = math.inv(husimi.cov)
            A = X @ (math.eye_like(husimi.cov) - Qinv)
            return Stellar(Gaussian(A, X @ Qinv @ husimi.mean))
        except AttributeError:
            print(f'husimi.ket to stellar...')
        except AttributeError:
            print(f'husimi.dm to sellar...')

# LEAVES

class CharP(Representation):
    def from_glauber(self, glauber):
        try:
            return CharP(Gaussian(math.inv(glauber.cov), glauber.mean))
        except AttributeError:
            print(f'Fourier transform of glauber.dm')

    def from_husimi(self, husimi):
        return self.from_glauber(Glauber(husimi))

class CharQ(Representation):
    def from_husimi(self, husimi):
        try:
            return CharQ(Gaussian(math.inv(husimi.cov), husimi.mean))
        except AttributeError:
            print(f'Fourier transform of husimi.ket')
        except AttributeError:
            print(f'Fourier transform of husimi.dm')

class CharW(Representation):
    def from_wigner(self, wigner):
        try:
            return CharW(Gaussian(math.inv(wigner.cov), wigner.mean))
        except AttributeError:
            print(f'Fourier transform of wigner.dm')

    def from_husimi(self, husimi):
        return self.from_wigner(Wigner(husimi))

class WavefunctionP(Representation):
    def from_wavefunctionx(self, wavefunctionX):
        try:
            return WavefunctionP(Gaussian(math.inv(wavefunctionX.cov), wavefunctionX.mean))
        except AttributeError:
            print(f'Fourier transform of wavefunctionX.ket')

    def from_husimi(self, husimi):
        self.from_wavefunctionx(WavefunctionX(husimi))

class Fock(Representation):
    def from_stellar(self, stellar):
        try:
            print(f'Recurrence relations')
        except AttributeError:
            print(f'stellar.ket to Fock...')
        except AttributeError:
            print(f'stellar.dm to Fock...')


# Data types

class Gaussian:
    def __init__(cov, mean, coeff):
        self.cov = math.atleast_3d(cov)
        self.mean = math.atleast_2d(mean)
        self.coeff = coeff or math.ones(1, dtype=mean.dtype)
        assert len(self.cov) == len(self.mean) == len(self.coeff)
        
    def __add__(self, other): # W1(x) + W2(x)
        if type(self) != type(other):
            return TypeError(f'Cannot add {self.__class__.__qualname__} and {other.__class__.__qualname__}')
        cov = math.concat(self.cov, other.cov, axis=0)
        means = math.concat(self.means, other.means, axis=0)
        coeffs = math.concat(self.coeffs, other.coeffs, axis=0)
        return Gaussian(cov=cov, means=means, coeff=coeffs)

    def __sub__(self, other):
        if type(self) != type(other):
            return TypeError(f'Cannot subtract {self.__class__.__qualname__} and {other.__class__.__qualname__}')
        return self + (-other)

    def __rmul__(self, other: Number):
        return self * other

    def __mul__(self, other: Number):
        return Gaussian(cov=self.cov, means=self.means, coeff=self.coeffs * other)

    def __neg__(self):
        return self * -1

    def outer(self, other: Gaussian):
        covs = math.astensor([math.block_diag([math.conj(cov_other),cov_self]) for cov_other in other.cov for cov_self in self.cov])
        means = math.astensor([math.concat([math.conj(mean_other), mean_self], axis=0) for mean_other in other.mean for mean_self in self.mean])
        coeffs = math.astensor([coeff_other * coeff_self for coeff_other in other.coeff for coeff_self in self.coeff])
        return Gaussian(cov=covs, mean=means, coeff=coeffs)
        




class Linear:
    # hilbert space algebra
    pass

class Convex(Linear):
    # convex algebra
    pass



class Ket:
    def __init__(self, array):
        self.array = array
    
    def __add__(self, other):
        return Ket(self.array + other.array)

    def __sub__(self, other):
        return Ket(self.array - other.array)

    def __mul__(self, other):
        return Ket(self.array * other.array)

    def __matmul__(self, other):
        return Ket(self.array @ other.array)
    
    def __truediv__(self, other):
        return Ket(self.array / other.array)



class DensityMatrix:
    def __init__(self, array):
        self.array = array

    def __add__(self, other):
        try:
            DensityMatrix(self.array + other.array)
        except AttributeError:
            try:
               return DensityMatrix(other + self.array)
            except (TypeError, ValueError):
                return DensityMatrix(self.array + other)

    def __sub__(self, other):
        try:
            DensityMatrix(self.array + other)
        except (TypeError, ValueError):
            return DensityMatrix(self.array - other.array)

    def __mul__(self, other):
        try:
            DensityMatrix(self.array * other)
        except (TypeError, ValueError):
            return DensityMatrix(self.array * other.array)

    def __matmul__(self, other):
        try:
            DensityMatrix(self.array @ other)
        except (TypeError, ValueError):
            return DensityMatrix(self.array @ other.array)
    