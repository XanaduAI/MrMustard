import numpy as np
from scipy.interpolate import interp1d

np.set_printoptions(suppress=True, linewidth=250)


class ComplexFunction1D:
    def __init__(self, x, y):
        self.interp_real = interp1d(x, np.real(y))
        self.interp_imag = interp1d(x, np.imag(y))

    def intersect_ranges(self, other):
        x = np.union1d(self.interp_real.x, other.interp_real.x)
        # find intersection of x ranges
        x_min = max(self.interp_real.x.min(), other.interp_real.x.min())
        x_max = min(self.interp_real.x.max(), other.interp_real.x.max())
        # keep only the intersection
        x = x[(x >= x_min) & (x <= x_max)]
        return x

    def __call__(self, x):
        return self.interp_real(x) + 1j * self.interp_imag(x)

    def __add__(self, other):
        if isinstance(other, ComplexFunction1D):
            x = self.intersect_ranges(other)
            y = self(x) + other(x)
            return ComplexFunction1D(x, y)
        elif isinstance(other, (int, float, complex)):
            x = self.interp_real.x
            y = self(x) + other
            return ComplexFunction1D(x, y)

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if isinstance(other, ComplexFunction1D):
            x = self.intersect_ranges(other)
            return ComplexFunction1D(x, self(x) * other(x))
        elif isinstance(other, (int, float, complex)):
            x = self.interp_real.x
            return ComplexFunction1D(x, self(x) * other)

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return ComplexFunction1D(self.interp_real.x, -self.interp_real.y - 1j * self.interp_imag.y)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        if isinstance(other, ComplexFunction1D):
            x = self.intersect_ranges(other)
            y = self(x) / other(x)
            return ComplexFunction1D(x, y)
        elif isinstance(other, (int, float, complex)):
            x = self.interp_real.x
            y = self(x) / other
            return ComplexFunction1D(x, y)

    def __rtruediv__(self, other):
        if isinstance(other, (int, float, complex)):
            x = self.interp_real.x
            y = other / self(x)
            return ComplexFunction1D(x, y)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == "__call__":
            x = inputs[0].interp_real.x
            y = ufunc(*(input(x) for input in inputs), **kwargs)
            return ComplexFunction1D(x, y)
        else:
            raise NotImplementedError
