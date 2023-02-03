import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

np.set_printoptions(suppress=True, linewidth=250)


class ComplexFunction1D:
    max_dom_points = 1000

    def __init__(self, x, y):
        self.interp_real = interp1d(x, np.real(y))
        self.interp_imag = interp1d(x, np.imag(y))

    def intersect_domains(self, other):
        x = np.union1d(self.domain, other.domain)
        # find intersection of x ranges
        x_min = max(self.domain.min(), other.domain.min())
        x_max = min(self.domain.max(), other.domain.max())
        # keep only the intersection
        x = x[(x >= x_min) & (x <= x_max)]
        return x

    @property
    def domain(self):
        return self.interp_real.x

    @property
    def values(self):
        return self.interp_real.y + 1j * self.interp_imag.y

    def plot(self):
        phase = np.angle(self.values)
        magnitude = np.abs(self.values)
        # convert phase to be between 0 and 1
        phase = (phase + np.pi) / (2 * np.pi)
        fig, ax = plt.subplots()
        ax.scatter(self.domain, magnitude, c=phase, cmap=cm.hsv, marker=".")
        ax.plot(self.domain, magnitude, color="black", linewidth=1)
        return ax

    def resample(self):
        """Resample the domain to have at most max_dom_points points.
        Sample more points where the derivative is large.
        """
        min_, max_ = self.domain.min(), self.domain.max()
        dom = np.linspace(min_, max_, self.max_dom_points)
        real_grad = np.gradient(self.interp_real(dom), dom)
        imag_grad = np.gradient(self.interp_imag(dom), dom)
        dy = np.abs(real_grad - 1j * imag_grad)  # Warning: unverified
        dy = interp1d(dom, dy)
        # we have a budget of max_dom_points,
        # we sample them between x.min() and x.max()
        # using dy as a probability distribution
        a = np.linspace(self.domain.min(), self.domain.max(), 10000)
        x = np.random.choice(
            a=a,
            size=self.max_dom_points,
            p=dy(a) / dy(a).sum(),
        )
        self.interp_real = interp1d(x, self.interp_real(x))
        self.interp_imag = interp1d(x, self.interp_imag(x))

    def __call__(self, x):
        return self.interp_real(x) + 1j * self.interp_imag(x)

    def __add__(self, other):
        if isinstance(other, self.__class__):
            x = self.intersect_domains(other)
            y = self(x) + other(x)
            f = ComplexFunction1D(x, y)
        elif isinstance(other, (int, float, complex)):
            x = self.domain
            y = self(x) + other
            f = ComplexFunction1D(x, y)
        else:
            raise TypeError(f"Cannot add {type(self)} and {type(other)}")
        if len(x) > self.max_dom_points:
            f.resample()
        return f

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if isinstance(other, ComplexFunction1D):
            x = self.intersect_domains(other)
            f = ComplexFunction1D(x, self(x) * other(x))
        elif isinstance(other, (int, float, complex)):
            x = self.domain
            f = ComplexFunction1D(x, self(x) * other)
        if len(x) > self.max_dom_points:
            f.resample()
        return f

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return ComplexFunction1D(self.domain, -self.interp_real.y - 1j * self.interp_imag.y)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __truediv__(self, other):
        if isinstance(other, ComplexFunction1D):
            x = self.intersect_domains(other)
            y = self(x) / other(x)
            f = ComplexFunction1D(x, y)
        elif isinstance(other, (int, float, complex)):
            x = self.domain
            y = self(x) / other
            f = ComplexFunction1D(x, y)
        if len(x) > self.max_dom_points:
            f.resample()
        return f

    def __rtruediv__(self, other):
        if isinstance(other, (int, float, complex)):
            x = self.domain
            y = other / self(x)
            return ComplexFunction1D(x, y)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == "__call__":
            x = inputs[0].domain
            y = ufunc(*(input(x) for input in inputs), **kwargs)
            return ComplexFunction1D(x, y)
        else:
            raise NotImplementedError
