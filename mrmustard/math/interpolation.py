from __future__ import annotations

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

from mrmustard.typing import ComplexVector, RealVector

np.set_printoptions(suppress=True, linewidth=250)


class ComplexFunction1D:
    r"""A complex function of a real variable."""
    max_dom_points = 1000

    def __init__(self, x: RealVector, y: ComplexVector):
        r"""Initialize the function with a set of points.
        Supports interpolation, plotting, and resampling.
        Args:
            x (Vector): the domain of the function
            y (Vector): the values of the function
        """
        self.interp_real = interp1d(x, np.real(y))
        self.interp_imag = interp1d(x, np.imag(y))

    def intersect_domains(self, other: ComplexFunction1D) -> RealVector:
        x = np.union1d(self.domain, other.domain)
        # find intersection of x ranges
        x_min = max(self.domain.min(), other.domain.min())
        x_max = min(self.domain.max(), other.domain.max())
        # keep only the intersection
        x = x[(x >= x_min) & (x <= x_max)]
        return x

    @property
    def domain(self) -> RealVector:
        return self.interp_real.x

    @property
    def values(self) -> ComplexVector:
        return self.interp_real.y + 1j * self.interp_imag.y

    # def plot(self):
    #     phase = np.angle(self.values)
    #     magnitude = np.abs(self.values)
    #     # convert phase to be between 0 and 1
    #     phase = (phase + np.pi) / (2 * np.pi)
    #     fig, ax = plt.subplots()
    #     ax.scatter(self.domain, magnitude, c=phase, cmap=cm.hsv, marker=".")
    #     ax.plot(self.domain, magnitude, color="black", linewidth=1)
    #     return ax
    # def plot(self):
    #     phase = (np.angle(self.values) + np.pi) / (2 * np.pi)
    #     magnitude = np.abs(self.values)
    #     fig, ax = plt.subplots()
    #     # Use fill_between() to fill the area under the curve with HUE based on phase angle
    #     ax.fill_between(
    #         self.domain,
    #         0,
    #         magnitude,
    #         where=magnitude >= 0,
    #         interpolate=True,
    #         color=cm.hsv(phase),
    #         alpha=1.0,
    #     )
    #     # Plot the curve in black color with linewidth=1
    #     ax.plot(self.domain, magnitude, color="black", linewidth=1)
    #     return ax
    def plot(self):
        phase = np.angle(self.values)
        magnitude = np.abs(self.values)
        # convert phase to be between 0 and 1
        phase = (phase + np.pi) / (2 * np.pi)
        fig, ax = plt.subplots()
        # Use fill_between() to fill the area under the curve with a varying color
        for i in range(len(self.domain) - 1):
            x0, x1 = self.domain[i], self.domain[i + 1]
            y0, y1 = magnitude[i], magnitude[i + 1]
            # Compute the average phase angle in the x interval [x0, x1]
            avg_phase = np.mean(phase[(self.domain >= x0) & (self.domain <= x1)])
            # Fill the area under the curve with a color based on the average phase angle
            ax.fill_between(
                [x0, x1],
                0,
                [y0, y1],
                where=[y0 >= 0, y1 >= 0],
                interpolate=True,
                color=cm.hsv(avg_phase),
                alpha=0.5,
            )
        # Plot the curve in black color with linewidth=1
        ax.plot(self.domain, magnitude, color="black", linewidth=1)
        return ax

    def resample(self) -> None:
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

    def __call__(self, x) -> complex:
        return self.interp_real(x) + 1j * self.interp_imag(x)

    def __add__(self, other: ComplexFunction1D | int | float | complex) -> ComplexFunction1D:
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

    def __radd__(self, other: int | float | complex) -> ComplexFunction1D:
        return self + other

    def __mul__(self, other: ComplexFunction1D | int | float | complex) -> ComplexFunction1D:
        if isinstance(other, ComplexFunction1D):
            x = self.intersect_domains(other)
            f = ComplexFunction1D(x, self(x) * other(x))
        elif isinstance(other, (int, float, complex)):
            x = self.domain
            f = ComplexFunction1D(x, self(x) * other)
        if len(x) > self.max_dom_points:
            f.resample()
        return f

    def __rmul__(self, other: int | float | complex) -> ComplexFunction1D:
        return self * other

    def __neg__(self) -> ComplexFunction1D:
        return ComplexFunction1D(self.domain, -self.interp_real.y - 1j * self.interp_imag.y)

    def __sub__(self, other: ComplexFunction1D | int | float | complex) -> ComplexFunction1D:
        return self + (-other)

    def __rsub__(self, other: int | float | complex) -> ComplexFunction1D:
        return other + (-self)

    def __truediv__(self, other: ComplexFunction1D | int | float | complex) -> ComplexFunction1D:
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

    def __rtruediv__(self, other: int | float | complex) -> ComplexFunction1D:
        if isinstance(other, (int, float, complex)):
            x = self.domain
            y = other / self(x)
            return ComplexFunction1D(x, y)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs) -> ComplexFunction1D:
        if method == "__call__":
            x = inputs[0].domain
            y = ufunc(*(input(x) for input in inputs), **kwargs)
            return ComplexFunction1D(x, y)
        else:
            raise NotImplementedError
