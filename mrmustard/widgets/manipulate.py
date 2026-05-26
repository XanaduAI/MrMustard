# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Interactive parameter exploration widget for Jupyter notebooks.

Provides a ``manipulate`` function similar to Mathematica's ``Manipulate``,
allowing real-time exploration of plots as parameters vary via sliders.

Supports both real-valued sliders (plain tuples) and complex-valued sliders
(``ComplexSlider``) that generate paired controls in either Cartesian
(Re/Im) or polar (r/θ) coordinates.

The figure canvas is updated in-place via ``draw_idle`` for flicker-free
rendering.  Requires the ipympl backend (``%matplotlib widget``).

Usage pattern: create ``fig, ax`` with ``plt.subplots`` (or a custom
``GridSpec`` layout), then pass them to ``manipulate`` along with a
drawing function and slider specs.  The axes are cleared automatically
before each redraw, and the drawing function receives ``ax`` as its
first argument so it can use matplotlib's OOP interface (``ax.plot``,
``ax.set_title``, etc.) instead of the pyplot state machine.
"""

from __future__ import annotations

import cmath
import inspect
import math
from collections.abc import Callable

import ipywidgets
import matplotlib as mpl
from IPython.display import display
from matplotlib.axes import Axes
from matplotlib.figure import Figure

__all__ = ["ComplexSlider", "manipulate"]


# ---------------------------------------------------------------------------
# Slider specification helpers
# ---------------------------------------------------------------------------


def _parse_slider_spec(spec: tuple[float, ...]) -> tuple[float, float, float, float]:
    """Parse a slider specification tuple into (min, max, step, default).

    Accepted formats:
        ``(min, max)``                  step = (max - min) / 100, default = midpoint
        ``(min, max, step)``            default = midpoint
        ``(min, max, step, default)``   all explicit

    Args:
        spec: Tuple describing the slider range.

    Returns:
        Tuple of (min_val, max_val, step, default).

    Raises:
        ValueError: If *spec* has fewer than 2 or more than 4 elements.
    """
    if len(spec) == 2:
        min_val, max_val = spec
        step = (max_val - min_val) / 100
        default = (min_val + max_val) / 2
    elif len(spec) == 3:
        min_val, max_val, step = spec
        default = (min_val + max_val) / 2
    elif len(spec) == 4:
        min_val, max_val, step, default = spec
    else:
        raise ValueError(
            f"Slider spec must be (min, max), (min, max, step), "
            f"or (min, max, step, default). Got tuple of length {len(spec)}."
        )
    return float(min_val), float(max_val), float(step), float(default)


class ComplexSlider:
    """Specification for a complex-valued parameter with paired sliders.

    Creates two sliders that combine into a single complex value passed to
    the plotting function.

    In **Cartesian mode** (default, ``polar=False``) the sliders control the
    real and imaginary parts.

    In **polar mode** (``polar=True``) the sliders control the magnitude
    *r* and phase angle *θ*.  The simple constructor maps *min_val* /
    *max_val* to the *r* range ``[0, max_val]`` and sets the *θ* range to
    ``[-π, π]``.

    Both sliders share the same range in Cartesian mode.  For full control
    over each component's range, use :meth:`from_parts`.

    Args:
        min_val: Minimum slider value.  In Cartesian mode this applies to
            both Re and Im.  In polar mode only *max_val* is used for the
            *r* range (the minimum is always 0).
        max_val: Maximum slider value (applies to both components in
            Cartesian mode, or to *r* in polar mode).
        step: Slider step size.  ``None`` means ``(max - min) / 100``.
        default: Default value (may be complex).  Decomposed into the
            appropriate components (Re/Im or r/θ) for the initial slider
            positions.
        polar: If ``True``, use r/θ (polar) coordinates instead of Re/Im.

    Examples::

        # Cartesian (Re/Im)
        ComplexSlider(-0.5, 0.5, 0.01, default=0.1 + 0.2j)

        # Polar (r/θ)
        ComplexSlider(-0.5, 0.5, 0.01, default=0.3, polar=True)
    """

    def __init__(
        self,
        min_val: float,
        max_val: float,
        step: float | None = None,
        default: complex = 0 + 0j,
        polar: bool = False,
    ) -> None:
        default = complex(default)
        self.polar = polar

        if polar:
            if step is None:
                step = max_val / 100
            r_default, arg_default = cmath.polar(default)
            self.first_spec = (0.0, float(max_val), float(step), r_default)
            self.second_spec = (-math.pi, math.pi, float(step), arg_default)
            self.first_label = "r"
            self.second_label = "θ"
        else:
            if step is None:
                step = (max_val - min_val) / 100
            self.first_spec = (float(min_val), float(max_val), float(step), default.real)
            self.second_spec = (float(min_val), float(max_val), float(step), default.imag)
            self.first_label = "Re"
            self.second_label = "Im"

    @classmethod
    def from_parts(
        cls,
        first_spec: tuple[float, ...],
        second_spec: tuple[float, ...],
        polar: bool = False,
    ) -> ComplexSlider:
        """Create a ``ComplexSlider`` with different ranges for each component.

        Each argument is a tuple in the same format accepted by
        :func:`manipulate` for real sliders: ``(min, max)``,
        ``(min, max, step)``, or ``(min, max, step, default)``.

        Args:
            first_spec: Range for Re (Cartesian) or r (polar).
            second_spec: Range for Im (Cartesian) or θ (polar).
            polar: If ``True``, interpret as r/θ instead of Re/Im.
        """
        obj = cls.__new__(cls)
        obj.polar = polar
        obj.first_spec = _parse_slider_spec(first_spec)
        obj.second_spec = _parse_slider_spec(second_spec)
        if polar:
            obj.first_label = "r"
            obj.second_label = "θ"
        else:
            obj.first_label = "Re"
            obj.second_label = "Im"
        return obj


# ---------------------------------------------------------------------------
# Internal: paired slider widget
# ---------------------------------------------------------------------------


def _format_complex(z: complex) -> str:
    """Format a complex number as ``re + im i`` to 4 decimal places."""
    re, im = z.real, z.imag
    if im >= 0:
        return f"{re:.4f} + {im:.4f}i"
    return f"{re:.4f} - {-im:.4f}i"


class _ComplexWidgetPair:
    """Two ``FloatSlider`` widgets that combine into a single complex value.

    Encapsulates the Cartesian-vs-polar conversion so that downstream code
    only needs to read ``.value``.  Also owns a live HTML label that
    displays the parameter name and current complex value.
    """

    def __init__(
        self,
        first: ipywidgets.FloatSlider,
        second: ipywidgets.FloatSlider,
        polar: bool,
        name: str,
    ) -> None:
        self.first = first
        self.second = second
        self.polar = polar
        self._name = name
        self.label = ipywidgets.HTML()
        self._refresh_label()
        self.first.observe(self._refresh_label, names="value")
        self.second.observe(self._refresh_label, names="value")

    def _refresh_label(self, *_) -> None:
        """Update the HTML label with the current complex value."""
        self.label.value = (
            f"<b style='font-size:13px; margin-left:4px'>{self._name}</b>"
            f"<span style='font-size:12px; color:#666; font-family:monospace'>"
            f" = {_format_complex(self.value)}</span>"
        )

    @property
    def value(self) -> complex:
        """Current complex value assembled from slider positions."""
        if self.polar:
            return cmath.rect(self.first.value, self.second.value)
        return complex(self.first.value, self.second.value)

    def observe(self, callback: Callable, *, names: str) -> None:
        """Register *callback* on the ``value`` trait of both sliders."""
        self.first.observe(callback, names=names)
        self.second.observe(callback, names=names)


# ---------------------------------------------------------------------------
# Widget builder helpers
# ---------------------------------------------------------------------------

_SLIDER_LAYOUT = ipywidgets.Layout(width="95%")
_SLIDER_STYLE = {"description_width": "initial"}


def _make_float_slider(
    description: str,
    spec: tuple[float, float, float, float],
    continuous_update: bool,
) -> ipywidgets.FloatSlider:
    """Build a single ``FloatSlider`` from a parsed (min, max, step, default) spec."""
    min_val, max_val, step, default = spec
    return ipywidgets.FloatSlider(
        value=default,
        min=min_val,
        max=max_val,
        step=step,
        description=description,
        continuous_update=continuous_update,
        style=_SLIDER_STYLE,
        layout=_SLIDER_LAYOUT,
        readout_format=".4f",
    )


def _build_slider_widgets(
    slider_specs: dict[str, tuple | ComplexSlider],
    continuous_update: bool,
) -> tuple[
    dict[str, ipywidgets.FloatSlider | _ComplexWidgetPair],
    list[ipywidgets.Widget],
]:
    """Build slider widgets from specifications.

    Returns:
        widget_map: mapping from parameter name to widget or widget pair.
        display_rows: ordered list of widgets for vertical layout.
    """
    widget_map: dict[str, ipywidgets.FloatSlider | _ComplexWidgetPair] = {}
    display_rows: list[ipywidgets.Widget] = []

    for name, spec in slider_specs.items():
        if isinstance(spec, ComplexSlider):
            first_slider = _make_float_slider(spec.first_label, spec.first_spec, continuous_update)
            second_slider = _make_float_slider(
                spec.second_label, spec.second_spec, continuous_update
            )
            pair = _ComplexWidgetPair(first_slider, second_slider, spec.polar, name)
            widget_map[name] = pair
            display_rows.append(
                ipywidgets.VBox(
                    [pair.label, first_slider, second_slider],
                    layout=ipywidgets.Layout(margin="0 0 6px 0"),
                )
            )
        else:
            parsed = _parse_slider_spec(spec)
            slider = _make_float_slider(name, parsed, continuous_update)
            widget_map[name] = slider
            display_rows.append(slider)

    return widget_map, display_rows


def _gather_kwargs(
    widget_map: dict[str, ipywidgets.FloatSlider | _ComplexWidgetPair],
) -> dict:
    """Collect current slider values into keyword arguments for the user function."""
    return {name: widget.value for name, widget in widget_map.items()}


def _observe_all(
    widget_map: dict[str, ipywidgets.FloatSlider | _ComplexWidgetPair],
    callback: Callable,
) -> None:
    """Wire up *callback* on the ``value`` trait of every slider."""
    for widget in widget_map.values():
        widget.observe(callback, names="value")


# ---------------------------------------------------------------------------
# Signature validation
# ---------------------------------------------------------------------------


def _validate_fn_accepts_sliders(
    fn: Callable,
    slider_specs: dict[str, tuple | ComplexSlider],
) -> None:
    """Raise ``TypeError`` early if *fn* cannot accept the slider keyword arguments.

    Skips validation when the signature cannot be introspected (e.g. some
    built-in or C-extension callables).
    """
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return  # cannot introspect -- skip validation

    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if has_var_keyword:
        return  # fn accepts **kwargs, any name is valid

    accepted = {
        name
        for name, p in sig.parameters.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    unexpected = set(slider_specs) - accepted
    if unexpected:
        sorted_unexpected = sorted(unexpected)
        sorted_accepted = sorted(accepted)
        raise TypeError(
            f"Slider name(s) {sorted_unexpected} do not match any parameter of "
            f"{getattr(fn, '__name__', repr(fn))!r}. "
            f"Expected keyword parameters: {sorted_accepted}"
        )


# ---------------------------------------------------------------------------
# Backend validation
# ---------------------------------------------------------------------------


def _require_ipympl_backend() -> None:
    """Raise ``RuntimeError`` if the active matplotlib backend is not ipympl.

    ``manipulate`` requires ipympl (``%matplotlib widget``) for flicker-free
    in-place canvas updates.
    """
    backend = mpl.get_backend()
    if "ipympl" not in backend.lower() and backend.lower() != "widget":
        raise RuntimeError(
            "manipulate requires the ipympl matplotlib backend for "
            "interactive figure updates.\n"
            "Activate it by running\n"
            "  %matplotlib widget\n"
            "in a notebook cell before calling manipulate.\n"
            f"Current backend: {backend!r}"
        )


def _dequeue_ipympl_figure(fig: Figure) -> None:
    """Remove *fig* from ipympl's pending-display queue.

    ipympl registers a ``post_execute`` hook (``flush_figures``) that calls
    ``display(canvas)`` for every figure created in interactive mode.  When
    ``manipulate`` has already placed the canvas inside its own container
    widget, this post-cell flush would display the canvas a second time.

    Removing *fig* from the queue prevents the duplicate.  Accesses ipympl
    internals; fails silently if the API has changed.
    """
    try:
        # ipympl's _Backend_ipympl._to_show is the list inspected by
        # flush_figures at cell end.
        from ipympl.backend_nbagg import _Backend_ipympl  # noqa: PLC0415

        _Backend_ipympl._to_show = [f for f in _Backend_ipympl._to_show if f is not fig]
    except (ImportError, AttributeError):
        pass


# ---------------------------------------------------------------------------
# Axes clearing
# ---------------------------------------------------------------------------


def _clear_axes(ax: Axes | list[Axes]) -> None:
    """Clear *ax*, whether a single ``Axes`` or a collection of them.

    Handles single ``Axes`` objects, lists, tuples, and numpy arrays
    (via the ``.flat`` iterator).
    """
    if isinstance(ax, Axes):
        ax.clear()
        return
    for a in getattr(ax, "flat", ax):
        a.clear()


# ---------------------------------------------------------------------------
# Main manipulate function
# ---------------------------------------------------------------------------


def _figure_max_width(fig: Figure, padding: int = 24) -> str:
    """Compute a CSS ``max-width`` that matches a matplotlib figure's pixel width.

    Adds *padding* extra pixels (split equally left/right) so that sliders
    sit just slightly wider than the plot canvas.
    """
    return f"{int(fig.get_figwidth() * fig.dpi) + padding}px"


def manipulate(
    fn: Callable,
    fig: Figure,
    ax: Axes | list[Axes],
    *,
    continuous_update: bool = True,
    max_width: str | None = None,
    **slider_specs: tuple[float, ...] | ComplexSlider,
) -> None:  # pragma: no cover
    """Interactive parameter exploration with sliders.

    Creates sliders for each named parameter and re-runs *fn* whenever a
    slider value changes, updating the matplotlib figure in-place for
    flicker-free rendering.  Requires the ipympl backend
    (``%matplotlib widget``).

    This is analogous to Mathematica's ``Manipulate``.

    **How to use:**

    1. Create a figure and axes with ``plt.subplots`` (or any custom layout).
    2. Define a drawing function that accepts ``ax`` as its first argument,
       followed by keyword arguments matching the slider names.  Use the
       OOP matplotlib interface (``ax.plot``, ``ax.set_title``, etc.)
       rather than the pyplot state machine (``plt.plot``, ``plt.title``).
    3. Call ``manipulate(fn, fig, ax, **slider_specs)``.

    The axes are cleared automatically before each call to *fn*, so the
    drawing function should only draw — it does not need to call
    ``ax.clear()``.

    If you need access to the ``fig`` object inside the drawing function
    (e.g. for ``fig.colorbar`` or ``fig.suptitle``), capture it from the
    enclosing scope — it is the same object you passed to ``manipulate``.

    Args:
        fn: Drawing function with signature ``fn(ax, **slider_kwargs)``.
            Receives the axes as its first argument, followed by slider
            values as keyword arguments.  Should draw on *ax* using the
            OOP interface and must **not** call ``plt.show()``.
        fig: A matplotlib ``Figure`` created by the caller (e.g. via
            ``plt.subplots``).  Used internally for canvas updates and
            layout; not passed to *fn*.
        ax: The axes associated with *fig*.  May be a single ``Axes``, a
            list of ``Axes``, or a numpy array of ``Axes`` (as returned by
            ``plt.subplots``).  All axes are cleared before each redraw,
            then *ax* is passed to *fn* as its first argument.
        continuous_update: If ``True`` (default), the plot updates during
            slider dragging.  Set to ``False`` for expensive computations
            so the plot updates only when the slider is released.
        max_width: CSS ``max-width`` for the widget container (e.g.
            ``"600px"``).  Defaults to the figure's pixel width plus a small
            padding so that sliders align with the plot canvas.  Pass
            ``"none"`` to disable the constraint.
        **slider_specs: Each keyword argument defines a slider.

            **Real slider** -- pass a tuple:

            * ``(min, max)`` -- step auto-computed, default at midpoint.
            * ``(min, max, step)`` -- default at midpoint.
            * ``(min, max, step, default)`` -- fully explicit.

            **Complex slider** -- pass a :class:`ComplexSlider`::

                # Cartesian (Re/Im)
                d1=ComplexSlider(-0.5, 0.5, 0.01, default=0.1+0.2j)

                # Polar (r/θ)
                d1=ComplexSlider(0, 0.5, 0.01, default=0.3, polar=True)

    Raises:
        RuntimeError: If the ipympl backend is not active.
        ValueError: If no slider parameters are specified.
        TypeError: If slider names don't match *fn*'s parameters.

    Examples::

        # In a prior cell: %matplotlib widget
        import numpy as np
        import matplotlib.pyplot as plt
        from mrmustard.widgets import manipulate

        # --- Single axes ---
        fig, ax = plt.subplots(figsize=(6, 4))

        def plot_sine(ax, freq, amp):
            x = np.linspace(0, 2 * np.pi, 200)
            ax.plot(x, amp * np.sin(freq * x))
            ax.set_ylim(-2, 2)
            ax.set_title(f"freq={freq:.1f}, amp={amp:.1f}")

        manipulate(plot_sine, fig, ax, freq=(0.5, 10, 0.01, 1.0), amp=(0.1, 2.0, 0.01, 1.0), continuous_update=True)

        # --- Multiple axes ---
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        def plot_trig(axes, freq):
            x = np.linspace(0, 2 * np.pi, 200)
            axes[0].plot(x, np.sin(freq * x))
            axes[0].set_title("sin")
            axes[1].plot(x, np.cos(freq * x))
            axes[1].set_title("cos")

        manipulate(plot_trig, fig, axes, freq=(0.5, 10, 0.01, 1.0), continuous_update=True)
    """
    if not slider_specs:
        raise ValueError("At least one slider parameter must be specified.")

    _require_ipympl_backend()
    _validate_fn_accepts_sliders(fn, slider_specs)

    widget_map, display_rows = _build_slider_widgets(slider_specs, continuous_update)
    slider_box = ipywidgets.VBox(display_rows, layout=ipywidgets.Layout(width="100%"))

    effective_max_width = max_width if max_width is not None else _figure_max_width(fig)

    def _recompute(*_) -> None:
        _clear_axes(ax)
        fn(ax, **_gather_kwargs(widget_map))
        fig.canvas.draw_idle()

    _observe_all(widget_map, _recompute)
    _recompute()

    container = ipywidgets.VBox(
        [fig.canvas, slider_box],
        layout=ipywidgets.Layout(max_width=effective_max_width),
    )
    display(container)
    _dequeue_ipympl_figure(fig)
