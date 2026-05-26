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

"""Tests for the manipulate widget module.

Covers the pure-logic helpers (slider spec parsing, complex formatting,
ComplexSlider construction), the signature validation that runs before
any ipywidgets are created, and the ipympl backend requirement.
"""

import cmath
import math

import ipywidgets
import matplotlib.pyplot as plt
import pytest

from mrmustard.widgets.manipulate import (
    ComplexSlider,
    _build_slider_widgets,
    _clear_axes,
    _ComplexWidgetPair,
    _dequeue_ipympl_figure,
    _figure_max_width,
    _format_complex,
    _gather_kwargs,
    _make_float_slider,
    _observe_all,
    _parse_slider_spec,
    _require_ipympl_backend,
    _validate_fn_accepts_sliders,
)

# ---------------------------------------------------------------------------
# _parse_slider_spec
# ---------------------------------------------------------------------------


class TestParseSliderSpec:
    """Tests for _parse_slider_spec tuple parsing."""

    def test_two_element_tuple(self):
        """(min, max) derives step and default automatically."""
        min_val, max_val, step, default = _parse_slider_spec((0.0, 1.0))
        assert min_val == 0.0
        assert max_val == 1.0
        assert step == pytest.approx(0.01)  # (1 - 0) / 100
        assert default == pytest.approx(0.5)  # midpoint

    def test_three_element_tuple(self):
        """(min, max, step) derives default as the midpoint."""
        min_val, max_val, step, default = _parse_slider_spec((-2.0, 2.0, 0.1))
        assert min_val == -2.0
        assert max_val == 2.0
        assert step == pytest.approx(0.1)
        assert default == pytest.approx(0.0)

    def test_four_element_tuple(self):
        """(min, max, step, default) uses all values explicitly."""
        min_val, max_val, step, default = _parse_slider_spec((1.0, 5.0, 0.5, 3.0))
        assert min_val == 1.0
        assert max_val == 5.0
        assert step == 0.5
        assert default == 3.0

    def test_values_are_cast_to_float(self):
        """Integer inputs are converted to float."""
        result = _parse_slider_spec((0, 10))
        assert all(isinstance(v, float) for v in result)

    @pytest.mark.parametrize("bad_spec", [(), (1,), (1, 2, 3, 4, 5)])
    def test_invalid_length_raises(self, bad_spec):
        """Tuples with fewer than 2 or more than 4 elements raise ValueError."""
        with pytest.raises(ValueError, match="Slider spec must be"):
            _parse_slider_spec(bad_spec)


# ---------------------------------------------------------------------------
# _format_complex
# ---------------------------------------------------------------------------


class TestFormatComplex:
    """Tests for the _format_complex display helper."""

    def test_positive_imaginary(self):
        """Positive imaginary part uses '+' separator."""
        assert _format_complex(1.0 + 2.0j) == "1.0000 + 2.0000i"

    def test_negative_imaginary(self):
        """Negative imaginary part uses '-' separator with positive magnitude."""
        assert _format_complex(1.0 - 2.0j) == "1.0000 - 2.0000i"

    def test_zero_imaginary(self):
        """Zero imaginary part formats as positive zero."""
        assert _format_complex(3.0 + 0.0j) == "3.0000 + 0.0000i"

    def test_both_zero(self):
        """Both parts zero."""
        assert _format_complex(0.0 + 0.0j) == "0.0000 + 0.0000i"


# ---------------------------------------------------------------------------
# ComplexSlider construction
# ---------------------------------------------------------------------------


class TestComplexSliderCartesian:
    """Tests for ComplexSlider in Cartesian (Re/Im) mode."""

    def test_default_cartesian_mode(self):
        """Default mode is Cartesian with labels Re/Im."""
        cs = ComplexSlider(-1.0, 1.0)
        assert cs.polar is False
        assert cs.first_label == "Re"
        assert cs.second_label == "Im"

    def test_cartesian_range_shared(self):
        """Both components share the same min/max/step in Cartesian mode."""
        cs = ComplexSlider(-0.5, 0.5, step=0.01)
        assert cs.first_spec[:3] == (-0.5, 0.5, 0.01)
        assert cs.second_spec[:3] == (-0.5, 0.5, 0.01)
        assert cs.first_spec[3] == pytest.approx(0.0)
        assert cs.second_spec[3] == pytest.approx(0.0)

    def test_cartesian_default_decomposition(self):
        """Complex default is decomposed into Re and Im defaults."""
        cs = ComplexSlider(-1.0, 1.0, step=0.1, default=0.3 + 0.7j)
        assert cs.first_spec[3] == pytest.approx(0.3)
        assert cs.second_spec[3] == pytest.approx(0.7)

    def test_cartesian_auto_step(self):
        """Step is auto-computed as (max - min) / 100 when not provided."""
        cs = ComplexSlider(-2.0, 2.0)
        expected_step = 4.0 / 100
        assert cs.first_spec[2] == pytest.approx(expected_step)
        assert cs.second_spec[2] == pytest.approx(expected_step)


class TestComplexSliderPolar:
    """Tests for ComplexSlider in polar (r/theta) mode."""

    def test_polar_labels(self):
        """Polar mode uses r/theta labels."""
        cs = ComplexSlider(0, 1.0, polar=True)
        assert cs.polar is True
        assert cs.first_label == "r"
        assert cs.second_label == "θ"

    def test_polar_r_range(self):
        """r range is [0, max_val]."""
        cs = ComplexSlider(0, 2.0, step=0.05, polar=True)
        assert cs.first_spec[0] == 0.0
        assert cs.first_spec[1] == 2.0
        assert cs.first_spec[2] == 0.05

    def test_polar_theta_range(self):
        """theta range is [-pi, pi]."""
        cs = ComplexSlider(0, 1.0, step=0.01, polar=True)
        assert cs.second_spec[0] == pytest.approx(-math.pi)
        assert cs.second_spec[1] == pytest.approx(math.pi)

    def test_polar_default_decomposition(self):
        """Complex default is decomposed into r and theta."""
        z = 1.0 + 1.0j  # r = sqrt(2), theta = pi/4
        cs = ComplexSlider(0, 2.0, step=0.01, default=z, polar=True)
        expected_r, expected_theta = cmath.polar(z)
        assert cs.first_spec[3] == pytest.approx(expected_r)
        assert cs.second_spec[3] == pytest.approx(expected_theta)

    def test_polar_zero_default_gives_zero_angle(self):
        """A zero default has r=0 and theta=0 (not undefined)."""
        cs = ComplexSlider(0, 1.0, step=0.01, default=0, polar=True)
        assert cs.first_spec[3] == 0.0
        assert cs.second_spec[3] == 0.0

    def test_polar_auto_step(self):
        """Step is auto-computed as max_val / 100 in polar mode."""
        cs = ComplexSlider(0, 5.0, polar=True)
        expected_step = 5.0 / 100
        assert cs.first_spec[2] == pytest.approx(expected_step)


# ---------------------------------------------------------------------------
# ComplexSlider.from_parts
# ---------------------------------------------------------------------------


class TestComplexSliderFromParts:
    """Tests for the from_parts alternate constructor."""

    def test_cartesian_from_parts(self):
        """from_parts creates a Cartesian slider with independent ranges."""
        cs = ComplexSlider.from_parts(
            first_spec=(-1.0, 1.0, 0.01),
            second_spec=(0.0, 2.0, 0.05),
        )
        assert cs.polar is False
        assert cs.first_label == "Re"
        assert cs.second_label == "Im"
        # first component (midpoint default = 0.0)
        assert cs.first_spec == pytest.approx((-1.0, 1.0, 0.01, 0.0))
        # second component (midpoint default = 1.0)
        assert cs.second_spec == pytest.approx((0.0, 2.0, 0.05, 1.0))

    def test_polar_from_parts(self):
        """from_parts creates a polar slider with independent ranges."""
        cs = ComplexSlider.from_parts(
            first_spec=(0.0, 3.0, 0.1),
            second_spec=(-math.pi, math.pi, 0.01),
            polar=True,
        )
        assert cs.polar is True
        assert cs.first_label == "r"
        assert cs.second_label == "θ"
        # first component (midpoint default = 1.5)
        assert cs.first_spec == pytest.approx((0.0, 3.0, 0.1, 1.5))
        # second component (midpoint default ≈ 0.0)
        assert cs.second_spec == pytest.approx((-math.pi, math.pi, 0.01, 0.0))

    def test_from_parts_uses_parse_slider_spec(self):
        """from_parts accepts 2, 3, or 4-element tuples like real sliders."""
        cs = ComplexSlider.from_parts(
            first_spec=(0.0, 1.0),  # 2-element: step and default auto-computed
            second_spec=(0.0, 1.0, 0.1, 0.5),  # 4-element: fully explicit
        )
        # first: step = 0.01, default = 0.5 (midpoint)
        assert cs.first_spec[2] == pytest.approx(0.01)
        assert cs.first_spec[3] == pytest.approx(0.5)
        # second: step and default as given
        assert cs.second_spec[2] == pytest.approx(0.1)
        assert cs.second_spec[3] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _validate_fn_accepts_sliders
# ---------------------------------------------------------------------------


class TestValidateFnAcceptsSliders:
    """Tests for the upfront signature validation."""

    def test_matching_names_passes(self):
        """No error when slider names match fn parameters (with ax positional arg)."""

        def fn(ax, x, y):
            pass

        _validate_fn_accepts_sliders(fn, {"x": (0, 1), "y": (0, 1)})

    def test_mismatched_names_raises(self):
        """TypeError when slider names don't match fn parameters."""

        def fn(ax, x):
            pass

        with pytest.raises(TypeError, match="do not match"):
            _validate_fn_accepts_sliders(fn, {"x": (0, 1), "z": (0, 1)})

    def test_var_keyword_accepts_any_name(self):
        """fn with **kwargs accepts any slider name."""

        def fn(ax, **kwargs):
            pass

        _validate_fn_accepts_sliders(fn, {"anything": (0, 1), "goes": (0, 1)})

    def test_keyword_only_params_accepted(self):
        """Keyword-only parameters (after *) are valid targets."""

        def fn(ax, *, alpha, beta):
            pass

        _validate_fn_accepts_sliders(fn, {"alpha": (0, 1), "beta": (0, 1)})

    def test_subset_of_params_is_fine(self):
        """Sliders need not cover all of fn's parameters (extras get defaults)."""

        def fn(ax, x, y=0, z=0):
            pass

        _validate_fn_accepts_sliders(fn, {"x": (0, 1)})

    def test_complex_slider_specs_validated(self):
        """ComplexSlider specs are validated the same way as tuples."""

        def fn(ax, x):
            pass

        with pytest.raises(TypeError, match="do not match"):
            _validate_fn_accepts_sliders(fn, {"x": (0, 1), "bad": ComplexSlider(-1, 1)})


# ---------------------------------------------------------------------------
# _ComplexWidgetPair
# ---------------------------------------------------------------------------


class TestComplexWidgetPair:
    """Tests for _ComplexWidgetPair value assembly and observation."""

    def test_cartesian_value_assembles_complex(self):
        """Cartesian mode returns complex(Re, Im) from slider positions."""
        first = ipywidgets.FloatSlider(value=1.5, min=-3, max=3)
        second = ipywidgets.FloatSlider(value=-0.7, min=-3, max=3)
        pair = _ComplexWidgetPair(first, second, polar=False, name="z")
        assert pair.value == complex(1.5, -0.7)

    def test_polar_value_converts_via_cmath_rect(self):
        """Polar mode converts (r, θ) to complex via cmath.rect."""
        r, theta = 2.0, math.pi / 4
        first = ipywidgets.FloatSlider(value=r, min=0, max=5)
        second = ipywidgets.FloatSlider(value=theta, min=-math.pi, max=math.pi)
        pair = _ComplexWidgetPair(first, second, polar=True, name="z")
        expected = cmath.rect(r, theta)
        assert pair.value.real == pytest.approx(expected.real, abs=1e-10)
        assert pair.value.imag == pytest.approx(expected.imag, abs=1e-10)

    def test_label_shows_name_and_formatted_value(self):
        """The HTML label displays the parameter name and current complex value."""
        first = ipywidgets.FloatSlider(value=1.0, min=-2, max=2)
        second = ipywidgets.FloatSlider(value=2.0, min=-2, max=2)
        pair = _ComplexWidgetPair(first, second, polar=False, name="alpha")
        assert "alpha" in pair.label.value
        assert "1.0000" in pair.label.value
        assert "2.0000" in pair.label.value

    def test_observe_registers_on_both_sliders(self):
        """observe() fires the callback when either slider changes."""
        first = ipywidgets.FloatSlider(value=0, min=-1, max=1, step=0.1)
        second = ipywidgets.FloatSlider(value=0, min=-1, max=1, step=0.1)
        pair = _ComplexWidgetPair(first, second, polar=False, name="z")
        calls = []
        pair.observe(lambda change: calls.append(1), names="value")
        first.value = 0.5
        second.value = -0.3
        assert len(calls) == 2


# ---------------------------------------------------------------------------
# _make_float_slider
# ---------------------------------------------------------------------------


class TestMakeFloatSlider:
    """Tests for the _make_float_slider builder."""

    def test_slider_matches_spec(self):
        """Built slider has value, min, max, step from the parsed spec."""
        slider = _make_float_slider("freq", (1.0, 10.0, 0.5, 3.0), continuous_update=False)
        assert slider.value == 3.0
        assert slider.min == 1.0
        assert slider.max == 10.0
        assert slider.step == 0.5
        assert slider.description == "freq"
        assert slider.continuous_update is False


# ---------------------------------------------------------------------------
# _build_slider_widgets
# ---------------------------------------------------------------------------


class TestBuildSliderWidgets:
    """Tests for _build_slider_widgets routing logic."""

    def test_tuple_spec_produces_float_slider(self):
        """A plain tuple spec yields a FloatSlider in the widget map."""
        widget_map, rows = _build_slider_widgets({"x": (0, 1)}, continuous_update=True)
        assert isinstance(widget_map["x"], ipywidgets.FloatSlider)
        assert len(rows) == 1

    def test_complex_spec_produces_widget_pair(self):
        """A ComplexSlider spec yields a _ComplexWidgetPair."""
        cs = ComplexSlider(-1, 1, 0.01)
        widget_map, rows = _build_slider_widgets({"z": cs}, continuous_update=True)
        assert isinstance(widget_map["z"], _ComplexWidgetPair)
        assert len(rows) == 1

    def test_mixed_specs_produce_correct_types(self):
        """Real and complex specs coexist in the same widget map."""
        specs = {"x": (0, 1), "z": ComplexSlider(-1, 1)}
        widget_map, rows = _build_slider_widgets(specs, continuous_update=True)
        assert isinstance(widget_map["x"], ipywidgets.FloatSlider)
        assert isinstance(widget_map["z"], _ComplexWidgetPair)
        assert len(rows) == 2


# ---------------------------------------------------------------------------
# _gather_kwargs
# ---------------------------------------------------------------------------


class TestGatherKwargs:
    """Tests for _gather_kwargs value collection."""

    def test_collects_real_and_complex_defaults(self):
        """Extracts .value from FloatSliders and _ComplexWidgetPairs alike."""
        specs = {
            "x": (0.0, 1.0, 0.01, 0.5),
            "z": ComplexSlider(-1, 1, 0.01, default=0.3 + 0.7j),
        }
        widget_map, _ = _build_slider_widgets(specs, continuous_update=True)
        kwargs = _gather_kwargs(widget_map)
        assert kwargs["x"] == pytest.approx(0.5)
        assert kwargs["z"].real == pytest.approx(0.3, abs=1e-3)
        assert kwargs["z"].imag == pytest.approx(0.7, abs=1e-3)


# ---------------------------------------------------------------------------
# _observe_all
# ---------------------------------------------------------------------------


class TestObserveAll:
    """Tests for _observe_all callback wiring."""

    def test_callback_fires_for_each_slider_change(self):
        """Changing any slider triggers the shared callback."""
        specs = {"a": (0, 1, 0.1, 0.5), "b": (0, 1, 0.1, 0.5)}
        widget_map, _ = _build_slider_widgets(specs, continuous_update=True)
        calls = []
        _observe_all(widget_map, lambda *_: calls.append(1))
        widget_map["a"].value = 0.8
        widget_map["b"].value = 0.2
        assert len(calls) == 2


# ---------------------------------------------------------------------------
# _clear_axes
# ---------------------------------------------------------------------------


class TestClearAxes:
    """Tests for _clear_axes handling of single and multiple axes."""

    def test_clears_single_axes(self):
        """A single Axes is cleared."""
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        assert len(ax.lines) == 1
        _clear_axes(ax)
        assert len(ax.lines) == 0
        plt.close(fig)

    def test_clears_list_of_axes(self):
        """Each Axes in a list is cleared."""
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot([0, 1], [0, 1])
        ax2.plot([0, 1], [1, 0])
        _clear_axes([ax1, ax2])
        assert len(ax1.lines) == 0
        assert len(ax2.lines) == 0
        plt.close(fig)

    def test_clears_numpy_array_of_axes(self):
        """Axes in a numpy ndarray (from plt.subplots) are cleared."""
        fig, axes = plt.subplots(2, 2)
        for ax in axes.flat:
            ax.plot([0, 1], [0, 1])
        _clear_axes(axes)
        for ax in axes.flat:
            assert len(ax.lines) == 0
        plt.close(fig)


# ---------------------------------------------------------------------------
# _figure_max_width
# ---------------------------------------------------------------------------


class TestFigureMaxWidth:
    """Tests for _figure_max_width CSS string computation."""

    def test_returns_pixel_string(self):
        """Result is a CSS pixel value derived from figure width and DPI."""
        fig = plt.figure(figsize=(8, 5))
        result = _figure_max_width(fig)
        assert result.endswith("px")
        assert int(result[:-2]) == int(8 * fig.dpi) + 24
        plt.close(fig)

    def test_custom_padding(self):
        """Custom padding is reflected in the output."""
        fig = plt.figure(figsize=(6, 4))
        result = _figure_max_width(fig, padding=50)
        assert int(result[:-2]) == int(6 * fig.dpi) + 50
        plt.close(fig)


# ---------------------------------------------------------------------------
# _require_ipympl_backend
# ---------------------------------------------------------------------------


class TestRequireIpymplBackend:
    """Tests for _require_ipympl_backend guard."""

    def test_raises_without_ipympl(self):
        """Raises RuntimeError when the active backend is not ipympl."""
        with pytest.raises(RuntimeError, match="ipympl"):
            _require_ipympl_backend()


# ---------------------------------------------------------------------------
# _dequeue_ipympl_figure
# ---------------------------------------------------------------------------


class TestDequeueIpymplFigure:
    """Tests for _dequeue_ipympl_figure silent-failure path."""

    def test_does_not_raise_without_ipympl(self):
        """Runs without error when ipympl is not the active backend."""
        fig = plt.figure()
        _dequeue_ipympl_figure(fig)
        plt.close(fig)
