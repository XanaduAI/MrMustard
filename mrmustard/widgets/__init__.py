# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""IPython widgets for various objects in MrMustard."""

import ipywidgets as widgets
import numpy as np
import plotly.graph_objs as go
from IPython import get_ipython
from IPython.terminal.interactiveshell import TerminalInteractiveShell

from mrmustard import math

from .css import FOCK, STATE, TABLE, WIRES

NO_MARGIN = {"l": 0, "r": 0, "t": 0, "b": 0}
IN_INTERACTIVE_SHELL = isinstance(get_ipython(), TerminalInteractiveShell)


def _batch_widget(obj, batch_size, widget_fn, *widget_args):
    """Return a batch of widgets with a slider."""
    if batch_size == 1:
        return widget_fn(obj, *widget_args, batch_idx=0)
    slider = widgets.IntSlider(0, min=0, max=batch_size - 1, description="Batch index:")
    stack = widgets.Stack(
        [widget_fn(obj, *widget_args, batch_idx=i) for i in range(batch_size)],
        selected_index=0,
    )
    widgets.jslink((slider, "value"), (stack, "selected_index"))
    return widgets.VBox([slider, stack])


def fock_1d(array):
    """Return a plot widget for a 2D Fock representation."""
    layout = {"margin": NO_MARGIN, "height": 200, "width": 250}
    mag_plot = go.Bar(y=abs(array))
    phase_plot = go.Bar(y=np.angle(array))
    plots = [go.FigureWidget(p, layout=layout) for p in [mag_plot, phase_plot]]
    return widgets.Tab(plots, titles=["Magnitude", "Phase"])


def fock_2d(array):
    """Return a plot widget for a 1D Fock representation."""
    layout = {"height": 200, "width": 250, "margin": NO_MARGIN, "yaxis": {"autorange": "reversed"}}
    mag_plot = go.Heatmap(z=abs(array), colorscale="viridis", showscale=False)
    phase_plot = go.Heatmap(z=np.angle(array), colorscale="agsunset", showscale=False)
    plots = [go.FigureWidget(data=p, layout=layout) for p in [mag_plot, phase_plot]]
    return widgets.Tab(plots, titles=["Magnitude", "Phase"])


def fock(rep):
    """Create a widget to display a Fock representation."""
    if rep.batch_size > 1:  # pragma: no cover
        # the batch dimension should be trivial for Fock representations
        return None

    shape = rep.core_shape
    rep_array = rep.array[0] if rep.batch_shape else rep.array
    if len(shape) == 1:
        plot_widget = fock_1d(rep_array)
    elif len(shape) == 2:
        plot_widget = fock_2d(rep_array)
    else:  # TODO: add multi-dimensional visualization
        return None

    header_widget = widgets.HTML("<h1 class=h1-fock>Fock Representation</h1>")
    table_widget = widgets.HTML(
        TABLE + f"<table class=table-fock><tr><th>Shape</th><td>{shape}</td></tr></table>",
    )
    left_widget = widgets.VBox(children=[header_widget, table_widget])
    plot_widget.layout.padding = "10px"
    left_widget.layout.padding = "10px"

    return widgets.HBox(
        children=[widgets.HTML(FOCK), left_widget, plot_widget],
        layout=widgets.Layout(flex_flow="row wrap"),
    )


def bargmann(rep, batch_idx: int | None = None):
    """Create a widget to display a Bargmann representation."""
    if batch_idx is None and rep.batch_shape:
        return _batch_widget(rep, rep.batch_size, bargmann)

    if rep.batch_shape:
        A = math.reshape(rep.A, (-1, *rep.A.shape[-2:]))[batch_idx]
        b = math.reshape(rep.b, (-1, *rep.b.shape[-1:]))[batch_idx]
        c = math.reshape(rep.c, (-1, *rep.c.shape[rep.batch_dims :]))[batch_idx]
    else:
        A, b, c = rep.triple

    def get_abc_str(A, b, c, round_val):
        if round_val >= 0:
            A = np.round(A, round_val)
            b = np.round(b, round_val)
            c = np.round(c, round_val)

        rows = ["".join([f"<td>{ele}</td>" for ele in row]) for row in A]
        A_str = f"<table><tr>{'</tr><tr>'.join(rows)}</tr></table>"

        b_str = [f"<tr><td>{x}</td></tr>" for x in b]
        b_str = f"<table>{''.join(b_str)}</table>"

        c_str = f"<div>{c}</div>"
        return A_str, b_str, c_str

    triple_fstr = """
    <table>
        <tr><th>A</th> <th>b</th> <th>c</th> </tr>
        <tr><td>{}</td><td>{}</td><td>{}</td></tr>
    </table>
    """

    round_default = 4
    round_w = widgets.IntText(value=round_default, description="Rounding (negative -> none):")
    round_w.style.description_width = "230px"
    header_w = widgets.HTML("<h1>Bargmann Representation</h1>")
    triple_w = widgets.HTML(TABLE + triple_fstr.format(*get_abc_str(A, b, c, round_default)))
    eigs_header_w = widgets.HTML("<h2>Eigenvalues of A</h2>")
    eigvals_w = go.FigureWidget(
        layout=go.Layout(
            xaxis={"range": [-1.1, 1.1], "minallowed": -1.1, "maxallowed": 1.1},
            yaxis={"range": [-1.1, 1.1], "scaleanchor": "x", "scaleratio": 1},
            width=180,
            height=180,
            margin=NO_MARGIN,
        ),
    )
    # Replace config to hide the Plotly mode bar
    # See: https://github.com/plotly/plotly.py/issues/1074#issuecomment-1471486307
    eigvals_w._config = eigvals_w._config | {"displayModeBar": False}
    eigvals_w.add_shape(
        type="circle",
        xref="x",
        yref="y",
        x0=-1,
        y0=-1,
        x1=1,
        y1=1,
        line_color="LightSeaGreen",
    )

    eigvals = np.linalg.eigvals(A)
    text = [f"re: {np.real(e)}<br />im: {np.imag(e)}" for e in eigvals]
    eigvals_w.add_trace(
        go.Scatter(
            x=np.real(eigvals),
            y=np.imag(eigvals),
            hoverinfo="text",
            text=text,
            mode="markers",
        ),
    )

    def on_value_change(change):
        round_val = change.new
        triple_w.value = triple_fstr.format(*get_abc_str(A, b, c, round_val))

    round_w.observe(on_value_change, names="value")

    eigs_vbox = widgets.VBox([eigs_header_w, eigvals_w])
    return widgets.Box(
        [widgets.VBox([header_w, round_w, triple_w]), eigs_vbox],
        layout=widgets.Layout(max_width="50%", flex_flow="row wrap"),
    )


def wires(obj):
    """Create a widget to display a Wires objects."""
    modes = [obj.input.bra, obj.output.bra, obj.input.ket, obj.output.ket]
    labels = ["in bra", "out bra", "in ket", "out ket"]
    colors = ["black" if m else "gainsboro" for m in modes]

    ##### The wires graphic #####

    def mode_to_str(m):
        max_modes = 3
        result = ", ".join(list(map(str, sorted(m.modes)))[:max_modes])
        return (result + ", ...") if len(m) > max_modes else result

    mode_div = """
    <div class="braket">
        <div style="color: {color}; font-size: 13px; padding-left: 5px">{label}: {mode}</div>
        <div class="line" style="background-color: {color}; border-top-color: {color}; height: 1px"></div>
    </div>
    """

    wire_labels = [
        mode_div.format(color=c, label=l, mode=mode_to_str(m))
        for m, l, c in zip(modes, labels, colors)
    ]

    ##### The index table #####

    wire_tables = []
    for i in [1, 0, 3, 2]:  # match index order
        mode = modes[i]
        if not mode:
            continue
        label = labels[i]

        title_row = f'<td rowspan="{len(mode)}">{label}</td>'
        table_data = [f"<td>{m}</td><td>{mode[m].indices[0]}</td>" for m in sorted(mode.modes)]
        wire_tables.append(title_row + "</tr><tr>".join(table_data))

    index_table = f"""
    <table>
        <tr>
            <th>Set</th>
            <th>Mode</th>
            <th>Index</th>
        </tr>
        <tr>
            {"</tr><tr>".join(wire_tables)}
        </tr>
    </table>
    """

    ##### The final widget #####

    return widgets.HTML(
        f"""
        {WIRES}{TABLE}
        <div class="modes-grid">
            <div class="square">Wires</div>
            {"".join(wire_labels)}
        </div></br>{index_table}
        """,
    )


def state(obj, is_ket, is_fock):
    """Create a widget to display a state."""
    fock_yn, bargmann_yn = ("✅", "❌") if is_fock else ("❌", "✅")
    table_widget = widgets.HTML(
        f"""{TABLE}{STATE}
    <h1>{obj.name or type(obj).__name__}</h1>
    <table class="state-table">
        <tr>
            <th>Purity</th>
            <th>Probability</th>
            <th>Number of modes</th>
            <th>Class</th>
            <th>Bargmann</th>
            <th>Fock</th>
        </tr>

        <tr>
            <td>{f"{obj.purity}" if obj.purity == 1 else f"{obj.purity:.2e}"}</td>
            <td>{f"{100 * obj.probability:.3e} %" if obj.probability < 0.001 else f"{obj.probability:.2%}"}</td></td>
            <td>{obj.n_modes}</td></td>
            <td>{"Ket" if is_ket else "DM"}</td></td>
            <td>{bargmann_yn}</td></td>
            <td>{fock_yn}</td>
        </tr>
    </table>
    """,
    )

    if obj.n_modes != 1:
        wires_widget = wires(obj.wires)
        table_widget.layout.padding = "10px"
        wires_widget.layout.padding = "10px"
        return widgets.Box([table_widget, wires_widget])

    left_widget = widgets.VBox(
        [table_widget, go.FigureWidget(obj.visualize_dm(return_fig=True))],
        layout=widgets.Layout(flex_flow="column nowrap", max_width="800px"),
    )
    right_widget = go.FigureWidget(obj.visualize_2d(resolution=100, return_fig=True))
    return widgets.HBox(
        [left_widget, right_widget],
        layout=widgets.Layout(flex="0 0 auto", flex_flow="row wrap"),
    )
