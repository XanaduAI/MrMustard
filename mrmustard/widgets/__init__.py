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

from .css import FOCK_CSS, WIRES_CSS


NO_MARGIN = {"l": 0, "r": 0, "t": 0, "b": 0}


def fock(rep):
    """Create a widget to display a Fock representation."""
    shape = rep.array.shape
    if len(shape) == 2:
        xaxis = "n"
        yaxis = "batch"
        array = rep.array
    elif len(shape) == 3 and shape[0] == 1:
        xaxis = "axis 1"
        yaxis = "axis 0"
        array = rep.array[0]
    else:  # TODO: add multi-dimensional visualization
        raise ValueError(f"unexpected Fock representation with shape {shape}")

    text = [
        [f"{yaxis}: {y}<br />{xaxis}: {x}<br />val: {val}<br />" for x, val in enumerate(row)]
        for y, row in enumerate(array)
    ]

    layout = {
        "height": 200,
        "width": 430,
        "margin": NO_MARGIN,
        "showlegend": False,
        "xaxis": {
            "title": xaxis,
            "showgrid": True,
            "showline": True,
        },
        "yaxis": {
            "title": yaxis,
            "autorange": "reversed",
            "showgrid": True,
            "showticklabels": False,
        },
    }
    plot_widget = go.FigureWidget(
        data=go.Heatmap(
            z=abs(array),
            colorscale="viridis",
            showscale=False,
            hoverinfo="text",
            text=text,
        ),
        layout=layout,
    )

    header_widget = widgets.HTML("<h1 class=h1-fock>Fock Representation</h1>")
    table_widget = widgets.HTML(
        "<table class=table-fock>"
        f"<tr><th>Ansatz</th><td>{rep.ansatz.__class__.__qualname__}</td></tr>"
        f"<tr><th>Shape</th><td>{shape}</td></tr>"
        "</table>"
    )
    return widgets.HBox(
        children=[
            widgets.HTML(FOCK_CSS),
            widgets.VBox(children=[header_widget, table_widget]),
            plot_widget,
        ],
        layout=widgets.Layout(flex_flow="row wrap"),
    )


def bargmann(rep, batch_idx=None):
    """Create a widget to display a Bargmann representation."""
    if batch_idx is None:
        batch_size = rep.A.shape[0]
        if batch_size == 1:  # no batching, omit the slider
            return bargmann(rep, batch_idx=0)
        batch_idx = 0
        slider = widgets.IntSlider(0, min=0, max=batch_size - 1, description="Batch index:")
        stack = widgets.Stack(
            [bargmann(rep, batch_idx=i) for i in range(batch_size)], selected_index=0
        )
        widgets.jslink((slider, "value"), (stack, "selected_index"))
        return widgets.VBox([slider, stack])

    A = rep.A[batch_idx]
    b = rep.b[batch_idx]
    c = rep.c[batch_idx]

    rows = ["".join([f"<td>{ele}</td>" for ele in row]) for row in A]
    matrix_A = f"<table><tr>{'</tr><tr>'.join(rows)}</tr></table>"

    b_str = [f"<tr><td>{x}</td></tr>" for x in b]
    vector_b = f"<table>{''.join(b_str)}</table>"

    scalar_c = f"<div>{c}</div>"

    header_w = widgets.HTML(
        f"""
        <h1>Bargmann Representation</h1>
        <div style="font-weight: bold;">
            Ansatz: {rep.ansatz.__class__.__qualname__}</br>
            Eigvals of A: {np.linalg.eigvals(A)}
        </div>
        """
    )
    triple_w = widgets.HTML(
        f"""
        <style>.triple th {{ text-align: center; background-color: #FBAB7E; }}</style>
        <table class="triple">
            <tr>
                <th>A</th>
                <th>b</th>
                <th>c</th>
            </tr>
            <tr>
                <td>{matrix_A}</td>
                <td>{vector_b}</td>
                <td>{scalar_c}</td>
            </tr>
        </table>
        """
    )

    return widgets.VBox([header_w, triple_w], layout={"max_width": "50%"})


def wires(obj):
    """Create a widget to display a Wires objects."""
    modes = [obj.output.bra, obj.input.bra, obj.output.ket, obj.input.ket]
    labels = ["out bra", "in bra", "out ket", "in ket"]
    colors = ["black" if m else "gainsboro" for m in modes]

    ##### The wires graphic #####

    def mode_to_str(m):
        max_modes = 3
        result = ", ".join(list(map(str, m.modes))[:max_modes])
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
    for mode, label in zip(modes, labels):
        if not mode:
            continue

        title_row = f'<td rowspan="{len(mode)}">{label}</td>'
        table_data = [f"<td>{m}</td><td>{mode[m].indices[0]}</td>" for m in mode.modes]
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
        {WIRES_CSS}
        <div class="modes-grid">
            <div class="square">Wires</div>
            {"".join(wire_labels)}
        </div></br>{index_table}
        """
    )


def state(obj, is_ket=False, is_fock=False):
    """Create a widget to display a state."""
    fock_yn, bargmann_yn = ("✅", "❌") if is_fock else ("❌", "✅")
    table_widget = widgets.HTML(
        f"""
        <h1>{obj.name or obj.__class__.__name__}</h1>
        <table style="border-collapse: collapse; text-align: center">
            <tr>
                <th>Purity</th>
                <th>Probability</th>
                <th>Number of modes</th>
                <th>Class</th>
                <th>Bargmann</th>
                <th>Fock</th>
            </tr>

            <tr>
                <td>{f"{obj.purity}" if obj.purity == 1 else f"{obj.purity :.2e}"}</td>
                <td>{f"{100*obj.probability:.3e} %" if obj.probability < 0.001 else f"{obj.probability:.2%}"}</td></td>
                <td>{obj.n_modes}</td></td>
                <td>{"Ket" if is_ket else "DM"}</td></td>
                <td>{bargmann_yn}</td></td>
                <td>{fock_yn}</td>
            </tr>
        </table>
        """
    )

    if obj.n_modes != 1:
        return table_widget

    style_widget = widgets.HTML(
        """
        <style>
        tr { display: block; float: left; }
        th, td { display: block; }
        table { margin: auto; }
        </style>
        """
    )
    left_widget = widgets.VBox(
        [table_widget, go.FigureWidget(obj.visualize_dm(40))],
        layout=widgets.Layout(flex_flow="column nowrap", max_width="800px"),
    )
    return widgets.HBox(
        [style_widget, left_widget, go.FigureWidget(obj.visualize_2d(resolution=100))],
        layout=widgets.Layout(flex="0 0 auto", flex_flow="row wrap"),
    )
