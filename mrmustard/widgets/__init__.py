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
        "<h1>Bargmann Representation</h1>" f"<h3>Ansatz: {rep.ansatz.__class__.__qualname__}</h3>"
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

    return widgets.VBox([header_w, triple_w])


def wires(obj):
    """Create a widget to display a Wires objects."""
    height_line_ob = "4px" if obj.output.bra else "2px"
    height_line_ib = "4px" if obj.input.bra else "2px"
    height_line_ok = "4px" if obj.output.ket else "2px"
    height_line_ik = "4px" if obj.input.ket else "2px"

    color_ob = "black" if obj.output.bra else "gainsboro"
    color_ib = "black" if obj.input.bra else "gainsboro"
    color_ok = "black" if obj.output.ket else "gainsboro"
    color_ik = "black" if obj.input.ket else "gainsboro"

    max_n_modes = 5
    n_modes = len(obj.modes) if len(obj.modes) < max_n_modes else max_n_modes
    dots = "" if n_modes < max_n_modes else ", ..."

    modes_ob = ", ".join(str(s) for s in list(obj.output.bra.modes)[:n_modes])
    modes_ib = ", ".join(str(s) for s in list(obj.input.bra.modes)[:n_modes])
    modes_ok = ", ".join(str(s) for s in list(obj.output.ket.modes)[:n_modes])
    modes_ik = ", ".join(str(s) for s in list(obj.input.ket.modes)[:n_modes])

    modes_ob += "" if not modes_ob else dots
    modes_ib += "" if not modes_ib else dots
    modes_ok += "" if not modes_ok else dots
    modes_ik += "" if not modes_ik else dots

    n_grid_items = sum(1 if m else 0 for m in [modes_ob, modes_ib, modes_ok, modes_ik])
    n_grid_rows = 1 if n_grid_items < 3 else 2
    n_grid_cols = 1 if n_grid_items == 1 else 2

    style_widget = widgets.HTML(WIRES_CSS.format(n_grid_cols, n_grid_rows))
    image_widget = widgets.HTML(
        f"""
        <h1>Wires</h1>
            <div class="container-wires">
            <div class="in">
                <div class="bra">
                <div class="line" style="background-color: {color_ib}; height: {height_line_ib}"></div>
                <p class="text-wires type" style="color: {color_ib}">in bra</p>
                <p class="text-wires modes">{modes_ib}</p>
                </div>
                <div class="ket">
                <div class="line" style="background-color: {color_ik}; height: {height_line_ik}"></div>
                <p class="text-wires type" style="color: {color_ik}">in ket</p>
                <p class="text-wires modes">{modes_ik}</p>
                </div>
            </div>

            <div class="square"></div>

            <div class="out">
                <div class="bra">
                <div class="line" style="background-color: {color_ob}; height: {height_line_ob}"></div>
                <p class="text-wires type" style="color: {color_ob}">out bra</p>
                <p class="text-wires modes">{modes_ob}</p>
                </div>
                <div class="ket">
                <div class="line" style="background-color: {color_ok}; height: {height_line_ok}"></div>
                <p class="text-wires type" style="color: {color_ok}">out ket</p>
                <p class="text-wires modes">{modes_ok}</p>
                </div>
            </div>
        </div>
        """
    )

    wire_tables = []
    for (
        mode,
        in_out,
        bra_ket,
    ) in [
        (modes_ob, "output", "bra"),
        (modes_ib, "input", "bra"),
        (modes_ok, "output", "ket"),
        (modes_ik, "input", "ket"),
    ]:
        if not mode:
            continue

        bra_ket_obj = getattr(getattr(obj, in_out), bra_ket)
        table_data = "\n".join(
            [
                "<tr>"
                f'<td class="td-wires">{m}</td>'
                f'<td class="td-wires">{bra_ket_obj[m].indices[0]}</td>'
                "</tr>"
                for m in bra_ket_obj.modes
            ]
        )
        wire_tables.append(
            f"""
            <div class="grid-item-wires">
                <div class="grid-item-type">{in_out[:-3]} {bra_ket}</div>
                <div class="table-container-wires">
                    <table class="table-wires">
                        <thead>
                            <tr>
                            <th class="th-wires">mode</th>
                            <th class="th-wires">index</th>
                            </tr>
                        </thead>
                        <tr>{table_data}</tr>
                    </table>
                </div>
            </div>
            """
        )

    tables_html = "\n".join(wire_tables)
    tables_widget = widgets.HTML(f'<div class="grid-container">{tables_html}</div>')
    return widgets.VBox([style_widget, image_widget, tables_widget])


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
