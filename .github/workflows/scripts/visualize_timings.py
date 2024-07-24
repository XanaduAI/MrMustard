"""Visualize timing results from many pytest runs to help identify regressions."""

from argparse import ArgumentParser
from collections import defaultdict
from itertools import groupby
from pathlib import Path
from random import random
from typing import Dict, List, Tuple

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

aggregates = (
    "tests/test_physics/test_fidelity.py::TestGaussianStates::test_fidelity_vac_to_displaced_squeezed",
)


def aggregate_keyfunc(name_and_time):
    """
    This is a ``groupby`` key function. If a test name
    is in the aggregates tuple, then we track the sum of
    all parametrized cases of the test.

    This is needed for tests with random parameters, otherwise
    the data for that test will be useless.
    """
    name, _ = name_and_time
    non_param_test_name = name.split("[")[0]
    return non_param_test_name if non_param_test_name in aggregates else name


def parse_time(val):
    """Convert a string like '1.23s' into the float 1.23"""
    assert val[-1] == "s"
    return float(val[:-1])


def load_timings(file: Path):
    """
    Given a file generated using ``pytest --durations=0 -vv``, return
    a list of tuples of the form ``(test_name, test_duration_secs)``.
    """
    timings = [line.split() for line in file.read_text().split("\n")[:-1]]
    assert all(len(t) == 2 for t in timings)

    # now aggregate for tests known to have random params
    # sort by test name or aggregation will fail
    timings.sort(key=lambda i: i[0])
    return [
        (test_name, sum(parse_time(t) for _, t in list(group)))
        for test_name, group in groupby(timings, key=aggregate_keyfunc)
    ]


def draw_mpl(timings_dict, ncols, num_commits):
    """
    Draw using the Matplotlib backend.

    LineCollection usage taken from: https://stackoverflow.com/a/15773341
    """
    groups = {
        group_name: list(group)
        for group_name, group in groupby(
            sorted(timings_dict), key=lambda x: x.split("/")[1]
        )
    }

    fig = plt.figure(1)
    my_cmap = plt.get_cmap("jet")

    tot = len(groups)
    cols = ncols
    rows = sum(divmod(tot, cols))

    for idx, (group, test_names) in enumerate(groups.items()):
        ax = fig.add_subplot(rows, cols, idx + 1)
        ax.set_title(group)
        colours = [my_cmap(random()) for _ in test_names]
        values = [timings_dict[n] for n in test_names]
        ln_coll = LineCollection(values, colors=colours)
        ax.add_collection(ln_coll)
        ax.set_xlim(0, num_commits - 1)
        ax.set_ylim(0, max(max(v[1] for v in value_set) for value_set in values) + 0.01)

    fig.tight_layout()
    fig.set_dpi(200)
    plt.show()


def draw_plotly(timings_dict, use_short_name):
    """Draw using the plotly backend."""
    layout = go.Layout(
        title="Test durations over history of commits",
        xaxis={"title": "Commit"},
        yaxis={"title": "Test Duration (s)"},
    )
    lines = []
    timings_sorted = dict(
        sorted(
            timings_dict.items(),
            key=lambda item: max(val[1] for val in item[1]),
            reverse=True,
        )
    )
    for test_name, data in timings_sorted.items():
        x, y = list(zip(*data))
        if use_short_name:
            test_name = test_name.split("::")[-1]
        lines.append(go.Scatter(x=x, y=y, mode="lines", name=test_name))
    fig = go.Figure(data=lines, layout=layout)

    fig.show()
    return fig


def draw_plotly_group(timings_dict, use_short_name, ncols):
    """Draw using the plotly backend, grouping by test module."""

    groups = {
        group_name: list(group)
        for group_name, group in groupby(
            sorted(timings_dict), key=lambda x: x.split("/")[1]
        )
    }
    tot = len(groups)
    cols = ncols
    rows = sum(divmod(tot, cols))
    fig = make_subplots(rows, cols, subplot_titles=list(groups))
    fig.update_layout({"title": "Test durations over history of commits"})

    for (row, col), (group, test_names) in zip(
        np.ndindex((rows, cols)), groups.items()
    ):
        for test_name in test_names:
            x, y = list(zip(*timings_dict[test_name]))
            if use_short_name:
                test_name = test_name.split("::")[-1]
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name=test_name,
                    legendgroup=group,
                    legendgrouptitle_text=group,
                ),
                row=row + 1,
                col=col + 1,
            )

    fig.show()
    return fig


def remove_n_largest_plotly(fig, n):
    """Hide the first ``n`` traces in a plotly figure."""
    fig.update_traces(visible=None)  # reset in case some were hidden before
    for i in range(-1, -1 - n, -1):
        fig.update_traces(selector=i, visible="legendonly")
    fig.show()


def main(data_folder, mode, ncols, use_short_name):
    """Load durations by commit, then draw them using the appropriate backend."""
    # sort duration files (named with epoch timestamps) then load timings
    duration_files = sorted(data_folder.glob("durations_*.txt"))
    all_timings = list(map(load_timings, duration_files))

    timings_dict: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for i, commit_timings in enumerate(all_timings):
        for test_name, timing in commit_timings:
            timings_dict[test_name].append((i, timing))

    # delete test_about and any tests that are all-zero
    to_delete = {"tests/test_about.py::test_about"}
    for test_name, commit_and_times in timings_dict.items():
        if all(t == 0 for _, t in commit_and_times):
            to_delete.add(test_name)

    for test_name in to_delete:
        del timings_dict[test_name]

    if mode == "mpl":
        draw_mpl(timings_dict, ncols, len(all_timings))
    elif mode == "plotly":
        draw_plotly(timings_dict, use_short_name)
    elif mode == "plotly-grouped":
        draw_plotly_group(timings_dict, use_short_name, ncols)

    # test_times: Dict[str, List[float]] = {test_name: [t for _, t in idx_and_timing] for test_name, idx_and_timing in timings_dict.items()}
    # """Map from test name to sorted list of timings. Useful for computational analysis."""


if __name__ == "__main__":
    parser = ArgumentParser(
        usage=(
            "visualize_timings.py [-h] data_folder [--mode <mode>] [--ncols <NCOLS>] [--short-name]"
            "\n\n1. Sync files from S3 using the AWS CLI. For example:\n\t"
            "aws s3 sync s3://top-secret-bucket-name/numpy_tests/develop/ /path/to/local/folder/\n"
            "2. Run this script:\n\t"
            "python .github/workflows/scripts/visualize_timings.py /path/to/local/folder"
        ),
        description="Visualize pytest duration data to detect regressions.",
    )
    parser.add_argument(
        "data_folder", type=str, help="Folder where duration files are synced"
    )
    parser.add_argument(
        "-m",
        "--mode",
        required=False,
        default="plotly",
        choices=["plotly", "mpl", "plotly-grouped"],
        help="The plotting backend to use",
    )
    parser.add_argument(
        "--ncols",
        default=3,
        required=False,
        type=int,
        help="Number of modules per row (does nothing in the default 'plotly' mode)",
    )
    parser.add_argument(
        "--short-name",
        action="store_true",
        help="Show only the test name without the file path",
    )
    args = parser.parse_args()
    folder = Path(args.data_folder)
    main(folder, args.mode, args.ncols, args.short_name)
