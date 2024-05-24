"""Visualize timing results from many pytest runs to help identify regressions."""

from collections import defaultdict
from itertools import groupby
from pathlib import Path
from random import random
from typing import Dict, List, Tuple

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

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


# sort duration files (named with epoch timestamps) then load timings

duration_files = sorted(Path(__file__).parent.glob("durations*.txt"))
all_timings = list(map(load_timings, duration_files))

timings_dict: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
for i, timings in enumerate(all_timings):
    for test_name, timing in timings:
        timings_dict[test_name].append((i, timing))

# LineCollection usage taken from: https://stackoverflow.com/a/15773341

my_cmap = plt.get_cmap("jet")
colours = [my_cmap(random()) for _ in timings_dict]
ln_coll = LineCollection(list(timings_dict.values()), colors=colours)

ax = plt.gca()
ax.add_collection(ln_coll)
ax.set_xlim(0, len(all_timings) - 1)
ax.set_ylim(0, 25)
plt.draw()

# test_times: Dict[str, List[float]] = {test_name: [t for _, t in idx_and_timing] for test_name, idx_and_timing in timings_dict.items()}
# """Map from test name to sorted list of timings. Useful for computational analysis."""
