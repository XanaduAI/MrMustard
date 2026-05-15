# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the state visualization."""

import base64
import json
from pathlib import Path

import numpy as np

from mrmustard import math, settings
from mrmustard.lab.states import Coherent


def plotly_array_spec_to_array(array_spec):
    data_buffer = base64.decodebytes(array_spec["bdata"].encode())
    array = np.frombuffer(data_buffer, dtype=array_spec["dtype"])
    return math.astensor(array)


def assert_plotly_data_close(actual, desired, **kwargs):
    actual_decoded = plotly_array_spec_to_array(actual)
    desired_decoded = plotly_array_spec_to_array(desired)
    np.testing.assert_allclose(actual_decoded, desired_decoded, **kwargs)


class TestVisualization:
    r"""
    Tests the functions to visualize states.
    """

    # set to ``True`` to regenerate the assets
    regenerate_assets = False

    # path
    path = Path(__file__).parent.parent / "assets"

    def test_visualize_2d(self):
        with settings(HBAR=2.0):
            st = Coherent(0, 1j) + Coherent(0, -1j)
            fig = st.visualize_2d(resolution=20, xbounds=(-3, 3), pbounds=(-4, 4), return_fig=True)
        data = fig.to_dict()

        if self.regenerate_assets:
            fig.write_json(self.path / "visualize_2d.json", remove_uids=True)

        with open(self.path / "visualize_2d.json") as file:
            ref_data = json.load(file)

        assert_plotly_data_close(data["data"][0]["x"], ref_data["data"][0]["x"])
        assert_plotly_data_close(data["data"][0]["y"], ref_data["data"][0]["y"])
        assert_plotly_data_close(data["data"][0]["z"], ref_data["data"][0]["z"])
        assert_plotly_data_close(data["data"][1]["x"], ref_data["data"][1]["x"])
        assert_plotly_data_close(data["data"][1]["y"], ref_data["data"][1]["y"])
        assert_plotly_data_close(data["data"][2]["x"], ref_data["data"][2]["x"])
        assert_plotly_data_close(data["data"][2]["y"], ref_data["data"][2]["y"])

    def test_visualize_3d(self):
        with settings(HBAR=2.0):
            st = Coherent(0, 1j) + Coherent(0, -1j)
            fig = st.visualize_3d(resolution=20, xbounds=(-3, 3), pbounds=(-4, 4), return_fig=True)
        data = fig.to_dict()

        if self.regenerate_assets:
            fig.write_json(self.path / "visualize_3d.json", remove_uids=True)

        with open(self.path / "visualize_3d.json") as file:
            ref_data = json.load(file)

        assert_plotly_data_close(data["data"][0]["x"], ref_data["data"][0]["x"])
        assert_plotly_data_close(data["data"][0]["y"], ref_data["data"][0]["y"])
        assert_plotly_data_close(data["data"][0]["z"], ref_data["data"][0]["z"])

    def test_visualize_dm(self):
        st = Coherent(0, 1j) + Coherent(0, -1j)
        st.manual_shape = (20,)
        fig = st.visualize_dm(20, return_fig=True)
        data = fig.to_dict()

        if self.regenerate_assets:
            fig.write_json(self.path / "visualize_dm.json", remove_uids=True)

        with open(self.path / "visualize_dm.json") as file:
            ref_data = json.load(file)
        assert_plotly_data_close(data["data"][0]["z"], ref_data["data"][0]["z"])
