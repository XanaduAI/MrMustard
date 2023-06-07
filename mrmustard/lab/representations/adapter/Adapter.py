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

from __future__ import annotations
import networkx as nx


class Adapter():

    def __init__(self) -> None:
        self.g = nx.DiGraph()

        # DEFINE NODES
        cp = "CharP"
        gp = "GlauberP"
        hq = "HusimiQ"
        s = "Stellar"
        f = "Fock"
        wx = "WavefunctionX"
        wp = "WavefunctionP"
        cw = "CharW"
        ww = "WignerW"
        cq = "CharQ"


        # DEFINE EDGES - CONNEXIONS
        cp2gp = (cp, gp)
        gp2hq = (gp, hq)
        hq2s = (hq, s)
        s2f = (s, f)
        f2wx = (f, wx)
        wx2wp = (wx, wp)
        cw2ww = (cw, ww)
        ww2hq = (ww, hq)
        cq2hq = (cq, hq)

        edges = [cp2gp, gp2hq, hq2s, s2f, f2wx, wx2wp, cw2ww, ww2hq, cq2hq]


        # DEFINE EDGE LABELS - AKA FORMULAS
        f_cp2gp = "formula from CP 2 GP"
        f_gp2hq = "formula from GP 2 HQ"
        f_hq2s = "formula from HQ 2 S"
        f_s2f = "formula from S 2 F"
        f_f2wx = "formula from F 2 WX"
        f_wx2wp = "formula from WX 2 WP"
        f_cw2ww = "formula from CW 2 WW"
        f_ww2hq = "formula from WW 2 HQ"
        f_cq2hq = "formula from CQ 2 HQ"

        formulas = {cp2gp: {"f":f_cp2gp}, 
                    gp2hq: {"f":f_gp2hq}, 
                    hq2s: {"f":f_hq2s}, 
                    s2f: {"f":f_s2f},
                    f2wx: {"f":f_f2wx},
                    wx2wp: {"f":f_wx2wp},
                    cw2ww: {"f":f_cw2ww},
                    ww2hq: {"f":f_ww2hq},
                    cq2hq: {"f":f_cq2hq}
                    }


        # CREATE GRAPH
        g.add_edges_from(edges)
        nx.set_edge_attributes(g, formulas)

