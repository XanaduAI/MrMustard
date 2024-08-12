# copyright: 2024 Xanadu

from typing import Sequence
from mrmustard.lab_dev.transformations.base import Map
from mrmustard.physics.representations import Bargmann
from mrmustard.physics import triples


class CFT(Map):
    r"""The Complex Fourier Transformation as a channel.
    The main use is to convert between Characteristic functions and phase space functions.

    Args:
        num_modes: number of modes of this channel.
    """

    def __init__(
        self,
        modes: Sequence[int],
    ):
        super().__init__(
            modes_out=modes,
            modes_in=modes,
            name="CFT",
        )
        self._representation = Bargmann.from_function(
            fn=triples.complex_fourier_transform_Abc, n_modes=len(modes)
        )
