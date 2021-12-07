# Copyright 2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from mrmustard.types import *
from mrmustard.utils.parametrized import Parametrized
from mrmustard.lab.abstract import State, FockMeasurement, GaussianMeasurement
from mrmustard.physics import fock, gaussian
from mrmustard.lab.states import DisplacedSqueezed, Coherent

__all__ = ["PNRDetector", "ThresholdDetector", "Homodyne", "Heterodyne", "Generaldyne"]


class PNRDetector(Parametrized, FockMeasurement):
    r"""
    Photon Number Resolving detector. If len(modes) > 1 the detector is applied in parallel to all of the modes provided.
    If a parameter is a single float, the parallel instances of the detector share that parameter.
    To apply mode-specific parmeters use a list of floats.
    One can optionally set bounds for each parameter, which the optimizer will respect.
    It can be supplied the full conditional detection probabilities, or it will compute them from
    the quantum efficiency (binomial) and the dark count probability (possonian).
    Arguments:
        efficiency (float or List[float]): list of quantum efficiencies for each detector
        efficiency_trainable (bool): whether the efficiency is trainable
        efficiency_bounds (Tuple[float, float]): bounds for the efficiency
        dark_counts (float or List[float]): list of expected dark counts
        dark_counts_trainable (bool): whether the dark counts are trainable
        dark_counts_bounds (Tuple[float, float]): bounds for the dark counts
        max_cutoffs (int or List[int]): largest Fock space cutoffs that the detector should expect
        conditional_probs (Optional 2d array): if supplied, these probabilities will be used for belief propagation
        modes (Optional List[int]): list of modes to apply the detector to
    """

    def __init__(
        self,
        efficiency: Union[float, List[float]] = 1.0,
        dark_counts: Union[float, List[float]] = 0.0,
        efficiency_trainable: bool = False,
        dark_counts_trainable: bool = False,
        efficiency_bounds: Tuple[Optional[float], Optional[float]] = (0.0, 1.0),
        dark_counts_bounds: Tuple[Optional[float], Optional[float]] = (0.0, None),
        max_cutoffs: Union[int, List[int]] = 50,  # TODO: make this a parameter in mm.settings
        conditional_probs=None,
        modes: List[int] = None,
    ):
        if not isinstance(max_cutoffs, Sequence):
            max_cutoffs = [max_cutoffs for m in modes]
        if not isinstance(efficiency, Sequence):
            efficiency = [efficiency for m in modes]
        if not isinstance(dark_counts, Sequence):
            dark_counts = [dark_counts for m in modes]

        Parametrized.__init__(
            self,
            efficiency=efficiency,
            dark_counts=dark_counts,
            efficiency_trainable=efficiency_trainable,
            dark_counts_trainable=dark_counts_trainable,
            efficiency_bounds=efficiency_bounds,
            dark_counts_bounds=dark_counts_bounds,
            max_cutoffs=max_cutoffs,
            conditional_probs=conditional_probs,
            modes=modes,
        )

        self.recompute_stochastic_channel()

    def recompute_stochastic_channel(self):
        self._stochastic_channel = []
        if self._conditional_probs is not None:
            self._stochastic_channel = self._conditional_probs
        else:
            for cut, qe, dc in zip(self._max_cutoffs, self.efficiency[:], self.dark_counts[:]):
                dark_prior = fock.math.poisson(max_k=cut, rate=dc)
                condprob = fock.math.binomial_conditional_prob(
                    success_prob=qe, dim_in=cut, dim_out=cut
                )
                self._stochastic_channel.append(
                    fock.math.convolve_probs_1d(
                        condprob, [dark_prior, fock.math.eye(condprob.shape[1])[0]]
                    )
                )



class ThresholdDetector(Parametrized, FockMeasurement):
    r"""
    Threshold detector: any Fock component other than vacuum counts toward a click in the detector.
    If len(modes) > 1 the detector is applied in parallel to all of the modes provided.
    If a parameter is a single float, its value is applied to all of the parallel instances of the detector.
    To apply mode-specific values use a list of floats.
    It can be supplied the full conditional detection probabilities, or it will compute them from
    the quantum efficiency (binomial) and the dark count probability (bernoulli).
    Arguments:
        conditional_probs (Optional 2d array): if supplied, these probabilities will be used for belief propagation
        efficiency (float or List[float]): list of quantum efficiencies for each detector
        dark_count_prob (float or List[float]): list of dark count probabilities for each detector
        max_cutoffs (int or List[int]): largest Fock space cutoffs that the detector should expect
    """

    def __init__(
        self,
        modes: List[int],
        conditional_probs=None,
        efficiency: Union[float, List[float]] = 1.0,
        dark_count_prob: Union[float, List[float]] = 0.0,
        efficiency_trainable: bool = False,
        dark_count_prob_trainable: bool = False,
        efficiency_bounds: Tuple[Optional[float], Optional[float]] = (0.0, 1.0),
        dark_count_prob_bounds: Tuple[Optional[float], Optional[float]] = (0.0, None),
        max_cutoffs: Union[int, List[int]] = 50,
    ):
        if not isinstance(max_cutoffs, Sequence):
            max_cutoffs = [max_cutoffs for m in modes]
        if not isinstance(efficiency, Sequence):
            efficiency = [efficiency for m in modes]
        if not isinstance(dark_count_prob, Sequence):
            dark_count_prob = [dark_count_prob for m in modes]

        Parametrized.__init__(
            self,
            modes=modes,
            conditional_probs=conditional_probs,
            efficiency=efficiency,
            dark_count_prob=dark_count_prob,
            efficiency_trainable=efficiency_trainable,
            dark_count_prob_trainable=dark_count_prob_trainable,
            efficiency_bounds=efficiency_bounds,
            dark_count_prob_bounds=dark_count_prob_bounds,
            max_cutoffs=max_cutoffs,
        )

        self.recompute_stochastic_channel()

    def recompute_stochastic_channel(self):
        self._stochastic_channel = []
        if self._conditional_probs is not None:
            self._stochastic_channel = self.conditional_probs
        else:
            for cut, qe, dc in zip(self._max_cutoffs, self.efficiency[:], self.dark_count_prob[:]):
                row1 = ((1.0 - qe) ** fock.math.arange(cut))[None, :] - dc
                row2 = 1.0 - row1
                rest = fock.math.zeros((cut - 2, cut), dtype=row1.dtype)
                condprob = fock.math.concat([row1, row2, rest], axis=0)
                self._stochastic_channel.append(condprob)

    @property
    def stochastic_channel(self) -> List[Matrix]:
        if self._stochastic_channel is None:
            self._stochastic_channel = fock.stochastic_channel()
        return self._stochastic_channel


class Homodyne(Parametrized, State):
    r"""
    Heterodyne measurement on given modes.
    """
    def __new__(cls,
            quadrature_angles=quadrature_angles,
            results=results,
            modes=modes):
        quadrature_angles = gaussian.math.astensor(quadrature_angles, dtype="float64")
        results = gaussian.math.astensor(results, dtype="float64")
        x = results * gaussian.math.cos(quadrature_angles)
        y = results * gaussian.math.sin(quadrature_angles)
        instance = DisplacedSqueezed(r=settings.HOMODYNE_SQUEEZING, phi=2*quadrature_angles, x=x, y=y)
        instance.__class__ = cls
        return instance

    def __init__(self, *args, **kwargs):
        pass


class Heterodyne(Parametrized, State):
    r"""
    Heterodyne measurement on given modes.
    """
    def __new__(cls, x, y, modes=None):
        instance = Coherent(x=x, y=y, modes=modes)
        instance.__class__ = cls
        return instance

    def __init__(self, *args, **kwargs):
        pass
