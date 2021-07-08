from typing import List, Union, Sequence, Optional, Tuple
from mrmustard.core.baseclasses.parametrized import Parametrized
from mrmustard.core.baseclasses import Detector


class PNRDetector(Parametrized, Detector):
    r"""
    Photon Number Resolving detector. If len(modes) > 1 the detector is applied in parallel to all of the modes provided.
    If a parameter is a single float, the parallel instances of the detector share that parameter.
    To apply mode-specific parmeters use a list of floats.
    One can optionally set bounds for each parameter, which the optimizer will respect.
    It can be supplied the full conditional detection probabilities, or it will compute them from
    the quantum efficiency (binomial) and the dark count probability (possonian).
    Arguments:
        conditional_probs (Optional 2d array): if supplied, these probabilities will be used for belief propagation
        efficiency (float or List[float]): list of quantum efficiencies for each detector
        dark_counts (float or List[float]): list of expected dark counts
        max_cutoffs (int or List[int]): largest Fock space cutoffs that the detector should expect
    """

    def __init__(
        self,
        modes: List[int],
        conditional_probs=None,
        efficiency: Union[float, List[float]] = 1.0,
        efficiency_trainable: bool = False,
        efficiency_bounds: Tuple[Optional[float], Optional[float]] = (0.0, 1.0),
        dark_counts: Union[float, List[float]] = 0.0,
        dark_counts_trainable: bool = False,
        dark_counts_bounds: Tuple[Optional[float], Optional[float]] = (0.0, None),
        max_cutoffs: Union[int, List[int]] = 50,
    ):
        if not isinstance(max_cutoffs, Sequence):
            max_cutoffs = [max_cutoffs for m in modes]
        if not isinstance(efficiency, Sequence):
            efficiency = [efficiency for m in modes]
        if not isinstance(dark_counts, Sequence):
            dark_counts = [dark_counts for m in modes]

        super().__init__(
            modes=modes,
            conditional_probs=conditional_probs,
            efficiency=efficiency,
            efficiency_trainable=efficiency_trainable,
            efficiency_bounds=efficiency_bounds,
            dark_counts=dark_counts,
            dark_counts_trainable=dark_counts_trainable,
            dark_counts_bounds=dark_counts_bounds,
            max_cutoffs=max_cutoffs,
        )

        self.make_stochastic_channel()

    def make_stochastic_channel(self):
        self._stochastic_channel = []
        if self._conditional_probs is not None:
            self._stochastic_channel = self._conditional_probs
        else:
            for cut, qe, dc in zip(self._max_cutoffs, self.efficiency[:], self.dark_counts[:]):
                dark_prior = self._math_backend.poisson(max_k=cut, rate=dc)
                condprob = self._math_backend.binomial_conditional_prob(success_prob=qe, dim_in=cut, dim_out=cut)
                self._stochastic_channel.append(
                    self._math_backend.convolve_probs_1d(condprob, [dark_prior, self._math_backend.identity(condprob.shape[1])[0]])
                )


class ThresholdDetector(Parametrized, Detector):
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
        efficiency_trainable: bool = False,
        efficiency_bounds: Tuple[Optional[float], Optional[float]] = (0.0, 1.0),
        dark_count_prob: Union[float, List[float]] = 0.0,
        dark_count_prob_trainable: bool = False,
        dark_count_prob_bounds: Tuple[Optional[float], Optional[float]] = (0.0, None),
        max_cutoffs: Union[int, List[int]] = 50,
    ):
        if not isinstance(max_cutoffs, Sequence):
            max_cutoffs = [max_cutoffs for m in modes]

        super().__init__(
            modes=modes,
            conditional_probs=conditional_probs,
            efficiency=efficiency,
            efficiency_trainable=efficiency_trainable,
            efficiency_bounds=efficiency_bounds,
            dark_count_prob=dark_count_prob,
            dark_count_prob_trainable=dark_count_prob_trainable,
            dark_count_prob_bounds=dark_count_prob_bounds,
            max_cutoffs=max_cutoffs,
        )

        self.make_stochastic_channel()

    def make_stochastic_channel(self):
        self._stochastic_channel = []

        if self._conditional_probs is not None:
            self._stochastic_channel = self.conditional_probs
        else:
            for cut, qe, dc in zip(self._max_cutoffs, self.efficiency[:], self.dark_count_probs[:]):
                row1 = ((1.0 - qe) ** self._math_backend.arange(cut))[None, :] - dc
                row2 = 1.0 - row1
                rest = self._math_backend.zeros((cut - 2, cut), dtype=row1.dtype)
                condprob = self._math_backend.concat([row1, row2, rest], axis=0)
                self._stochastic_channel.append(condprob)
