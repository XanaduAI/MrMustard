from typing import List, Union, Sequence
from mrmustard._backends import MathBackendInterface, utils
from scipy.stats import poisson
from mrmustard._states import State


class Detector:
    r"""
    Base class for photon detectors. It implements a conditional probability P(out|in) as a stochastic matrix,
    so that an input prob distribution P(in) is transformed to P(out) via belief propagation.
    """
    _math_backend: MathBackendInterface

    def __init__(self, modes: List[int]):
        self.modes = modes

    def apply_stochastic_channel(self, fock_probs):
        cutoffs = [fock_probs.shape[m] for m in self.modes]
        for i, mode in enumerate(self.modes):
            if cutoffs[mode] > self._stochastic_channel[i].shape[1]:
                raise IndexError(
                    f"This detector does not support so many input photons in mode {mode} (you could increase max_input_photons or reduce the cutoff)"
                )
        detector_probs = fock_probs
        for i, mode in enumerate(self.modes):
            detector_probs = self._math_backend.tensordot(
                detector_probs,
                self._stochastic_channel[i][: cutoffs[mode], : cutoffs[mode]],
                [[mode], [1]],
            )
            indices = list(range(fock_probs.ndim - 1))
            indices.insert(mode, fock_probs.ndim - 1)
            detector_probs = self._math_backend.transpose(detector_probs, indices)
        return detector_probs

    def __call__(self, state: State, cutoffs: List[int]):
        return self.apply_stochastic_channel(state.fock_probabilities(cutoffs))


class PNR(Detector):
    r"""
    Photon Number Resolving detector. If len(modes) > 1 the detector is applied in parallel to all of the modes provided.
    If a parameter is a single float, its value is applied to all of the parallel instances of the detector.
    To apply mode-specific values use a list of floats.
    It can be supplied the full conditional detection probabilities, or it will compute them from
    the quantum efficiency (binomial) and the dark count probability (possonian).
    Arguments:
        conditional_probs (Optional 2d array): if supplied, these probabilities will be used for belief propagation
        quantum_efficiency (float or List[float]): list of quantum efficiencies for each detector
        dark_count_prob (float or List[float]): list of dark count probabilities for each detector
        max_cutoffs (int or List[int]): largest Fock space cutoffs that the detector should expect
    """

    def __init__(
        self,
        modes: List[int],
        conditional_probs=None,
        quantum_efficiency: Union[float, List[float]] = 1.0,
        dark_count_prob: Union[float, List[float]] = 0.0,
        max_cutoffs: Union[int, List[int]] = 50,
    ):
        super().__init__(modes)

        if not isinstance(quantum_efficiency, Sequence):
            quantum_efficiency = [quantum_efficiency for m in modes]
        if not isinstance(dark_count_prob, Sequence):
            dark_count_prob = [dark_count_prob for m in modes]
        if not isinstance(max_cutoffs, Sequence):
            max_cutoffs = [max_cutoffs for m in modes]
        self.quantum_efficiency = quantum_efficiency
        self.dark_count_prob = dark_count_prob
        self.max_cutoffs = max_cutoffs
        self._stochastic_channel = []

        if conditional_probs is not None:
            self._stochastic_channel = conditional_probs
        else:
            for cut, qe, dc in zip(self.max_cutoffs, quantum_efficiency, dark_count_prob):
                dark_prior = poisson.pmf(self._math_backend.arange(cut), dc)
                condprob = utils.binomial_conditional_prob(success_prob=qe, dim_in=cut, dim_out=cut)
                self._stochastic_channel.append(
                    self._math_backend.convolve_probs_1d(
                        condprob, [dark_prior, self._math_backend.identity(condprob.shape[1])[0]]
                    )
                )


class ThresholdDetector(Detector):
    r"""
    Threshold detector: any Fock component other than vacuum counts toward a click in the detector.
    If len(modes) > 1 the detector is applied in parallel to all of the modes provided.
    If a parameter is a single float, its value is applied to all of the parallel instances of the detector.
    To apply mode-specific values use a list of floats.
    It can be supplied the full conditional detection probabilities, or it will compute them from
    the quantum efficiency (binomial) and the dark count probability (possonian).
    Arguments:
        conditional_probs (Optional 2d array): if supplied, these probabilities will be used for belief propagation
        quantum_efficiency (float or List[float]): list of quantum efficiencies for each detector
        dark_count_prob (float or List[float]): list of dark count probabilities for each detector
        max_cutoffs (int or List[int]): largest Fock space cutoffs that the detector should expect
    """

    pass
