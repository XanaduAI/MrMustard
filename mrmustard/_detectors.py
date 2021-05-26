from typing import List, Union, Sequence, Optional, Tuple
from mrmustard._backends import MathBackendInterface
from mrmustard._states import State


class Detector:
    r"""
    Base class for photon detectors. It implements a conditional probability P(out|in) as a stochastic matrix,
    so that an input prob distribution P(in) is transformed to P(out) via belief propagation.
    """
    _math_backend: MathBackendInterface

    def __init__(self, modes: List[int]):
        self.modes = modes

    def project(self, state: State, cutoffs: Sequence[int], measurements: Sequence[Optional[int]]) -> State:
        r"""
        Projects the state onto a Fock measurement in the form [a,b,c,...] where integers
        indicate the Fock measurement on that mode and None indicates no projection on that mode.

        Returns the renormalized state (in the Fock basis) in the unmeasured modes
        and the measurement probability.
        """
        if (len(cutoffs) != state.num_modes) or (len(measurements) != state.num_modes):
            raise ValueError('the length of cutoffs and/or measurements does not match the number of modes')
        measurement = [m if m is not None else slice(None) for m in measurements]
        if state.mixed:
            measurement = measurement + measurement
            dm = state.dm(cutoffs=cutoffs)[measurement]
            prob = self._math_backend.trace(dm)
            return dm / prob, prob
        else:
            ket = state.ket(cutoffs=cutoffs)[measurement]
            norm = self._math_backend.norm(ket)
            return ket / norm, self._math_backend.abs(norm)**2

    def __call__(self, state: State, cutoffs: List[int]):
        fock_probs = state.fock_probabilities(cutoffs)
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

    @property
    def euclidean_parameters(self) -> List:
        return [p for i, p in enumerate(self._parameters) if self._trainable[i]]


class PNRDetector(Detector):
    r"""
    Photon Number Resolving detector. If len(modes) > 1 the detector is applied in parallel to all of the modes provided.
    If a parameter is a single float, its value is applied to all of the parallel instances of the detector.
    To apply mode-specific values use a list of floats.
    It can be supplied the full conditional detection probabilities, or it will compute them from
    the quantum efficiency (binomial) and the dark count probability (possonian).
    Arguments:
        conditional_probs (Optional 2d array): if supplied, these probabilities will be used for belief propagation
        quantum_efficiency (float or List[float]): list of quantum efficiencies for each detector
        expected_dark_counts (float or List[float]): list of expected dark counts
        max_cutoffs (int or List[int]): largest Fock space cutoffs that the detector should expect
    """

    def __init__(
        self,
        modes: List[int],
        conditional_probs=None,
        quantum_efficiency: Union[float, List[float]] = 1.0,
        quantum_efficiency_trainable: bool = False,
        quantum_efficiency_bounds: Tuple[Optional[float], Optional[float]] = (0.0, 1.0),
        expected_dark_counts: Union[float, List[float]] = 0.0,
        expected_dark_counts_trainable: bool = False,
        expected_dark_counts_bounds: Tuple[Optional[float], Optional[float]] = (0.0, None),
        max_cutoffs: Union[int, List[int]] = 50,
    ):
        super().__init__(modes)

        self._trainable = [quantum_efficiency_trainable, expected_dark_counts_trainable]
        self._parameters = [
            self._math_backend.make_euclidean_parameter(
                quantum_efficiency,
                quantum_efficiency_trainable,
                quantum_efficiency_bounds,
                (len(modes),),
                "quantum_efficiency",
            ),
            self._math_backend.make_euclidean_parameter(
                expected_dark_counts,
                expected_dark_counts_trainable,
                expected_dark_counts_bounds,
                (len(modes),),
                "expected_dark_counts",
            ),
        ]
        if not isinstance(max_cutoffs, Sequence):
            max_cutoffs = [max_cutoffs for m in modes]
        self.quantum_efficiency = self._parameters[0]
        self.expected_dark_counts = self._parameters[1]
        self.max_cutoffs = max_cutoffs
        self.conditional_probs = conditional_probs
        self.make_stochastic_channel()

    def make_stochastic_channel(self):
        self._stochastic_channel = []
        if self.conditional_probs is not None:
            self._stochastic_channel = self.conditional_probs
        else:
            for cut, qe, dc in zip(
                self.max_cutoffs, self.quantum_efficiency[:], self.expected_dark_counts[:]
            ):
                dark_prior = self._math_backend.poisson(max_k=cut, rate=dc)
                condprob = self._math_backend.binomial_conditional_prob(
                    success_prob=qe, dim_in=cut, dim_out=cut
                )
                self._stochastic_channel.append(
                    self._math_backend.convolve_probs_1d(
                        condprob, [dark_prior, self._math_backend.identity(condprob.shape[1])[0]]
                    )
                )


class ThresholdDetector(Detector):
    r"""
    Threshold detector: any state with more photons than vacuum is detected as a single photon.
    If len(modes) > 1 the detector is applied in parallel to all of the modes provided.
    If a parameter is a single float, its value is applied to all of the parallel instances of the detector.
    To apply mode-specific values use a list of floats.
    It can be supplied the full conditional detection probabilities, or it will compute them from
    the quantum efficiency (binomial) and the dark count probability (bernoulli).
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
        quantum_efficiency_trainable: bool = False,
        quantum_efficiency_bounds: Tuple[Optional[float], Optional[float]] = (0.0, 1.0),
        expected_dark_probs: Union[float, List[float]] = 0.0,
        expected_dark_probs_trainable: bool = False,
        expected_dark_probs_bounds: Tuple[Optional[float], Optional[float]] = (0.0, None),
        max_cutoffs: Union[int, List[int]] = 50,
    ):
        super().__init__(modes)

        self._trainable = [quantum_efficiency_trainable, expected_dark_probs_trainable]
        self._parameters = [
            self._math_backend.make_euclidean_parameter(
                quantum_efficiency,
                quantum_efficiency_trainable,
                quantum_efficiency_bounds,
                (len(modes),),
                "quantum_efficiency",
            ),
            self._math_backend.make_euclidean_parameter(
                expected_dark_probs,
                expected_dark_probs_trainable,
                expected_dark_probs_bounds,
                (len(modes),),
                "expected_dark_counts",
            ),
        ]
        if not isinstance(max_cutoffs, Sequence):
            max_cutoffs = [max_cutoffs for m in modes]
        self.quantum_efficiency = self._parameters[0]
        self.expected_dark_counts = self._parameters[1]
        self.max_cutoffs = max_cutoffs
        self.conditional_probs = conditional_probs
        self.make_stochastic_channel()

    def make_stochastic_channel(self):
        self._stochastic_channel = []

        if self.conditional_probs is not None:
            self._stochastic_channel = self.conditional_probs
        else:
            for cut, qe, dc in zip(
                self.max_cutoffs, self.quantum_efficiency[:], self.dark_count_probs[:]
            ):
                row1 = ((1.0 - qe) ** self._math_backend.arange(cut))[None, :] - dc
                row2 = 1.0 - row1
                rest = self._math_backend.zeros((cut - 2, cut), dtype=row1.dtype)
                condprob = self._math_backend.concat([row1, row2, rest], axis=0)
                self._stochastic_channel.append(condprob)
