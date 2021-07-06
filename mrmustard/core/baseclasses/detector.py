from abc import ABC
from typing import List, Sequence, Optional
from mrmustard.core.backends import MathBackendInterface
from mrmustard.core.baseclasses.state import State


class Detector(ABC):
    r"""
    Base class for photon detectors. It implements a conditional probability P(out|in) as a stochastic matrix,
    so that an input prob distribution P(in) is transformed to P(out) via belief propagation.
    """
    _math_backend: MathBackendInterface

    def project(self, state: State, cutoffs: Sequence[int], measurements: Sequence[Optional[int]]) -> State:
        r"""
        Projects the state onto a Fock measurement in the form [a,b,c,...] where integers
        indicate the Fock measurement on that mode and None indicates no projection on that mode.

        Returns the renormalized state (in the Fock basis) in the unmeasured modes
        and the measurement probability.
        """
        if (len(cutoffs) != state.num_modes) or (len(measurements) != state.num_modes):
            raise ValueError("the length of cutoffs/measurements does not match the number of modes")
        dm = state.dm(cutoffs=cutoffs)
        measured = 0
        for mode, (stoch, meas) in enumerate(zip(self._stochastic_channel, measurements)):
            if meas is not None:
                # put both indices last and compute sum_m P(meas|m)rho_mm for every meas
                last = [mode - measured, mode + state.num_modes - 2 * measured]
                perm = list(set(range(dm.ndim)).difference(last)) + last
                dm = self._math_backend.transpose(dm, perm)
                dm = self._math_backend.diag(dm)
                dm = self._math_backend.tensordot(
                    dm, stoch[meas, : dm.shape[-1]], [[-1], [0]], dtype=dm.dtype
                )
                measured += 1
        prob = self._math_backend.sum(self._math_backend.all_diagonals(dm, real=False))
        return dm / prob, self._math_backend.abs(prob)

    def apply_stochastic_channel(self, fock_probs: State):
        cutoffs = [fock_probs.shape[m] for m in self._modes]
        for i, mode in enumerate(self._modes):
            if cutoffs[mode] > self._stochastic_channel[i].shape[1]:
                raise IndexError(
                    f"This detector does not support so many input photons in mode {mode} (you could increase max_input_photons or reduce the cutoff)"
                )
        detector_probs = fock_probs
        for i, mode in enumerate(self._modes):
            detector_probs = self._math_backend.tensordot(
                detector_probs,
                self._stochastic_channel[i][: cutoffs[mode], : cutoffs[mode]],
                [[mode], [1]],
            )
            indices = list(range(fock_probs.ndim - 1))
            indices.insert(mode, fock_probs.ndim - 1)
            detector_probs = self._math_backend.transpose(detector_probs, indices)
        return detector_probs

    def __call__(
        self,
        state: State,
        cutoffs: List[int],
        measurements: Optional[Sequence[Optional[int]]] = None,
    ):
        if measurements is None:
            fock_probs = state.fock_probabilities(cutoffs)
            return self.apply_stochastic_channel(fock_probs)
        else:
            return self.project(state, cutoffs, measurements)

    @property
    def euclidean_parameters(self) -> List:
        return self._trainable_parameters
