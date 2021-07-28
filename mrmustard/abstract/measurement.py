from abc import ABC
from mrmustard.typing import *
from mrmustard.plugins import FockPlugin, SymplecticPlugin
from mrmustard.abstract import State, Parametrized


class Measurement(ABC, Parametrized):
    r"""
    Base class for measurements. It implements a conditional probability P(out|in) as a stochastic matrix,
    so that an input prob distribution P(in) is transformed to P(out) via belief propagation.
    """
    
    _symplectic: SymplecticPlugin
    _fock: FockPlugin

    def __init__(self, **kwargs):
        # NOTE: call super().__init__() last to avoid overwrites
        self._modes: List[int] = []
        self._stochastic_channel: Optional[Matrix] = None
        super().__init__(**kwargs)
        

    def project(self, state: State, cutoffs: Sequence[int], measurement: Sequence[Optional[int]]) -> State:
        r"""
        Projects the state onto a Fock measurement in the form [a,b,c,...] where integers
        indicate the Fock measurement on that mode and None indicates no projection on that mode.

        Returns the renormalized state (in the Fock basis) in the unmeasured modes
        and the measurement probability.
        """
        if (len(cutoffs) != state.num_modes) or (len(measurement) != state.num_modes):
            raise ValueError("the length of cutoffs/measurements does not match the number of modes")
        dm = state.dm(cutoffs=cutoffs)
        measured = 0
        for mode, (stoch, meas) in enumerate(zip(self._stochastic_channel, measurement)):
            if meas is not None:
                # put both indices last and compute sum_m P(meas|m)rho_mm for every meas
                last = [mode - measured, mode + state.num_modes - 2 * measured]
                perm = list(set(range(dm.ndim)).difference(last)) + last
                dm = self._backend.transpose(dm, perm)
                dm = self._backend.diag(dm)
                dm = self._backend.tensordot(dm, stoch[meas, : dm.shape[-1]], [[-1], [0]], dtype=dm.dtype)
                measured += 1
        prob = self._backend.sum(self._backend.all_diagonals(dm, real=False))
        return dm / prob, self._backend.abs(prob)

    def apply_stochastic_channel(self, fock_probs: State):
        cutoffs = [fock_probs.shape[m] for m in self._modes]
        for i, mode in enumerate(self._modes):
            if cutoffs[mode] > self._stochastic_channel[i].shape[1]:
                raise IndexError(
                    f"This detector does not support so many input photons in mode {mode} (you could increase max_input_photons or reduce the cutoff)"
                )
        detector_probs = fock_probs
        for i, mode in enumerate(self._modes):
            detector_probs = self._backend.tensordot(
                detector_probs,
                self._stochastic_channel[i][: cutoffs[mode], : cutoffs[mode]],
                [[mode], [1]],
            )
            indices = list(range(fock_probs.ndim - 1))
            indices.insert(mode, fock_probs.ndim - 1)
            detector_probs = self._backend.transpose(detector_probs, indices)
        return detector_probs

    def __call__(
        self,
        state: State,
        cutoffs: List[int],
        measurement: Optional[Sequence[Optional[int]]] = None,
    ):  # TODO: this is ugly: the return type should be consistent.
        if measurement is None:
            fock_probs = state.fock_probabilities(cutoffs)
            return self.apply_stochastic_channel(fock_probs)
        else:
            return self.project(state, cutoffs, measurement)



    @property
    def euclidean_parameters(self) -> List:
        return 

    @property
    def symplectic_parameters(self) -> List:
        return self._trainable_parameters
