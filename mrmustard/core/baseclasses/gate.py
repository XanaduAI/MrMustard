from abc import ABC


class Gate(ABC):
    _math_backend: MathBackendInterface
    _gate_backend: GateBackendInterface

    def _apply_gaussian_channel(self, state, modes, symplectic=None, displacement=None, noise=None):
        output = State(state.num_modes, hbar=state.hbar, mixed=noise is not None)
        output.cov = self._math_backend.sandwich(bread=symplectic, filling=state.cov, modes=modes)
        output.cov = self._math_backend.add(old=output.cov, new=noise, modes=modes)
        output.means = self._math_backend.matvec(mat=symplectic, vec=state.means, modes=modes)
        output.means = self._math_backend.add(old=output.means, new=displacement, modes=modes)
        return output

    def __call__(self, state: State) -> State:
        return self._apply_gaussian_channel(
            state,
            self.modes,
            self.symplectic_matrix(state.hbar),
            self.displacement_vector(state.hbar),
            self.noise_matrix(state.hbar),
        )

    def __repr__(self):
        with np.printoptions(precision=3, suppress=True):
            return f"{self.__class__.__qualname__}({self._repr_string(*[str(np.atleast_1d(p)) for p in self._parameters])})"

    def symplectic_matrix(self, hbar: float) -> Optional:
        return None

    def displacement_vector(self, hbar: float) -> Optional:
        return None

    def noise_matrix(self, hbar: float) -> Optional:
        return None

    @property
    def euclidean_parameters(self) -> List:
        return [
            p for i, p in enumerate(self._parameters) if self._trainable[i]
        ]  # NOTE overridden in Ggate

    @property
    def symplectic_parameters(self) -> List:
        return []