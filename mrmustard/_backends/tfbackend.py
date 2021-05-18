import numpy as np
import tensorflow as tf
from scipy.stats import unitary_group, truncnorm
from scipy.linalg import expm
from typing import List, Tuple, Callable, Sequence, Optional
from itertools import product

from mrmustard._gates import GateBackendInterface
from mrmustard._opt import OptimizerBackendInterface
from mrmustard._circuit import CircuitBackendInterface
from mrmustard._backends import MathBackendInterface
from mrmustard._states import StateBackendInterface
from mrmustard._detectors import DetectorBackendInterface

import mrmustard._backends.utils as utils


#          CCCCCCCCCCCCC  iiii                                                             iiii          tttt
#       CCC::::::::::::C i::::i                                                           i::::i      ttt:::t
#     CC:::::::::::::::C  iiii                                                             iiii       t:::::t
#    C:::::CCCCCCCC::::C                                                                              t:::::t
#   C:::::C       CCCCCCiiiiiiirrrrr   rrrrrrrrr       ccccccccccccccccuuuuuu    uuuuuu  iiiiiiittttttt:::::ttttttt
#  C:::::C              i:::::ir::::rrr:::::::::r    cc:::::::::::::::cu::::u    u::::u  i:::::it:::::::::::::::::t
#  C:::::C               i::::ir:::::::::::::::::r  c:::::::::::::::::cu::::u    u::::u   i::::it:::::::::::::::::t
#  C:::::C               i::::irr::::::rrrrr::::::rc:::::::cccccc:::::cu::::u    u::::u   i::::itttttt:::::::tttttt
#  C:::::C               i::::i r:::::r     r:::::rc::::::c     cccccccu::::u    u::::u   i::::i      t:::::t
#  C:::::C               i::::i r:::::r     rrrrrrrc:::::c             u::::u    u::::u   i::::i      t:::::t
#  C:::::C               i::::i r:::::r            c:::::c             u::::u    u::::u   i::::i      t:::::t
#   C:::::C       CCCCCC i::::i r:::::r            c::::::c     cccccccu:::::uuuu:::::u   i::::i      t:::::t    tttttt
#    C:::::CCCCCCCC::::Ci::::::ir:::::r            c:::::::cccccc:::::cu:::::::::::::::uui::::::i     t::::::tttt:::::t
#     CC:::::::::::::::Ci::::::ir:::::r             c:::::::::::::::::c u:::::::::::::::ui::::::i     tt::::::::::::::t
#       CCC::::::::::::Ci::::::ir:::::r              cc:::::::::::::::c  uu::::::::uu:::ui::::::i       tt:::::::::::tt
#          CCCCCCCCCCCCCiiiiiiiirrrrrrr                cccccccccccccccc    uuuuuuuu  uuuuiiiiiiii         ttttttttttt


class TFCircuitBackend(CircuitBackendInterface):
    def _ABC(
        self, cov: tf.Tensor, means: tf.Tensor, mixed: bool = False, hbar: float = 2.0
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        num_modes = means.shape[-1] // 2

        R = utils.rotmat(num_modes)
        sigma = R @ tf.cast(cov / hbar, tf.complex128) @ tf.math.conj(tf.transpose(R))  # cov in amplitude basis
        beta = tf.linalg.matvec(R, tf.cast(means / np.sqrt(hbar), tf.complex128))  # means in amplitude basis

        sQ = sigma + 0.5 * tf.eye(2 * num_modes, dtype=tf.complex128)
        sQinv = tf.linalg.inv(sQ)

        A = tf.cast(utils.Xmat(num_modes), tf.complex128) @ (np.identity(2 * num_modes) - sQinv)
        B = tf.linalg.matvec(tf.transpose(sQinv), tf.math.conj(beta))

        T = tf.math.exp(-0.5 * tf.einsum("i,ij,j", tf.math.conj(beta), sQinv, beta)) / tf.math.sqrt(tf.linalg.det(sQ))
        N = num_modes + num_modes * mixed
        return (
            A[N:, N:],
            B[N:],
            T ** (0.5 + 0.5 * mixed),
        )  # will be off by global phase because T is real

    @tf.custom_gradient
    def _recursive_state(self, A: tf.Tensor, B: tf.Tensor, C: tf.Tensor, cutoffs: Sequence[int]):
        mixed = len(B) == 2 * len(cutoffs)
        cutoffs_minus_1 = tuple([c - 1 for c in cutoffs] + [c - 1 for c in cutoffs] * mixed)
        state = np.zeros(tuple(cutoffs) + tuple(cutoffs) * mixed, dtype=np.complex128)
        state[(0,) * (len(cutoffs) + len(cutoffs) * mixed)] = C
        state = utils.fill_amplitudes(state, A, B, cutoffs_minus_1)

        def grad(dy):
            dA = np.zeros(tuple(cutoffs) + tuple(cutoffs) * mixed + A.shape, dtype=np.complex128)
            dB = np.zeros(tuple(cutoffs) + tuple(cutoffs) * mixed + B.shape, dtype=np.complex128)
            dA, dB = utils.fill_gradients(dA, dB, state, A, B, cutoffs_minus_1)
            dC = state / C
            dLdA = np.sum(dy[..., None, None] * np.conj(dA), axis=tuple(range(dy.ndim)))
            dLdB = np.sum(dy[..., None] * np.conj(dB), axis=tuple(range(dy.ndim)))
            dLdC = np.sum(dy * np.conj(dC), axis=tuple(range(dy.ndim)))
            return dLdA, dLdB, dLdC

        return state, grad


#       OOOOOOOOO                                  tttt            iiii
#     OO:::::::::OO                             ttt:::t           i::::i
#   OO:::::::::::::OO                           t:::::t            iiii
#  O:::::::OOO:::::::O                          t:::::t
#  O::::::O   O::::::Oppppp   ppppppppp   ttttttt:::::ttttttt    iiiiiii    mmmmmmm    mmmmmmm
#  O:::::O     O:::::Op::::ppp:::::::::p  t:::::::::::::::::t    i:::::i  mm:::::::m  m:::::::mm
#  O:::::O     O:::::Op:::::::::::::::::p t:::::::::::::::::t     i::::i m::::::::::mm::::::::::m
#  O:::::O     O:::::Opp::::::ppppp::::::ptttttt:::::::tttttt     i::::i m::::::::::::::::::::::m
#  O:::::O     O:::::O p:::::p     p:::::p      t:::::t           i::::i m:::::mmm::::::mmm:::::m
#  O:::::O     O:::::O p:::::p     p:::::p      t:::::t           i::::i m::::m   m::::m   m::::m
#  O:::::O     O:::::O p:::::p     p:::::p      t:::::t           i::::i m::::m   m::::m   m::::m
#  O::::::O   O::::::O p:::::p    p::::::p      t:::::t    tttttt i::::i m::::m   m::::m   m::::m
#  O:::::::OOO:::::::O p:::::ppppp:::::::p      t::::::tttt:::::ti::::::im::::m   m::::m   m::::m
#   OO:::::::::::::OO  p::::::::::::::::p       tt::::::::::::::ti::::::im::::m   m::::m   m::::m
#     OO:::::::::OO    p::::::::::::::pp          tt:::::::::::tti::::::im::::m   m::::m   m::::m
#       OOOOOOOOO      p::::::pppppppp              ttttttttttt  iiiiiiiimmmmmm   mmmmmm   mmmmmm
#                      p:::::p
#                      p:::::p
#                     p:::::::p
#                     p:::::::p
#                     p:::::::p
#                     ppppppppp


class TFOptimizerBackend(OptimizerBackendInterface):
    _backend_opt = tf.optimizers.Adam

    def _loss_and_gradients(
        self,
        symplectic_params: Sequence[tf.Tensor],
        euclidean_params: Sequence[tf.Tensor],
        loss_fn: Callable,
    ):
        with tf.GradientTape() as tape:
            loss = loss_fn()
        symp_grads, eucl_grads = tape.gradient(loss, [symplectic_params, euclidean_params])
        return loss.numpy(), symp_grads, eucl_grads

    def _update_symplectic(self, symplectic_grads: Sequence[tf.Tensor], symplectic_params: Sequence[tf.Tensor]) -> None:
        for S, dS_eucl in zip(symplectic_params, symplectic_grads):
            Jmat = utils.J(S.shape[-1] // 2)
            Z = np.matmul(np.transpose(S), dS_eucl)
            Y = 0.5 * (Z + np.matmul(np.matmul(Jmat, Z.T), Jmat))
            S.assign(
                S @ expm(-self._symplectic_lr * np.transpose(Y)) @ expm(-self._symplectic_lr * (Y - np.transpose(Y))),
                read_value=False,
            )

    def _update_euclidean(self, euclidean_grads: Sequence[tf.Tensor], euclidean_params: Sequence[tf.Tensor]) -> None:
        self._opt.apply_gradients(zip(euclidean_grads, euclidean_params))

    def _all_symplectic_parameters(self, circuits: Sequence):
        symp = []
        for circ in circuits:
            for s in circ.symplectic_parameters:
                if s.ref() not in symp:
                    symp.append(s.ref())
        return [s.deref() for s in symp]

    def _all_euclidean_parameters(self, circuits: Sequence):
        eucl = []
        for circ in circuits:
            for e in circ.euclidean_parameters:
                if e.ref() not in eucl:
                    eucl.append(e.ref())
        return [e.deref() for e in eucl]


#     SSSSSSSSSSSSSSS      tttt                                    tttt
#   SS:::::::::::::::S  ttt:::t                                 ttt:::t
#  S:::::SSSSSS::::::S  t:::::t                                 t:::::t
#  S:::::S     SSSSSSS  t:::::t                                 t:::::t
#  S:::::S        ttttttt:::::ttttttt      aaaaaaaaaaaaa  ttttttt:::::ttttttt        eeeeeeeeeeee
#  S:::::S        t:::::::::::::::::t      a::::::::::::a t:::::::::::::::::t      ee::::::::::::ee
#   S::::SSSS     t:::::::::::::::::t      aaaaaaaaa:::::at:::::::::::::::::t     e::::::eeeee:::::ee
#    SS::::::SSSSStttttt:::::::tttttt               a::::atttttt:::::::tttttt    e::::::e     e:::::e
#      SSS::::::::SS    t:::::t              aaaaaaa:::::a      t:::::t          e:::::::eeeee::::::e
#         SSSSSS::::S   t:::::t            aa::::::::::::a      t:::::t          e:::::::::::::::::e
#              S:::::S  t:::::t           a::::aaaa::::::a      t:::::t          e::::::eeeeeeeeeee
#              S:::::S  t:::::t    tttttta::::a    a:::::a      t:::::t    tttttte:::::::e
#  SSSSSSS     S:::::S  t::::::tttt:::::ta::::a    a:::::a      t::::::tttt:::::te::::::::e
#  S::::::SSSSSS:::::S  tt::::::::::::::ta:::::aaaa::::::a      tt::::::::::::::t e::::::::eeeeeeee
#  S:::::::::::::::SS     tt:::::::::::tt a::::::::::aa:::a       tt:::::::::::tt  ee:::::::::::::e
#   SSSSSSSSSSSSSSS         ttttttttttt    aaaaaaaaaa  aaaa         ttttttttttt      eeeeeeeeeeeeee


class TFStateBackend(StateBackendInterface):
    def photon_number_mean(self, cov: tf.Tensor, means: tf.Tensor, hbar: float) -> tf.Tensor:
        N = means.shape[-1] // 2
        return (means[:N] ** 2 + means[N:] ** 2 + tf.linalg.diag_part(cov[:N, :N]) + tf.linalg.diag_part(cov[N:, N:]) - hbar) / (
            2 * hbar
        )

    def photon_number_covariance(self, cov, means, hbar) -> tf.Tensor:
        N = means.shape[-1] // 2
        mCm = cov * means[:, None] * means[None, :]
        dd = tf.linalg.diag(tf.linalg.diag_part(mCm[:N, :N] + mCm[N:, N:] + mCm[:N, N:] + mCm[N:, :N])) / (2 * hbar ** 2)
        CC = (cov ** 2 + mCm) / (2 * hbar ** 2)
        return CC[:N, :N] + CC[N:, N:] + CC[:N, N:] + CC[N:, :N] + dd - 0.25 * np.identity(N)


#          GGGGGGGGGGGGG                          tttt
#       GGG::::::::::::G                       ttt:::t
#     GG:::::::::::::::G                       t:::::t
#    G:::::GGGGGGGG::::G                       t:::::t
#   G:::::G       GGGGGG  aaaaaaaaaaaaa  ttttttt:::::ttttttt        eeeeeeeeeeee
#  G:::::G                a::::::::::::a t:::::::::::::::::t      ee::::::::::::ee
#  G:::::G                aaaaaaaaa:::::at:::::::::::::::::t     e::::::eeeee:::::ee
#  G:::::G    GGGGGGGGGG           a::::atttttt:::::::tttttt    e::::::e     e:::::e
#  G:::::G    G::::::::G    aaaaaaa:::::a      t:::::t          e:::::::eeeee::::::e
#  G:::::G    GGGGG::::G  aa::::::::::::a      t:::::t          e:::::::::::::::::e
#  G:::::G        G::::G a::::aaaa::::::a      t:::::t          e::::::eeeeeeeeeee
#   G:::::G       G::::Ga::::a    a:::::a      t:::::t    tttttte:::::::e
#    G:::::GGGGGGGG::::Ga::::a    a:::::a      t::::::tttt:::::te::::::::e
#     GG:::::::::::::::Ga:::::aaaa::::::a      tt::::::::::::::t e::::::::eeeeeeee
#       GGG::::::GGG:::G a::::::::::aa:::a       tt:::::::::::tt  ee:::::::::::::e
#          GGGGGG   GGGG  aaaaaaaaaa  aaaa         ttttttttttt      eeeeeeeeeeeeee


class TFGateBackend(GateBackendInterface):
    def loss_X(self, transmissivity: tf.Tensor) -> tf.Tensor:
        r"""Returns the X matrix for the lossy bosonic channel.
        The channel is applied to a covariance matrix `\Sigma` as `X\Sigma X^T + Y`.
        """
        D = tf.math.sqrt(transmissivity)
        return tf.linalg.diag(tf.concat([D, D], axis=0))

    def loss_Y(self, transmissivity: tf.Tensor, hbar: float) -> tf.Tensor:
        r"""Returns the Y (noise) matrix for the lossy bosonic channel.
        The channel is applied to a covariance matrix `\Sigma` as `X\Sigma X^T + Y`.
        """
        D = tf.math.sqrt((1.0 - transmissivity) * hbar / 2.0)
        return tf.linalg.diag(tf.concat([D, D], axis=0))

    def thermal_X(self, nbar: tf.Tensor, hbar: float) -> Tuple[tf.Tensor, tf.Tensor]:
        raise NotImplementedError

    def thermal_Y(self, nbar: tf.Tensor, hbar: float) -> Tuple[tf.Tensor, tf.Tensor]:
        raise NotImplementedError

    def displacement(self, x: tf.Tensor, y: tf.Tensor, hbar: float) -> tf.Tensor:
        return np.sqrt(2 * hbar) * tf.concat([x, y], axis=0)

    def beam_splitter_symplectic(self, theta: tf.Tensor, phi: tf.Tensor) -> tf.Tensor:
        r"""Beam-splitter.
        Args:
            theta: transmissivity parameter
            phi: phase parameter
        Returns:
            array: symplectic-orthogonal transformation matrix of an interferometer with angles theta and phi
        """
        ct = tf.math.cos(theta)
        st = tf.math.sin(theta)
        cp = tf.math.cos(phi)
        sp = tf.math.sin(phi)
        return tf.convert_to_tensor(
            [
                [ct, -cp * st, 0, -sp * st],
                [cp * st, ct, -sp * st, 0],
                [0, sp * st, ct, -cp * st],
                [sp * st, 0, cp * st, ct],
            ]
        )

    def rotation_symplectic(self, phi: tf.Tensor) -> tf.Tensor:
        r"""Rotation gate.
        Args:
            phi: rotation angles
        Returns:
            array: rotation matrices by angle theta
        """
        num_modes = phi.shape[-1]
        x = tf.math.cos(phi)
        y = tf.math.sin(phi)
        return tf.linalg.diag(tf.concat([x, x], axis=0)) + tf.linalg.diag(-y, k=num_modes) + tf.linalg.diag(y, k=-num_modes)

    def squeezing_symplectic(self, r: tf.Tensor, phi: tf.Tensor) -> tf.Tensor:
        r"""Squeezing. In fock space this corresponds to \exp(\tfrac{1}{2}r e^{i \phi} (a^2 - a^{\dagger 2}) ).

        Args:
            r: squeezing magnitude
            phi: rotation parameter
        Returns:
            array: symplectic transformation matrix
        """
        # pylint: disable=assignment-from-no-return
        num_modes = phi.shape[-1]
        cp = tf.math.cos(phi)
        sp = tf.math.sin(phi)
        ch = tf.math.cosh(r)
        sh = tf.math.sinh(r)
        return (
            tf.linalg.diag(tf.concat([ch - cp * sh, ch + cp * sh], axis=0))
            + tf.linalg.diag(-sp * sh, k=num_modes)
            + tf.linalg.diag(-sp * sh, k=-num_modes)
        )

    def two_mode_squeezing_symplectic(self, r: tf.Tensor, phi: tf.Tensor) -> tf.Tensor:
        r"""Two-mode squeezing.
        Args:
            r: squeezing magnitude
            phi: rotation parameter
        Returns:
            array: symplectic transformation matrix
        """
        # pylint: disable=assignment-from-no-return
        cp = tf.math.cos(phi)
        sp = tf.math.sin(phi)
        ch = tf.math.cosh(r)
        sh = tf.math.sinh(r)
        return tf.convert_to_tensor(
            [
                [ch, cp * sh, 0, sp * sh],
                [cp * sh, ch, sp * sh, 0],
                [0, sp * sh, ch, -cp * sh],
                [sp * sh, 0, -cp * sh, ch],
            ]
        )


#  MMMMMMMM               MMMMMMMM                          tttt         hhhhhhh
#  M:::::::M             M:::::::M                       ttt:::t         h:::::h
#  M::::::::M           M::::::::M                       t:::::t         h:::::h
#  M:::::::::M         M:::::::::M                       t:::::t         h:::::h
#  M::::::::::M       M::::::::::M  aaaaaaaaaaaaa  ttttttt:::::ttttttt    h::::h hhhhh
#  M:::::::::::M     M:::::::::::M  a::::::::::::a t:::::::::::::::::t    h::::hh:::::hhh
#  M:::::::M::::M   M::::M:::::::M  aaaaaaaaa:::::at:::::::::::::::::t    h::::::::::::::hh
#  M::::::M M::::M M::::M M::::::M           a::::atttttt:::::::tttttt    h:::::::hhh::::::h
#  M::::::M  M::::M::::M  M::::::M    aaaaaaa:::::a      t:::::t          h::::::h   h::::::h
#  M::::::M   M:::::::M   M::::::M  aa::::::::::::a      t:::::t          h:::::h     h:::::h
#  M::::::M    M:::::M    M::::::M a::::aaaa::::::a      t:::::t          h:::::h     h:::::h
#  M::::::M     MMMMM     M::::::Ma::::a    a:::::a      t:::::t    tttttth:::::h     h:::::h
#  M::::::M               M::::::Ma::::a    a:::::a      t::::::tttt:::::th:::::h     h:::::h
#  M::::::M               M::::::Ma:::::aaaa::::::a      tt::::::::::::::th:::::h     h:::::h
#  M::::::M               M::::::M a::::::::::aa:::a       tt:::::::::::tth:::::h     h:::::h
#  MMMMMMMM               MMMMMMMM  aaaaaaaaaa  aaaa         ttttttttttt  hhhhhhh     hhhhhhh


class TFMathbackend(MathBackendInterface):
    def identity(self, size: int) -> tf.Tensor:
        return tf.eye(size, dtype=tf.float64)

    def zeros(self, size: int) -> tf.Tensor:
        return tf.zeros(size, dtype=tf.float64)

    def add(self, old: tf.Tensor, new: Optional[tf.Tensor], modes: List[int]) -> tf.Tensor:
        if new is None:
            return old
        N = old.shape[-1] // 2
        indices = modes + [m + N for m in modes]
        return tf.tensor_scatter_nd_add(old, list(product(*[indices] * len(new.shape))), tf.reshape(new, -1))

    def concat(self, lst: List[tf.Tensor]) -> tf.Tensor:  # TODO: remove?
        return tf.concat(lst, axis=-1)

    def sandwich(self, bread: Optional[tf.Tensor], filling: tf.Tensor, modes: List[int]) -> tf.Tensor:
        if bread is None:
            return filling
        N = filling.shape[-1] // 2
        indices = tf.convert_to_tensor(modes + [m + N for m in modes])
        rows = tf.matmul(bread, tf.gather(filling, indices))
        filling = tf.tensor_scatter_nd_update(filling, indices[:, None], rows)
        columns = bread @ tf.gather(tf.transpose(filling), indices)
        return tf.transpose(tf.tensor_scatter_nd_update(tf.transpose(filling), indices[:, None], columns))

    def matvec(self, mat: Optional[tf.Tensor], vec: tf.Tensor, modes: List[int]) -> tf.Tensor:
        if mat is None:
            return vec
        N = vec.shape[-1] // 2
        indices = tf.convert_to_tensor(modes + [m + N for m in modes])
        updates = tf.linalg.matvec(mat, tf.gather(vec, indices))
        return tf.tensor_scatter_nd_update(vec, indices[:, None], updates)

    def modsquare(self, array: tf.Tensor) -> tf.Tensor:
        return tf.abs(array) ** 2

    def all_diagonals(self, rho: tf.Tensor) -> tf.Tensor:
        cutoffs = rho.shape[: rho.ndim // 2]
        rho = tf.reshape(rho, (np.prod(cutoffs), np.prod(cutoffs)))
        diag = tf.linalg.diag_part(rho)
        return tf.reshape(diag, cutoffs)

    def block(self, blocks: List[List]):
        rows = [tf.concat(row, axis=1) for row in blocks]
        return tf.concat(rows, axis=0)

    def make_symplectic_parameter(
        self,
        init_value: Optional[tf.Tensor] = None,
        trainable: bool = True,
        num_modes: int = 1,
        name: str = "symplectic",
    ) -> tf.Tensor:
        if init_value is None:
            if num_modes == 1:
                W = np.exp(1j * np.random.uniform(size=(1, 1)))
                V = np.exp(1j * np.random.uniform(size=(1, 1)))
            else:
                W = unitary_group.rvs(num_modes)
                V = unitary_group.rvs(num_modes)
            r = np.random.uniform(size=num_modes)
            OW = self.unitary_to_orthogonal(W)
            OV = self.unitary_to_orthogonal(V)
            dd = tf.concat([tf.math.exp(-r), tf.math.exp(r)], 0)
            val = tf.einsum("ij,j,jk->ik", OW, dd, OV)
        else:
            val = init_value
        if trainable:
            return tf.Variable(val, dtype=tf.float64, name=name)
        else:
            return tf.constant(val, dtype=tf.float64, name=name)

    def unitary_to_orthogonal(self, U):
        r"""Unitary to orthogonal mapping.
        Args:
            U (array): unitary matrix in U(n)
        Returns:
            array: Orthogonal matrix in O(2n)
        """
        X = tf.math.real(U)
        Y = tf.math.imag(U)
        return self.block([[X, -Y], [Y, X]])

    def make_euclidean_parameter(
        self,
        init_value: Optional[float] = None,
        trainable: bool = True,
        bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        shape: Optional[Sequence[int]] = None,
        name: str = "",
    ) -> tf.Tensor:

        bounds = (bounds[0] or -np.inf, bounds[1] or np.inf)
        if not bounds == (-np.inf, np.inf):
            constraint: Optional[Callable] = lambda x: tf.clip_by_value(x, bounds[0], bounds[1])
        else:
            constraint = None

        if init_value is None:
            if trainable:
                val = truncnorm.rvs(*bounds, size=shape)
            else:
                val = tf.zeros(shape, dtype=tf.float64)
        else:
            val = init_value if shape is None else np.atleast_1d(init_value)

        if trainable:
            return tf.Variable(val, dtype=tf.float64, name=name, constraint=constraint)
        else:
            return tf.constant(val, dtype=tf.float64, name=name)

    def tensordot(self, a, b, axes):
        return tf.tensordot(a, b, axes)

    def transpose(self, a, perm):
        return tf.transpose(a, perm)


class TFDetectorBackend(DetectorBackendInterface):
    def convolve_probs_1d(self, prob: tf.Tensor, other_probs: List[tf.Tensor]) -> tf.Tensor:
        "Convolution of a joint probability with a list of single-index probabilities"

        if prob.ndim > 3 or len(other_probs) > 3:
            raise ValueError("cannot convolve arrays with more than 3 axes")
        if not all([q.ndim == 1 for q in other_probs]):
            raise ValueError("other_probs must contain 1d arrays")
        if not all([len(q) == s for q, s in zip(other_probs, prob.shape)]):
            raise ValueError("The length of the 1d prob vectors must match shape of prob")

        q = other_probs[0]
        for q_ in other_probs[1:]:
            q = q[..., None] * q_[(None,) * q.ndim + (slice(None),)]

        return self.convolve_probs(prob, q)

    def convolve_probs(self, prob: tf.Tensor, other: tf.Tensor) -> tf.Tensor:
        """Convolve two probability distributions.
        Note that the output is not a complete joint probability,
        as it's computed only up to the dimension of the base probs."""

        if prob.ndim > 3 or other.ndim > 3:
            raise ValueError("cannot convolve arrays with more than 3 axes")
        if not prob.shape == other.shape:
            raise ValueError("prob and other must have the same shape")

        prob_padded = tf.pad(prob, [(s - 1, 0) for s in other.shape])
        other_reversed = other[(slice(None, None, -1),) * other.ndim]
        return tf.nn.convolution(prob_padded[None, ..., None], other_reversed[..., None, None], padding="VALID")[0, ..., 0]
