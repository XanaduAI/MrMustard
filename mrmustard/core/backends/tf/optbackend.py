import tensorflow as tf
import numpy as np
from typing import Sequence, Callable
from scipy.linalg import expm

from mrmustard.core.backends import OptimizerBackendInterface
from mrmustard.core import utils


class OptimizerBackend(OptimizerBackendInterface):
    euclidean_opt = tf.keras.optimizers.Adam()

    def loss_and_gradients(
        self,
        symplectic_params: Sequence[tf.Tensor],
        euclidean_params: Sequence[tf.Tensor],
        cost_fn: Callable,
    ):
        with tf.GradientTape() as tape:
            loss = cost_fn()
        symp_grads, eucl_grads = tape.gradient(loss, [symplectic_params, euclidean_params])
        return loss.numpy(), symp_grads, eucl_grads

    def update_symplectic(
        self,
        symplectic_grads: Sequence[tf.Tensor],
        symplectic_params: Sequence[tf.Tensor],
        symplectic_lr: float,
    ) -> None:
        for S, dS_eucl in zip(symplectic_params, symplectic_grads):
            Jmat = utils.J(S.shape[-1] // 2)
            Z = np.matmul(np.transpose(S), dS_eucl)
            Y = 0.5 * (Z + np.matmul(np.matmul(Jmat, Z.T), Jmat))
            S.assign(
                S @ expm(-symplectic_lr * np.transpose(Y)) @ expm(-symplectic_lr * (Y - np.transpose(Y))),
                read_value=False,
            )

    def update_euclidean(
        self,
        euclidean_grads: Sequence[tf.Tensor],
        euclidean_params: Sequence[tf.Tensor],
        euclidean_lr: float,
    ) -> None:
        print("Updating euclidean param!")
        self.euclidean_opt.lr = euclidean_lr
        self.euclidean_opt.apply_gradients(zip(euclidean_grads, euclidean_params))

    def extract_symplectic_parameters(self, items: Sequence):
        symp = []
        for item in items:
            try:
                for s in item.symplectic_parameters:
                    if s.ref() not in symp:
                        symp.append(s.ref())
            except AttributeError:
                continue
        return [s.deref() for s in symp]

    def extract_euclidean_parameters(self, items: Sequence):
        eucl = []
        for item in items:
            try:
                for e in item.euclidean_parameters:
                    if e.ref() not in eucl:
                        eucl.append(e.ref())
            except AttributeError:
                continue
        return [e.deref() for e in eucl]
