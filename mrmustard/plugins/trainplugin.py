import tensorflow as tf
import numpy as np
from scipy.linalg import expm
from mrmustard.typing import *
from mrmustard import utils


class TrainPlugin:
    euclidean_opt = tf.keras.optimizers.Adam()
    symplectic_opt = 

    def __init__(self):
        # this is ugly: we shouldn't store the parameters in the plugin
        self._eucl_params: Sequence[tf.Tensor] = []
        self._symp_params: Sequence[tf.Tensor] = []
        self._orth_params: Sequence[tf.Tensor] = []

    def loss_and_gradients(self, cost_fn: Callable):
        with tf.GradientTape() as tape:
            loss = cost_fn()
        symp_grads, orth_grads, eucl_grads = tape.gradient(loss, [self._symp_params, self._orth_params, self._eucl_params])
        return loss.numpy(), symp_grads, orth_grads, eucl_grads

    def store_symp_params(self, items: Sequence[Trainable]):
        for item in items:
            try:
                for s in item.symplectic_parameters:
                    symp.append(s.ref())
            except AttributeError:
                continue
        return symp

    def update_symplectic(self, symplectic_grads: Sequence[tf.Tensor], symplectic_params: Sequence[tf.Tensor], symplectic_lr: float):
        for S, dS_eucl in zip(symplectic_params, symplectic_grads):
            Jmat = utils.J(S.shape[-1] // 2)
            Z = np.matmul(np.transpose(S), dS_eucl)
            Y = 0.5 * (Z + np.linalg.multi_dot([Jmat, Z.T, Jmat]))
            S.assign(S @ expm(-symplectic_lr * np.transpose(Y)) @ expm(-symplectic_lr * (Y - np.transpose(Y))), read_value=False)

    def update_orthogonal(self, orthogonal_grads: Sequence[tf.Tensor], orthogonal_params: Sequence[tf.Tensor], symplectic_lr: float):
        for O, dO_eucl in zip(orthogonal_params, orthogonal_grads):
            D = 0.5 * (dO_eucl - np.linalg.multi_dot([O, np.transpose(dO_eucl), O]))
            O.assign(O @ expm(symplectic_lr * np.matmul(np.transpose(D), O)), read_value=False)

    def update_euclidean(self, euclidean_grads: Sequence[tf.Tensor], euclidean_params: Sequence[tf.Tensor], euclidean_lr: float):
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

    def extract_orthogonal_parameters(self, items: Sequence):
        orth = []
        for item in items:
            try:
                for o in item.orthogonal_parameters:
                    if o.ref() not in orth:
                        orth.append(o.ref())
            except AttributeError:
                continue
        return [o.deref() for o in orth]

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

    def new_symplectic_parameter(
        self,
        init_value: Optional[Matrix] = None,
        trainable: bool = True,
        num_modes: int = 1,
        name: str = "symplectic",
    ):
        

    def new_euclidean_parameter(
        self,
        init_value: Optional[Union[Scalar, Vector]] = None,
        trainable: bool = True,
        bounds: Tuple[Optional[float], Optional[float]] = (None, None),
        shape: Optional[Sequence[int]] = None,
        name: str = "",
    ):
        ...

    
        

    