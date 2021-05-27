class TFOptimizerBackend(OptimizerBackendInterface):
    _backend_opt = tf.optimizers.Adam

    def _loss_and_gradients(
        self,
        symplectic_params: Sequence[tf.Tensor],
        euclidean_params: Sequence[tf.Tensor],
        cost_fn: Callable,
    ):
        with tf.GradientTape() as tape:
            loss = cost_fn()
        symp_grads, eucl_grads = tape.gradient(loss, [symplectic_params, euclidean_params])
        return loss.numpy(), symp_grads, eucl_grads

    def _update_symplectic(
        self, symplectic_grads: Sequence[tf.Tensor], symplectic_params: Sequence[tf.Tensor]
    ) -> None:
        for S, dS_eucl in zip(symplectic_params, symplectic_grads):
            Jmat = utils.J(S.shape[-1] // 2)
            Z = np.matmul(np.transpose(S), dS_eucl)
            Y = 0.5 * (Z + np.matmul(np.matmul(Jmat, Z.T), Jmat))
            S.assign(
                S
                @ expm(-self._symplectic_lr * np.transpose(Y))
                @ expm(-self._symplectic_lr * (Y - np.transpose(Y))),
                read_value=False,
            )

    def _update_euclidean(
        self, euclidean_grads: Sequence[tf.Tensor], euclidean_params: Sequence[tf.Tensor]
    ) -> None:
        self._opt.apply_gradients(zip(euclidean_grads, euclidean_params))

    def _all_symplectic_parameters(self, items: Sequence):
        symp = []
        for item in items:
            try:
                for s in item.symplectic_parameters:
                    if s.ref() not in symp:
                        symp.append(s.ref())
            except AttributeError:
                continue
        return [s.deref() for s in symp]

    def _all_euclidean_parameters(self, items: Sequence):
        eucl = []
        for item in items:
            try:
                for e in item.euclidean_parameters:
                    if e.ref() not in eucl:
                        eucl.append(e.ref())
            except AttributeError:
                continue
        return [e.deref() for e in eucl]