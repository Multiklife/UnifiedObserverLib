import tensorflow as tf

class SelfObservingOptimizer(tf.keras.optimizers.Optimizer):
    def __init__(self, unified_observer, learning_rate=0.01, name="SelfObservingOptimizer", **kwargs):
        super(SelfObservingOptimizer, self).__init__(name, **kwargs)
        self.unified_observer = unified_observer
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("decay", kwargs.get("decay", 0.95))

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "momentum")

    @tf.function
    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        momentum = self.get_slot(var, "momentum")

        q = tf.constant(self.unified_observer.q, dtype=var_dtype)
        new_momentum = momentum * q + grad * (1 - q)
        
        var_update = var - lr_t * new_momentum
        momentum_update = momentum.assign(new_momentum, use_locking=self._use_locking)

        self.unified_observer.observe(tf.reduce_mean(tf.abs(grad)))

        return tf.group(*[var.assign(var_update, use_locking=self._use_locking),
                          momentum_update])

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        raise NotImplementedError("Sparse gradients are not supported yet.")

    def get_config(self):
        base_config = super(SelfObservingOptimizer, self).get_config()
        return {
            **base_config,
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "decay": self._serialize_hyperparameter("decay"),
        }
