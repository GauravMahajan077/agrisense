# CELL 11 — graph-safe macro-F1
# result() is traced into the compiled graph, so it must be pure TF ops — numpy here fails.
# get_config is required for the metric to survive model.save / load_model.
@tf.keras.utils.register_keras_serializable(package="agrisense")
class MacroF1(tf.keras.metrics.Metric):
    def __init__(self, k, name="macro_f1", dtype=None, **kw):
        super().__init__(name=name, dtype=dtype, **kw)
        self.k = int(k)
        self.cm = self.add_weight(shape=(int(k), int(k)), initializer="zeros",
                                  dtype=tf.float32, name="confusion")
    def update_state(self, y_true, y_pred, sample_weight=None):
        m = tf.math.confusion_matrix(tf.argmax(y_true, -1), tf.argmax(y_pred, -1),
                                     num_classes=self.k, dtype=tf.float32)
        self.cm.assign_add(m)
    def result(self):
        cm = tf.cast(self.cm, tf.float32)
        tp = tf.linalg.diag_part(cm)
        prec = tf.math.divide_no_nan(tp, tf.reduce_sum(cm, axis=0))
        rec  = tf.math.divide_no_nan(tp, tf.reduce_sum(cm, axis=1))
        return tf.reduce_mean(tf.math.divide_no_nan(2 * prec * rec, prec + rec))
    def reset_state(self):
        self.cm.assign(tf.zeros_like(self.cm))
    def get_config(self):
        c = super().get_config(); c["k"] = self.k; return c

def macro_f1(): return MacroF1(NC)
