# CELL 12 — model + trainability
# training=False on the backbone call is deliberate and does NOT block fine-tuning:
# the `training` arg only switches Dropout and BatchNorm to inference mode. Conv/Dense
# kernels still receive gradients whenever layer.trainable is True. It also guarantees
# BatchNorm never updates its running statistics, which is what you want on 8k images.
def build_model(k):
    inp = tf.keras.Input((SIZE, SIZE, 3), dtype=tf.float32)
    base = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet",
                                                 input_shape=(SIZE, SIZE, 3), pooling="avg")
    x = base(inp, training=False)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="head_dense")(x)
    x = tf.keras.layers.Dropout(0.20)(x)
    out = tf.keras.layers.Dense(k, activation="softmax", dtype="float32", name="pred")(x)
    return tf.keras.Model(inp, out, name="agrisense_b0")

def base_of(m):
    return next(l for l in m.layers if isinstance(l, tf.keras.Model))

def set_trainable(m, n, verbose=True):
    """Unfreeze n layers of the BACKBONE. The head always trains.
    Walking base.layers is essential: Keras names EfficientNet internals stem_conv /
    block1a_dwconv, so any name-based match on 'efficientnet' finds only the wrapper."""
    base = base_of(m)
    m.trainable = True                                    # head trainable in every stage
    wl = [l for l in base.layers if l.get_weights()]
    keep = [] if n == 0 else (wl[:n] if n > 0 else wl[n:])
    ks = {id(l) for l in keep}
    for l in base.layers:
        l.trainable = id(l) in ks
    n_bb = sum(int(l.trainable) for l in wl)
    head_n = len([l for l in m.layers if l is not base and l.get_weights() and l.trainable])
    if verbose:
        print(f"  backbone {n_bb}/{len(wl)} layers trainable | "
              f"head {head_n} layers trainable | {len(m.trainable_weights)} trainable tensors")
    return n_bb

if STRATEGY is not None:
    with STRATEGY.scope():
        model = build_model(NC)
else:
    model = build_model(NC)
assert model.output_shape[-1] == NC, f"CLASS/OUTPUT MISMATCH {NC} vs {model.output_shape[-1]}"
set_trainable(model, 0, verbose=False)
model.summary()
_bb = len([l for l in base_of(model).layers if l.get_weights()])
print(f"backbone has {_bb} weighted layers; stages B/C unfreeze the last 18 and 70")
