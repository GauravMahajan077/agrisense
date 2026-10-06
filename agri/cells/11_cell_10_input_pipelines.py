# CELL 10 — input pipelines
# Two things this deliberately avoids:
#   * dataset-level .shuffle() on images: a 20k buffer of 196 KB images is ~3.9 GB. We
#     permute INDICES in Python instead, so the shuffle buffer holds ints.
#   * from_tensor_slices(X): that embeds the array as a graph constant and trips the 2 GB
#     protobuf limit. from_generator keeps X in host memory, outside the graph.
def _rrc(im):
    """Random resized crop via sample_distorted_bounding_box (tf.image has no
    random_resized_crop). Returns float32 0..255 at SIZE."""


    a = CFG["aug"]
    begin, size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(im), bounding_boxes=tf.zeros([1, 0, 4], tf.float32),
        area_range=a["rrc_scale"], aspect_ratio_range=(3/4., 4/3.),
        max_attempts=10, use_image_if_no_bounding_boxes=True)
    return tf.image.resize(tf.slice(im, begin, size), (SIZE, SIZE))

def _aug(im, y):
    """All colour ops run in 0..1, which is what adjust_hue/adjust_saturation assume."""
    a = CFG["aug"]
    h = _rrc(im) / 255.0
    if a["flip_h"]: h = tf.image.random_flip_left_right(h)
    if a["flip_v"]: h = tf.image.random_flip_up_down(h)
    h = tf.image.random_brightness(h, a["brightness"])
    h = tf.image.random_contrast(h, 1 - a["contrast"], 1 + a["contrast"])
    h = tf.image.random_saturation(h, 1 - a["sat"], 1 + a["sat"])
    h = tf.image.random_hue(h, a["hue"])
    h = tf.clip_by_value(h, 0., 1.) * 255.
    h.set_shape([SIZE, SIZE, 3])             # dynamic crop size must not leak unknown H/W
    return h, tf.one_hot(y, NC)

def _eval_t(im, y):
    # PRE (256) -> SIZE (224): the model input is fixed at SIZE, so eval MUST resize.
    # Also cast: resize on a uint8 tensor returns uint8, but the model wants float32.
    h = tf.cast(tf.image.resize(im, (SIZE, SIZE)), tf.float32)
    h.set_shape([SIZE, SIZE, 3])
    return h, tf.one_hot(y, NC)

rng_ds = np.random.RandomState(SEED + 1)
if USE_RAM:
    n_tr = len(ytr)
    def gen_train():
        while True:
            for i in rng_ds.permutation(n_tr):
                yield Xtr[i], ytr[i]
    def gen_eval(X, y):
        for i in range(len(y)):
            yield X[i], y[i]
    train_ds = tf.data.Dataset.from_generator(
        gen_train, output_signature=(tf.TensorSpec([PRE, PRE, 3], tf.uint8),
                                     tf.TensorSpec([], tf.int32)))
    train_ds = train_ds.map(_aug, num_parallel_calls=AUTOTUNE).batch(BS).prefetch(AUTOTUNE)
    def mk_eval(X, y):
        return (tf.data.Dataset.from_generator(
                    lambda X=X, y=y: gen_eval(X, y),
                    output_signature=(tf.TensorSpec([PRE, PRE, 3], tf.uint8),
                                      tf.TensorSpec([], tf.int32)))
                .map(_eval_t, num_parallel_calls=AUTOTUNE).batch(BS*2).prefetch(AUTOTUNE))
    val_ds, test_ds = mk_eval(Xva, yva), mk_eval(Xte, yte)
else:
    def dec(p):
        img = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
        return tf.cast(tf.image.resize(img, (PRE, PRE), method="bilinear",
                                       antialias=True), tf.uint8)
    n_tr = len(ytr)
    def gen_train():
        paths = tr["path"].tolist()
        while True:
            for i in rng_ds.permutation(n_tr):
                yield dec(paths[i]), ytr[i]
    train_ds = (tf.data.Dataset.from_generator(
                    gen_train, output_signature=(tf.TensorSpec([PRE, PRE, 3], tf.uint8),
                                                 tf.TensorSpec([], tf.int32)))
                .map(_aug, num_parallel_calls=AUTOTUNE).batch(BS).prefetch(AUTOTUNE))
    def mk_eval(df, y):
        paths = df["path"].tolist()
        return (tf.data.Dataset.from_generator(
                    lambda p=paths, yy=y: ((dec(q), yy[i]) for i, q in enumerate(p)),
                    output_signature=(tf.TensorSpec([PRE, PRE, 3], tf.uint8),
                                      tf.TensorSpec([], tf.int32)))
                .map(_eval_t, num_parallel_calls=AUTOTUNE).batch(BS*2).prefetch(AUTOTUNE))
    val_ds, test_ds = mk_eval(va, yva), mk_eval(te, yte)

STEPS = int(np.ceil(n_tr / BS))
print(f"pipeline {'RAM-preloaded' if USE_RAM else 'disk stream'} | {STEPS} steps/epoch | "
      f"train {n_tr} | val {len(yva)} | test {len(yte)} | one-hot targets (NC={NC})")
