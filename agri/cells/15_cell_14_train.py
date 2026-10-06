# CELL 14 — train
import contextlib
T_TRAIN0 = time.time()
cw = None if mode in ("oversample", "none") else {i: float(w[i]) for i in range(NC)}

for st in CFG["stages"]:
    spent = (time.time() - T_TRAIN0) / 60.0
    if spent > CFG["train_budget_min"]:
        print(f"budget spent ({spent:.1f}m) before {st['name']} -> skipping remaining stages")
        break
    print(f"\n=== {st['name']} | unfreeze last {st['unfreeze']} | {st['epochs']} epochs | "
          f"lr {st['lr']} | spent {spent:.1f}m ===")
    nbb = set_trainable(model, st["unfreeze"])
    expect = 0 if st["unfreeze"] == 0 else abs(st["unfreeze"])
    if st["unfreeze"] == 0:
        assert nbb == 0, "stage A should have a frozen backbone"
    else:
        assert nbb == expect, f"expected {expect} backbone layers trainable, got {nbb}"
    assert len(model.trainable_weights) > len(base_of(model).trainable_weights), \
        "nothing is trainable — the head itself is frozen"

    kw = {"jit_compile": True} if JIT else {}
    # compile INSIDE the strategy scope, or optimizer/metric variables land outside it
    with (STRATEGY.scope() if STRATEGY is not None else contextlib.nullcontext()):
        model.compile(optimizer=tf.keras.optimizers.Adam(st["lr"]),
                      loss=LOSS(), metrics=["accuracy", macro_f1()], **kw)
    model.fit(train_ds, validation_data=val_ds, epochs=st["epochs"],
              steps_per_epoch=STEPS, class_weight=cw,
              callbacks=make_cbs(st["name"]), verbose=2)

print(f"\ntraining wall clock: {(time.time()-T_TRAIN0)/60:.1f} min "
      f"(budget {CFG['train_budget_min']}) | best val macro_f1 seen {SaveBestF1.best:.4f}")

# compile=False: we only need inference, and this avoids rebuilding the custom metric
model = tf.keras.models.load_model(BEST_PATH, compile=False)
assert model.output_shape[-1] == NC, f"CLASS/OUTPUT MISMATCH {NC} vs {model.output_shape[-1]}"
print(f"loaded best.keras | outputs {model.output_shape[-1]} == declared {NC}  OK")
