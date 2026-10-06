# CELL 13 — callbacks
# ONE global best, written to ONE file. Per-stage counters reset to -1 meant Stage C
# always saved its first epoch regardless of F1.
BEST_PATH = str(OUT / "best.keras")

class SaveBestF1(KC.Callback):
    best = -1.0
    def on_epoch_end(self, epoch, logs=None):
        f1 = logs.get("val_macro_f1", -1.0)
        if f1 > SaveBestF1.best:
            SaveBestF1.best = f1
            self.model.save(BEST_PATH)
            print(f"  ** saved best.keras (val_macro_f1={f1:.4f})")

class BudgetStop(KC.Callback):
    """Caps TRAINING only. T_TRAIN0 is set here, not at notebook start, so downloads,
    grouping and preloading do not eat the budget."""
    def on_epoch_begin(self, epoch, logs=None):
        if (time.time() - T_TRAIN0) / 60.0 > CFG["train_budget_min"]:
            self.model.stop_training = True
            print(f"!! training budget {CFG['train_budget_min']} min hit -> stopping")

def make_cbs(tag):
    return [SaveBestF1(),
            KC.EarlyStopping(monitor="val_macro_f1", mode="max", patience=3, verbose=1),
            KC.ReduceLROnPlateau(monitor="val_macro_f1", mode="max", factor=0.5,
                                 patience=1, min_lr=1e-6, verbose=1),
            KC.CSVLogger(str(OUT / f"log_{tag}.csv"), append=False),
            BudgetStop()]
