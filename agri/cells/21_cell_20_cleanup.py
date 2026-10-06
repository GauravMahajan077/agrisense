# CELL 20 — cleanup
del train_ds, val_ds, test_ds
if USE_RAM:
    del Xtr, Xva, Xte
import gc; gc.collect()
print(f"freed input arrays; RAM available now {_ram_gb()[1]:.1f} GB")
