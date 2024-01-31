from __future__ import annotations

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Timer
from pytorch_lightning.loggers import CSVLogger

trainer_config = {
    "debug": {
        "accelerator": "cpu",
        "limit_train_batches": 10,
        "limit_val_batches": 10,
        "log_every_n_steps": 1,
        "max_epochs": 2,
    },
    "8gpu": {
        "accelerator": "gpu",
        "strategy": "ddp_find_unused_parameters",
        "devices": 8,
    },
    "generic": {"min_epochs": 15, "max_epochs": 100},
}

callbacks = [
    ModelCheckpoint(monitor=OPTIMIZATION_TARGET, save_top_k=5),
    EarlyStopping(
        patience=15,
        monitor={OPTIMIZATION_TARGET},
        mode="min",
        verbose=True,
        check_finite=False,
    ),
    CSVLogger(save_dir=log_path),
    Timer(),
]
