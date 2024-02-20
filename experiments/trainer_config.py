from __future__ import annotations

import pytorch_lightning as pl
from model_config import available_models
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from matsciml.lightning.callbacks import CodeCarbonCallback
from pytorch_lightning.loggers import CSVLogger

from matsciml.lightning.callbacks import Timer
from matsciml.models.base import (
    BinaryClassificationTask,
    CrystalSymmetryClassificationTask,
    ForceRegressionTask,
    GradFreeForceRegressionTask,
    MaceEnergyForceTask,
    ScalarRegressionTask,
)

trainer_config = {
    "debug": {
        "accelerator": "cpu",
        "limit_train_batches": 10,
        "limit_val_batches": 10,
        "log_every_n_steps": 1,
        "max_epochs": 2,
    },
    "experiment": {
        "accelerator": "gpu",
        "strategy": "ddp_find_unused_parameters_true",
    },
    "generic": {"min_epochs": 15, "max_epochs": 100},
}


def setup_callbacks(opt_target, log_path):
    callbacks = [
        ModelCheckpoint(monitor=opt_target, save_top_k=5),
        CodeCarbonCallback(
            output_dir=log_path, country_iso_code="USA", measure_power_secs=1
        ),
        EarlyStopping(
            patience=15,
            monitor={opt_target},
            mode="min",
            verbose=True,
            check_finite=False,
        ),
        Timer(),
    ]
    return callbacks


def setup_logger(log_path):
    logger = CSVLogger(save_dir=log_path)
    return logger


def setup_task(args):
    task_map = {
        "sr": ScalarRegressionTask,
        "fr": ForceRegressionTask,
        "bc": BinaryClassificationTask,
        "cs": CrystalSymmetryClassificationTask,
        "me": MaceEnergyForceTask,
        "gffr": GradFreeForceRegressionTask,
    }

    task = task_map[args.task]
    model_args = available_models["generic"]
    model_args.update(available_models[args.model])
    model_args.update({"task_keys": [args.target]})
    task = task(**model_args)
    return task


def setup_trainer(args, callbacks, logger):
    trainer_args = trainer_config["generic"]
    trainer_args.update(trainer_config[args.run_type])
    if args.run_type == "experiment":
        trainer_args.update({"devices": args.gpus})

    trainer = pl.Trainer(callbacks=callbacks, logger=logger, **trainer_args)

    return trainer
