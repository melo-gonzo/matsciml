from __future__ import annotations

import os

import pytorch_lightning as pl
from data_config import available_data
from model_config import available_models
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from matsciml.lightning.callbacks import CodeCarbonCallback

from matsciml.lightning.callbacks import Timer
from matsciml.models.base import (
    BinaryClassificationTask,
    CrystalSymmetryClassificationTask,
    ForceRegressionTask,
    GradFreeForceRegressionTask,
    MaceEnergyForceTask,
    ScalarRegressionTask,
    MultiTaskLitModule,
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
        # CodeCarbonCallback(
        #     output_dir=log_path, country_iso_code="USA", measure_power_secs=1
        # ),
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
    log_path.replace("/", "-")

    cg_wb_dir = "/store/nosnap/chem-ai/wb-logs"

    if os.path.exists(cg_wb_dir):
        save_dir = cg_wb_dir
    else:
        save_dir = "./experiments-2024/wandb"

    logger = WandbLogger(
        log_model="all",
        name=log_path.replace("/", "-"),
        save_dir=save_dir,
        project="debug",
        entity="ml-logs",
        mode="online",
    )
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

    tasks = []
    for task in args.tasks:
        task = task_map[task]
        task_args = available_models["generic"]
        # TODOD: support multi data
        dset = available_data[args.data[0]]
        normalize_kwargs = dset[args.run_type].pop("normalize_kwargs", None)
        task_args.update(available_models[args.model])
        task_args.update({"task_keys": args.targets})
        task_args.update({"normalize_kwargs": normalize_kwargs})
        task = task(**task_args)
        tasks.append(task)
    if len(tasks) > 1:
        datas = []
        if len(args.data) == 1:
            datas = [available_data[args.data[0]]["dataset"]] * len(tasks)
        for data in args.data:
            datas.append(available_data[data]["dataset"])
        task = MultiTaskLitModule(tuple(zip((datas, tasks))))

    return task


def setup_trainer(args, callbacks, logger):
    trainer_args = trainer_config["generic"]
    trainer_args.update(trainer_config[args.run_type])
    if args.run_type == "experiment":
        trainer_args.update({"devices": args.gpus})

    trainer = pl.Trainer(callbacks=callbacks, logger=logger, **trainer_args)

    return trainer
