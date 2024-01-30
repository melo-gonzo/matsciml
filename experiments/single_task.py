# import pytorch_lightning as pl
from __future__ import annotations

import sys

sys.path.append("/store/code/open-catalyst/private-repo/matsciml-fork")
import logging
import os
import socket
import time
import traceback

import pytorch_lightning as pl
import torch
from ocpmodels.lightning.callbacks import GradientCheckCallback, ThroughputCallback
from ocpmodels.lightning.data_utils import IS2REDGLDataModule
from ocpmodels.models import PLEGNNBackbone
from ocpmodels.models.base import ForceRegressionTask, ScalarRegressionTask
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.nn import LazyBatchNorm1d, SiLU

NUM_DEVICES = 8
NUM_NODES = 1
BATCH_SIZE = 32
NUM_WORKERS = 32

TRAIN_PATH = "/datasets-alt/open-catalyst/dgl_is2re/is2re/all/train/"
VAL_PATH = "/datasets-alt/open-catalyst/dgl_is2re/is2re/all/val_id/"

OPTIMIZATION_TARGET = "val_energy_relaxed"

log_path = "experiments/egnn/energy/is2re"

log_path = os.path.join(
    "/store/code/open-catalyst/matsciml-logs/neurips23dnb/paper",
    log_path,
)

if not os.path.exists(log_path):
    os.makedirs(log_path)


try:
    trainer_logger = logging.getLogger("pytorch_lightning")
    trainer_logger.setLevel(logging.DEBUG)
    trainer_logger.addHandler(
        logging.FileHandler(os.path.join(log_path, "trainer.log")),
    )

    pl.seed_everything(21616)

    model_args = {
        "embed_in_dim": 1,
        "embed_hidden_dim": 32,
        "embed_out_dim": 128,
        "embed_depth": 5,
        "embed_feat_dims": [128, 128, 128],
        "embed_message_dims": [128, 128, 128],
        "embed_position_dims": [64, 64],
        "embed_edge_attributes_dim": 0,
        "embed_activation": "relu",
        "embed_residual": True,
        "embed_normalize": True,
        "embed_tanh": True,
        "embed_activate_last": False,
        "embed_k_linears": 1,
        "embed_use_attention": False,
        "embed_attention_norm": "sigmoid",
        "readout": "sum",
        "node_projection_depth": 3,
        "node_projection_hidden_dim": 128,
        "node_projection_activation": "relu",
        "prediction_out_dim": 1,
        "prediction_depth": 3,
        "prediction_hidden_dim": 128,
        "prediction_activation": "relu",
    }

    task = ScalarRegressionTask(
        encoder_class=PLEGNNBackbone,
        encoder_kwargs=model_args,
        output_kwargs={"norm": LazyBatchNorm1d, "hidden_dim": 256, "activation": SiLU},
        lr=1e-4,
    )
    dm = IS2REDGLDataModule(
        train_path=TRAIN_PATH,
        val_path=VAL_PATH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    dm.setup(None)
    batch = next(iter(dm.train_dataloader()))
    with torch.no_grad():
        task(batch)

    mc = ModelCheckpoint(monitor=OPTIMIZATION_TARGET, save_top_k=5)
    es = EarlyStopping(
        patience=15,
        monitor=OPTIMIZATION_TARGET,
        mode="min",
        verbose=True,
        check_finite=False,
    )

    logger = CSVLogger(save_dir=log_path)

    trainer = pl.Trainer(
        min_epochs=20,
        max_epochs=50,
        accelerator="gpu",
        devices=NUM_DEVICES,
        num_nodes=NUM_NODES,
        strategy="ddp",
        log_every_n_steps=100,
        logger=logger,
        callbacks=[
            ThroughputCallback(batch_size=BATCH_SIZE),
            GradientCheckCallback(),
            es,
            mc,
        ],
    )

    trainer.fit(task, datamodule=dm)


except Exception as e:
    with open(os.path.join(log_path, "error_log.txt"), "a+") as file:
        file.write("\n" + str(traceback.format_exc()))
        print(traceback.format_exc())
