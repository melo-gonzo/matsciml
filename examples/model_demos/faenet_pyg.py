from __future__ import annotations

import sys
sys.path.append("/workspace/ai-mat-top/matsciml_top/forks/carmelo_matsciml/")

from matsciml.datasets.transforms import (
    FrameAveraging,
    PointCloudToGraphTransform,
)
from matsciml.lightning.data_utils import MatSciMLDataModule
from matsciml.models.base import ScalarRegressionTask, ForceRegressionTask, GradFreeForceRegressionTask, MultiTaskLitModule
from matsciml.models.pyg import FAENet


# Atomic Energies table
import mendeleev
import pytest
import pytorch_lightning as pl
from mendeleev.fetch import fetch_ionization_energies
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
from tqdm import tqdm
from matsciml.lightning.callbacks import GradientCheckCallback

"""
This example script runs through a fast development run of the IS2RE devset
in combination with a PyG implementation of FAENet.
"""

TRAIN_PATH = "/datasets-alt/molecular-data/mat_traj/mp-traj-full/train"
VAL_PATH = "/datasets-alt/molecular-data/mat_traj/mp-traj-full/val"
DATASET = "mp-traj"


# Multi-Task with Scalar (Energy) and GradFreeForce (Force)

task_energy = ScalarRegressionTask(
    encoder_class=FAENet,
    encoder_kwargs={
        "average_frame_embeddings": False,  # set to false for use with FA transform
        "pred_as_dict": False,
        "hidden_dim": 128,
        "out_dim": 64,
        "tag_hidden_channels": 0,
    },
    output_kwargs={"lazy": False, "input_dim": 64, "hidden_dim": 64},
    task_keys=["energy"],
    lr=0.005,
)


task_force = GradFreeForceRegressionTask(
    encoder_class=FAENet,
    encoder_kwargs={
        "average_frame_embeddings": True,
        "pred_as_dict": False,
        "hidden_dim": 128,
        "out_dim": 64,
        "tag_hidden_channels": 0,
    },
    output_kwargs={"lazy": False, "input_dim": 64, "hidden_dim": 64},
    # task_keys=["energy"],
    lr=0.005,
)

task = MultiTaskLitModule(
    ("MaterialsProjectDataset", task_energy),
    ("MaterialsProjectDataset", task_force),
)



dm = MatSciMLDataModule(
    "MaterialsProjectDataset",
    train_path=TRAIN_PATH,
    val_split=VAL_PATH,
    dset_kwargs={
        "transforms": [
            UnitCellCalculator(),
            PointCloudToGraphTransform(
                "pyg",
                cutoff_dist=20.0,
                node_keys=["pos", "atomic_numbers"],
            ),
            FrameAveraging(frame_averaging="3D", fa_method="stochastic"),
        ],
    },
)
    # dm.setup()

wandb.init(project='faenet_debug', entity='smiret', mode='online', )
logger = WandbLogger(log_model="all", name=f"faenet-{DATASET}-data", save_dir='/workspace/nosnap/matsciml/mace_train')

mc = ModelCheckpoint(monitor="val_energy", save_top_k=5)


# run a quick training loop
trainer = pl.Trainer(accelerator="gpu",
        devices=1,
        min_epochs=20,
        max_epochs=100,
        log_every_n_steps=100,
        strategy="ddp_find_unused_parameters_true",
        logger=logger,
        callbacks=[
            GradientCheckCallback(),
            mc,
        ],)
trainer.fit(task, datamodule=dm)


########################################################################################
########################################################################################


# # construct Materials Project band gap regression with PyG implementation of FAENet
# task = ScalarRegressionTask(
#     encoder_class=FAENet,
#     encoder_kwargs={
#         "pred_as_dict": False,
#         "hidden_dim": 128,
#         "out_dim": 64,
#         "tag_hidden_channels": 0,
#         "input_dim": 128,
#     },
#     output_kwargs={"lazy": False, "input_dim": 64, "hidden_dim": 64},
#     task_keys=["band_gap"],
# )

# dm = MatSciMLDataModule.from_devset(
#     "MaterialsProjectDataset",
#     dset_kwargs={
#         "transforms": [
#             UnitCellCalculator(),
#             PointCloudToGraphTransform(
#                 "pyg",
#                 cutoff_dist=20.0,
#                 node_keys=["pos", "atomic_numbers"],
#             ),
#             FrameAveraging(frame_averaging="3D", fa_method="stochastic"),
#         ],
#     },
# )

# # run a quick training loop
# trainer = pl.Trainer(fast_dev_run=10)
# trainer.fit(task, datamodule=dm)
