from __future__ import annotations

import argparse
import sys

sys.path.append("/workspace/ai-mat-top/matsciml_top/forks/carmelo_matsciml/")

import e3nn

# Atomic Energies table
import mendeleev
import pytest
import pytorch_lightning as pl
from mendeleev.fetch import fetch_ionization_energies
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
from tqdm import tqdm

from matsciml.datasets import transforms

sys.path.append(
    "/workspace/ai-mat-top/matsciml_top/forks/carmelo_matsciml/",
)  # Path to matsciml directory(or matsciml installed as package )
from matsciml.datasets.lips import LiPSDataset, lips_devset
from matsciml.datasets.transforms import (
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
)
from matsciml.lightning.callbacks import GradientCheckCallback
from matsciml.lightning.data_utils import MatSciMLDataModule
from matsciml.models.base import MaceEnergyForceTask
from matsciml.models.pyg.mace import data, modules, tools
from matsciml.models.pyg.mace.modules.blocks import *
from matsciml.models.pyg.mace.modules.models import ScaleShiftMACE
from matsciml.models.pyg.mace.modules.utils import compute_mean_std_atomic_inter_energy
from matsciml.models.pyg.mace.tools import atomic_numbers_to_indices, to_one_hot

pl.seed_everything(6)


atomic_energies = fetch_ionization_energies(degree=list(range(1, 100))).sum(axis=1)
atomic_energies *= -1
atomic_energies = torch.Tensor(list(atomic_energies[:100].to_dict().values()))
# atomic_energies = torch.ones_like(atomic_energies)



def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.cpu().detach().numpy()


### Gnome
# pre_compute_params = {
#     "mean": 64690.4765625,
#     "std": 42016.30859375,
#     "avg_num_neighbors": 25.7051,
# }
# DATASET = "gnome"
# TRAIN_PATH = "/store/code/open-catalyst/data_lmdbs/gnome/train"
# VAL_PATH = "/store/code/open-catalyst/data_lmdbs/gnome/val"


### MP-Traj
# pre_compute_params = {
#     "mean": 27179.298828125,
#     "std": 28645.603515625,
#     "avg_num_neighbors": 52.0138,
# }
# DATASET = "mp-traj"
# TRAIN_PATH = "/store/code/open-catalyst/data_lmdbs/mp-traj/train"
# VAL_PATH = "/store/code/open-catalyst/data_lmdbs/mp-traj/val"


### Combined Datasets
# pre_compute_params = {
#     "mean": 59693.9375,
#     "std": 45762.0234375,
#     "avg_num_neighbors": 34.1558,
# }


### Combined Datasets
pre_compute_params = {
    "mean": 1000,
    "std": 1,
    "avg_num_neighbors": 34.1558,
}

DATASET = "multi"
# TRAIN_PATH = "/store/code/open-catalyst/data_lmdbs/mp-traj-gnome-combo/train"
# VAL_PATH = "/store/code/open-catalyst/data_lmdbs/mp-traj-gnome-combo/val"
# TRAIN_PATH = "/datasets-alt/molecular-data/mat_traj/mp-traj-gnome-combo/train"
# VAL_PATH = "/datasets-alt/molecular-data/mat_traj/mp-traj-gnome-combo/val"
TRAIN_PATH = "/datasets-alt/molecular-data/mat_traj/mp-traj-full/train"
VAL_PATH = "/datasets-alt/molecular-data/mat_traj/mp-traj-full/val"


def main(args):
    # Load Data
    dm = MatSciMLDataModule(
        "MaterialsProjectDataset",
        train_path=TRAIN_PATH,
        val_split=VAL_PATH,
        dset_kwargs={
            "transforms": [
                PeriodicPropertiesTransform(cutoff_radius=10.0),
                PointCloudToGraphTransform("pyg", cutoff_dist=args.r_max),
            ],
        },
        batch_size=16,
        num_workers=16,
    )

    dm.setup()
    train_loader = dm.train_dataloader()
    dataset_iter = iter(train_loader)
    batch = next(dataset_iter)

    atomic_numbers = torch.arange(0, 100)

    atomic_inter_shift = pre_compute_params["mean"]
    atomic_inter_scale = pre_compute_params["std"]
    avg_num_neighbors = pre_compute_params["avg_num_neighbors"]

    # Load Model
    model_config = dict(
        r_max=args.r_max,
        num_bessel=args.num_bessel,
        num_polynomial_cutoff=args.num_polynomial_cutoff,
        max_ell=args.max_ell,
        # max_L=args.max_L,
        # num_channels=args.num_channels,
        # num_radial_basis=args.num_radial_basis,
        # scaling=args.scaling,
        interaction_cls=RealAgnosticResidualInteractionBlock,
        num_interactions=args.num_interactions,
        num_elements=len(atomic_numbers),
        hidden_irreps=e3nn.o3.Irreps(args.hidden_irreps),
        atomic_energies=atomic_energies,
        avg_num_neighbors=avg_num_neighbors,
        atomic_numbers=atomic_numbers,
        correlation=args.correlation_order,
        gate=torch.nn.functional.silu,
        interaction_cls_first=RealAgnosticResidualInteractionBlock,
        MLP_irreps=e3nn.o3.Irreps(args.MLP_irreps),
        atomic_inter_scale=atomic_inter_scale,
        atomic_inter_shift=atomic_inter_shift,
        training=True,
    )

    task = MaceEnergyForceTask(
        encoder_class=ScaleShiftMACE,
        encoder_kwargs=model_config,
        task_keys=["energy", "force"],
        output_kwargs={
            "energy": {
                "block_type": "IdentityOutputBlock",
                "output_dim": 1,
                "hidden_dim": None,
            },
            "force": {
                "block_type": "IdentityOutputBlock",
                "output_dim": 3,
                "hidden_dim": None,
            },
        },
        loss_coeff={"energy": 1.0, "force": 10.0},
        lr=0.005,
        weight_decay=1e-8,

    )

    # Print model
    print(task)

    # Start Training
    # logger = CSVLogger(save_dir="./mace_experiments")
    wandb.init(project='mace_debug-full-traj-128', entity='m3rg', mode='online', )
    logger = WandbLogger(log_model="all", name=f"mace-{DATASET}-data", save_dir='/workspace/nosnap/matsciml/mace_train')

    mc = ModelCheckpoint(monitor="val_force", save_top_k=5)

    trainer = pl.Trainer(
        max_epochs=200,
        min_epochs=20,
        log_every_n_steps=100,
        accelerator="gpu",
        devices=8,
        strategy="ddp_find_unused_parameters_true",
        logger=logger,
        callbacks=[
            GradientCheckCallback(),
            mc,
        ],
        gradient_clip_val=100, 
        gradient_clip_algorithm="value"
    )

    trainer.fit(task, datamodule=dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MACE Training script")
    parser.add_argument("--r_max", type=float, default=6.0, help="Neighbor cutoff")
    parser.add_argument("--max_ell", type=int, default=3, help="Spherical harmonic Lmax")
    parser.add_argument(
        "--num_bessel",
        type=int,
        default=3,
        help="Bessel embeding size",
    )
    parser.add_argument(
        "--num_polynomial_cutoff",
        type=int,
        default=5,
        help="Radial basis polynomial cutoff",
    )
    parser.add_argument(
        "--num_interactions",
        type=int,
        default=2,
        help="No. of interaction layers",
    )
    parser.add_argument(
        "--hidden_irreps",
        type=str,
        default="128x0e+128x1o+128x2e",
        help="Hidden Irrep Shape",
    )
    parser.add_argument(
        "--correlation_order",
        type=int,
        default=3,
        help="Correlation Order",
    )
    parser.add_argument(
        "--MLP_irreps",
        type=str,
        default="16x0e",
        help="Irreps of Non-linear readout block",
    )

    parser.add_argument(
        "--max_L",
        type=int,
        default=2,
        help="max_L",
    )

    parser.add_argument(
        "--num_channels",
        type=int,
        default=128,
        help="num_channels",
    )

    parser.add_argument(
        "--num_radial_basis",
        type=int,
        default=10,
        help="num_radial_basis",
    )

    parser.add_argument(
        "--scaling",
        type=str,
        default="rms_forces_scaling",
        help="scaling parameter",
    )

    parser.add_argument("--max_epochs", type=int, default=100, help="Max epochs")

    args = parser.parse_args()
    main(args)
