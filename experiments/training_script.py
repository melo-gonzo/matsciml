from __future__ import annotations

import os
import sys

cg_msl = "/store/code/open-catalyst/public-repo/matsciml"

if os.path.exists(cg_msl):
    sys.path.append(cg_msl)

sm_msl = "/workspace/ai-mat-top/matsciml_top/forks/carmelo_matsciml/"

if os.path.exists(sm_msl):
    sys.path.append(sm_msl)

from argparse import ArgumentParser

from data_config import *
from trainer_config import *
from training_utils.utils import *

do_ip_setup()


def main(args, log_path):
    check_args(args, data_targets)
    print("fix here main")
    if len(args.targets) > 1:
        opt_target = "val.total_loss"
    else:
        if args.targets[0] == "symmetry_group":
            opt_target = f"val_spacegroup"
        else:
            opt_target = f"val_{args.targets[0]}"
    os.makedirs(log_path, exist_ok=True)

    callbacks = setup_callbacks(opt_target, log_path)
    logger = setup_logger(log_path)

    dm = setup_datamodule(args)
    task = setup_task(args)
    trainer = setup_trainer(args, callbacks, logger)
    trainer.fit(task, datamodule=dm)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--model",
        required=True,
        choices=["egnn", "megnet", "faenet", "m3gnet", "gala", "mace", "tensornet"],
    )
    parser.add_argument(
        "--data",
        nargs="+",
        required=True,
        choices=[
            "is2re",
            "s2ef",
            "carolina",
            "materials-project",
            "lips",
            "nomad",
            "oqmd",
            "mp-traj",
            "gnome",
        ],
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        required=True,
        choices=["sr", "fr", "bc", "csc", "mef", "gffr"],
        help="ScalarRegressionTask\nForceRegressionTask\nBinaryClassificationTask\nCrystalSymmetryClassificationTask\nMaceEnergyForceTask\nGradFreeForceRegressionTask",
    )

    parser.add_argument(
        "--targets",
        nargs="+",
        required=True,
        default="energy",
    )

    parser.add_argument("--gpus", default=1, help="Number of gpu's to use")

    args = parser.parse_args()
    if args.debug:
        args.run_type = "debug"
    else:
        args.run_type = "experiment"

    log_path = os.path.join(
        "./experiments-2024-logs-full-runs/",
        args.run_type,
        args.model,
        "-".join(args.data),
        "-".join(args.targets),
    )

    try:
        main(args, log_path)
    except Exception as e:
        error_log(e, log_path)


# Single Task Single Dataset
# python experiments/training_script.py --model egnn --data materials-project --task sr --targets formation_energy_per_atom --gpus 1
# python experiments/training_script.py --model egnn --data oqmd --task sr --targets energy --gpus 1
# python experiments/training_script.py --model egnn --data nomad --task sr --targets relative_energy --gpus 1
# python experiments/training_script.py --model egnn --data carolina --task sr --targets energy --gpus 1

# python experiments/training_script.py --model megnet --data materials-project --task sr --targets formation_energy_per_atom --gpus 1
# python experiments/training_script.py --model megnet --data oqmd --task sr --targets energy --gpus 1
# python experiments/training_script.py --model megnet --data nomad --task sr --targets relative_energy --gpus 1
# python experiments/training_script.py --model megnet --data carolina --task sr --targets energy --gpus 1

# python experiments/training_script.py --model egnn --data materials-project --task sr --targets band_gap --gpus 1
# python experiments/training_script.py --model egnn --data materials-project --task sr --targets efermi --gpus 1
# python experiments/training_script.py --model egnn --data materials-project --task bc --targets is_stable --gpus 1
# python experiments/training_script.py --model egnn --data materials-project --task csc --targets symmetry_group --gpus 1

# python experiments/training_script.py --model megnet --data materials-project --task sr --targets band_gap --gpus 1
# python experiments/training_script.py --model megnet --data materials-project --task sr --targets efermi --gpus 1
# python experiments/training_script.py --model megnet --data materials-project --task bc --targets is_stable --gpus 1
# python experiments/training_script.py --model megnet --data materials-project --task csc --targets symmetry_group --gpus 1


# MultiTask Single Dataset
# python experiments/training_script.py --model egnn --data materials-project --task sr sr --targets band_gap efermi --gpus 1
# python experiments/training_script.py --model egnn --data materials-project --task sr bc --targets band_gap is_stable --gpus 1
# python experiments/training_script.py --model egnn --data materials-project --task sr csc --targets band_gap symmetry_group --gpus 1
# python experiments/training_script.py --model egnn --data materials-project --task sr bc --targets efermi is_stable --gpus 1
# python experiments/training_script.py --model egnn --data materials-project --task sr csc --targets efermi symmetry_group --gpus 1
# python experiments/training_script.py --model egnn --data materials-project --task bc csc --targets is_stable symmetry_group --gpus 1

# python experiments/training_script.py --model megnet --data materials-project --task sr sr --targets band_gap efermi --gpus 1
# python experiments/training_script.py --model megnet --data materials-project --task sr bc --targets band_gap is_stable --gpus 1
# python experiments/training_script.py --model megnet --data materials-project --task sr csc --targets band_gap symmetry_group --gpus 1
# python experiments/training_script.py --model megnet --data materials-project --task sr bc --targets efermi is_stable --gpus 1
# python experiments/training_script.py --model megnet --data materials-project --task sr csc --targets efermi symmetry_group --gpus 1
# python experiments/training_script.py --model megnet --data materials-project --task bc csc --targets is_stable symmetry_group --gpus 1


# Multi Data Multi Task

# python experiments/training_script.py --model egnn --data materials-project nomad --task sr sr --targets formation_energy_per_atom relative_energy --gpus 1
# python experiments/training_script.py --model egnn --data materials-project s2ef --task sr sr --targets formation_energy_per_atom energy --gpus 1
# python experiments/training_script.py --model egnn --data materials-project is2re --task sr sr --targets formation_energy_per_atom energy_init --gpus 1
# python experiments/training_script.py --model egnn --data materials-project lips --task sr sr --targets formation_energy_per_atom energy --gpus 1

# python experiments/training_script.py --model megnet --data materials-project nomad --task sr sr --targets formation_energy_per_atom relative_energy --gpus 1
# python experiments/training_script.py --model megnet --data materials-project s2ef --task sr sr --targets formation_energy_per_atom energy --gpus 1
# python experiments/training_script.py --model megnet --data materials-project is2re --task sr sr --targets formation_energy_per_atom energy_init --gpus 1
# python experiments/training_script.py --model megnet --data materials-project lips --task sr sr --targets formation_energy_per_atom energy --gpus 1
