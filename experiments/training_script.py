from __future__ import annotations

import os
import sys

cg_msl = "/store/code/open-catalyst/public-repo/matsciml"

if os.path.exists(cg_msl):
    sys.path.append(cg_msl)

from argparse import ArgumentParser

from data_config import *
from trainer_config import *
from training_utils.utils import *

do_ip_setup()


def main(args):
    opt_target = f"val_{args.target}"
    log_path = os.path.join("./experiments-2024/", args.run_type, args.model, args.data)
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
        choices=["egnn", "megnet", "faenet", "m3gnet", "gala", "mace"],
    )
    parser.add_argument(
        "--data",
        required=True,
        choices=[
            "is2re",
            "s2ef",
            "cmd",
            "mp",
            "lips",
            "nomad",
            "oqmd",
            "mp-traj",
            "gnome",
        ],
    )

    parser.add_argument(
        "--task",
        required=True,
        choices=["sr", "fr", "bc", "csc", "mef", "gffr"],
        help="ScalarRegressionTask\nForceRegressionTask\nBinaryClassificationTask\nCrystalSymmetryClassificationTask\nMaceEnergyForceTask\nGradFreeForceRegressionTask",
    )

    parser.add_argument(
        "--target",
        required=True,
        default="energy",
    )

    parser.add_argument("--gpus", default=1, help="Number of gpu's to use")

    args = parser.parse_args()
    if args.debug:
        args.run_type = "debug"
    else:
        args.run_type = "experiment"

    if args.target not in data_targets[args.data]:
        raise Exception(
            f"Requested target {args.target} not available in {args.data} dataset.",
            f"Available keys are: {data_targets[args.data]}",
        )

    main(args)

# python experiments/training_script.py --model faenet --data mp-traj --task sr --target corrected_total_energy --gpus 4
