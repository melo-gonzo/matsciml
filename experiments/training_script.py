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
    print("fix here main")
    opt_target = f"val_{args.targets[0]}"
    log_path = os.path.join(
        "./experiments-2024/", args.run_type, args.model, "-".join(args.data)
    )
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

    # for idx, target in enumerate(args.targets):
    #     for data in args.data:
    #         if target not in data_targets[data]:
    #             raise Exception(
    #                 f"Requested target {target} not available in {data} dataset.",
    #                 f"Available keys are: {data_targets[data]}",
    #             )

    main(args)

# python experiments/training_script.py --model faenet --data mp-traj --task sr --targets corrected_total_energy force --debug
# MultiTask single Dataset
# python experiments/training_script.py --model faenet --data gnome --task sr gffr --targets energy force --debug
