from __future__ import annotations

import sys

sys.path.append("/store/code/open-catalyst/private-repo/matsciml-fork")

from argparse import ArgumentParser

import pytorch_lightning as pl
from data_config import *
from trainer_config import *
from training_utils.utils import *

do_ip_setup()


def main(args):
    opt_target = f"val_{args.target}"
    log_path = os.path.join("./experiments-2024/", args.run_type, args.model, args.data)

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
        choices=["is2re", "s2ef", "cmd", "mp", "lips", "nomad", "oqmd"],
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
        )

    main(args)
