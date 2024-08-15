import argparse
import subprocess
import time
from copy import deepcopy
from pathlib import Path

import matsciml
import mp_tests
import yaml
from matsciml.interfaces.ase import MatSciMLCalculator
from mp_tests.calculator import MatSciMLCalculator as MPTestsCalculator
from mp_tests.utils import mp_species

from experiments.utils.configurator import configurator
from experiments.utils.utils import _get_next_version, instantiate_arg_dict


def get_calculator(calc):
    if calc == "matsciml":
        return MatSciMLCalculator
    if calc == "mp":
        return MPTestsCalculator
    if calc == "kusp":
        return "KUSP__MO_000000000000_000"
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--calculator", default="matsciml")
    parser.add_argument("-m", "--model", default=None)
    parser.add_argument("-p", "--checkpoint", default=None)
    parser.add_argument("-t", "--task", default="ForceRegressionTask")
    parser.add_argument("-s", "--test", default="Elasticity")
    parser.add_argument("-j", "--job-n", default=0)
    parser.add_argument("-n", "--n-calcs", default=10_733)
    parser.add_argument("-i", "--it", default=10_000)
    parser.add_argument(
        "--dataset_config",
        type=Path,
        default=Path(__file__).parent.joinpath("configs", "datasets"),
        help="Dataset config folder or yaml file to use.",
    )
    parser.add_argument(
        "--trainer_config",
        type=Path,
        default=Path(__file__).parent.joinpath("configs", "trainer"),
        help="Trainer config folder or yaml file to use.",
    )
    parser.add_argument(
        "--model_config",
        type=Path,
        default=Path(__file__).parent.joinpath("configs", "models"),
        help="Model config folder or yaml file to use.",
    )
    args = parser.parse_args()

    # python mp_test_runner.py -c matsciml -m mace_pyg -p 2023-12-10-mace-128-L0_epoch-199.model
    # python mp_test_runner.py -c matsciml -m egnn_dgl -p ../experiment_logs/egnn_dgl_lips/version_4/lightning_logs/version_0/checkpoints/epoch=13-step=3836.ckpt
    # python mp_test_runner.py -c matsciml -m chgnet_dgl -p /store/code/ai4science/matgl/pretrained_models/CHGNet-MPtrj-2024.2.13-PES-11M/model.pt

    # python kusp_server.py --model egnn_dgl --checkpoint ../experiment_logs/egnn_dgl_lips/version_4/lightning_logs/version_0/checkpoints/epoch=13-step=3836.ckpt

    configurator.configure_models(args.model_config)
    configurator.configure_datasets(args.dataset_config)
    configurator.configure_trainer(args.trainer_config)

    log_dir_base = Path(*["calculator_results", args.calculator, args.test, args.model])
    log_dir = log_dir_base.joinpath(_get_next_version(log_dir_base))
    log_dir.mkdir(parents=True, exist_ok=True)

    db_name = log_dir.joinpath("mp_tests.json")
    with open(log_dir.joinpath("cli_args.yaml"), "w") as f:
        yaml.safe_dump({k: str(v) for k, v in args.__dict__.items()}, f, indent=2)

    model_args = instantiate_arg_dict(deepcopy(configurator.models[args.model]))
    calc = get_calculator(args.calculator)

    if args.calculator == "matsciml" and args.task == "ForceRegressionTask":
        if "mace" not in args.model:
            calc = calc.from_pretrained_force_regression(
                args.checkpoint, transforms=model_args["transforms"]
            )
        else:
            from experiments.models.pretrained_mace import calc

    if args.calculator == "mp":
        if "mace" not in args.model:
            model = getattr(matsciml.models, args.task)
            transforms = model_args.pop("transforms")
            model = model.load_from_checkpoint(
                args.checkpoint,
            )
        else:
            from experiments.models.pretrained_mace import model
        calc = calc(model, transforms=transforms)

    if args.calculator == "kusp":
        subprocess.Popen(
            [
                "python",
                "./kusp/kusp_server.py",
                "--model",
                args.model,
                "--checkpoint",
                args.checkpoint,
            ]
        )
        time.sleep(60)

    test = getattr(mp_tests, args.test)
    test = test(calc, supported_species=mp_species, db_name=db_name)

test.mp_tests(
    it=args.it,
    job_n=args.job_n,
    n_calcs=args.n_calcs,
    ignore_relax=True,
    method="stress-condensed-fast",
    only_force=True,
    ftol=0.001,
    # optimize=True,
    # only_optimize=True,
)


# import multiprocessing
# from multiprocessing import Pool


# def run_tests(job_n_list):
#     db_name = log_dir.joinpath(f"mp_tests_{job_n_list}.json")
#     test = getattr(mp_tests, args.test)
#     test = test(calc, supported_species=mp_species, db_name=db_name)
#     test.mp_tests(
#         it=args.it,
#         job_n=job_n_list,
#         n_calcs=args.n_calcs,
#         ignore_relax=True,
#         method="energy-condensed",
#         optimize=True,
#         only_optimize=True,
#     )


# n_cpu = min(os.cpu_count() - 1, 16)
# args.n_calcs = 10_733 // n_cpu

# pool = Pool(n_cpu)
# pool.map(run_tests, list(range(n_cpu)))
