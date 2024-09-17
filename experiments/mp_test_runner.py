# import sys
# sys.path.insert(0, "/store/code/ai4science/mp_tests")
# sys.path.insert(0, "/store/code/ai4science/matsciml")

import argparse
import subprocess
import time
from copy import deepcopy
from pathlib import Path
import torch

import matsciml
import mp_tests
import yaml
from matsciml.interfaces.ase import MatSciMLCalculator
from mp_tests.calculator import MatSciMLCalculator as MPTestsCalculator
from mp_tests.utils import mp_species


from experiments.utils.configurator import configurator
from experiments.utils.utils import _get_next_version, instantiate_arg_dict


torch.set_default_dtype(torch.float64)


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
    parser.add_argument("-p", "--checkpoint", default=None, type=str)
    parser.add_argument("-t", "--task", default="ForceRegressionTask")
    parser.add_argument("-s", "--test", default="Elasticity")
    parser.add_argument("-j", "--job-n", default=0, type=int)
    parser.add_argument("-n", "--n-calcs", default=10_733, type=int)
    parser.add_argument("-i", "--it", default=500, type=int)
    parser.add_argument("-e", "--extra_logger_name", default="debug")
    parser.add_argument("-f", "--ftol", default=0.001, type=float)
    parser.add_argument("-k", "--pkl", default=None)
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
    # python mp_test_runner.py -c matsciml -m m3gnet_dgl -p /store/code/ai4science/matgl/pretrained_models/M3GNet-MP-2021.2.8-PES/model.pt
    # python kusp_server.py --model egnn_dgl --checkpoint ../experiment_logs/egnn_dgl_lips/version_4/lightning_logs/version_0/checkpoints/epoch=13-step=3836.ckpt

    # python mp_test_runner.py --calculator matsciml --model mace_pyg --checkpoint 2023-12-10-mace-128-L0_epoch-199.model --job-n '0' --n_calcs '215'

    configurator.configure_models(args.model_config)
    configurator.configure_datasets(args.dataset_config)
    configurator.configure_trainer(args.trainer_config)
    if args.extra_logger_name is not None:
        log_dir_base = Path(
            *[
                "calculator_results",
                args.calculator,
                args.extra_logger_name,
                args.test,
                args.model,
                f"job_n-{args.job_n}",
            ]
        )
    else:
        log_dir_base = Path(
            *[
                "calculator_results",
                args.calculator,
                args.test,
                args.model,
                args.job_n,
            ]
        )
    log_dir = log_dir_base.joinpath(_get_next_version(log_dir_base))
    log_dir.mkdir(parents=True, exist_ok=True)

    db_name = log_dir.joinpath("mp_tests.json")
    with open(log_dir.joinpath("cli_args.yaml"), "w") as f:
        command = "python mp_test_runner.py " + " ".join(
            f"--{k} {v}" for k, v in vars(args).items()
        )
        args.command = command
        yaml.safe_dump({k: str(v) for k, v in args.__dict__.items()}, f, indent=2)

    model_args = instantiate_arg_dict(deepcopy(configurator.models[args.model]))
    calc = get_calculator(args.calculator)

    if args.calculator == "matsciml" and args.task == "ForceRegressionTask":
        if args.model in ["chgnet_dgl", "m3gnet_dgl"]:
            from models.matgl_pretrained import load_matgl

            model = load_matgl(args.checkpoint, model_args)
            model = model.to(torch.double)
            calc = MatSciMLCalculator(
                model, transforms=model_args["transforms"], from_matsciml=False
            )
        elif "mace" not in args.model:
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
            model = model.to(torch.double)
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

    try:
        test.mp_tests(
            it=args.it,
            job_n=args.job_n,
            n_calcs=args.n_calcs,
            pkl=args.pkl,
            ignore_relax=True,
            method="stress-condensed-fast",
            only_force=True,
            ftol=args.ftol,
        )
    except Exception:
        import traceback

        with open(log_dir.joinpath("error.txt"), "w") as f:
            f.write("\n" + str(traceback.format_exc()))
            print(traceback.format_exc())
