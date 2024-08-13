from __future__ import annotations

from pathlib import Path

import matsciml
import matsciml.datasets.transforms  # noqa: F401
import pytest

import experiments
from experiments.datasets.data_module_config import setup_datamodule
from experiments.utils.configurator import configurator

base_path = Path(experiments.__file__).parent
model_path = base_path.joinpath("configs", "models")
datasets_path = base_path.joinpath("configs", "datasets")
trainer_path = base_path.joinpath("configs", "trainer")
configurator.configure_models(model_path)
configurator.configure_datasets(datasets_path)
configurator.configure_trainer(trainer_path)

single_task = {
    "model": "egnn_dgl",
    "dataset": {"oqmd": [{"task": "ScalarRegressionTask", "targets": ["band_gap"]}]},
}
multi_task = {
    "dataset": {
        "s2ef": [
            {"task": "ScalarRegressionTask", "targets": ["energy"]},
            {"task": "ForceRegressionTask", "targets": ["force"]},
        ]
    }
}
multi_data = {
    "model": "faenet_pyg",
    "dataset": {
        "oqmd": [{"task": "ScalarRegressionTask", "targets": ["energy"]}],
        "is2re": [
            {
                "task": "ScalarRegressionTask",
                "targets": ["energy_init", "energy_relaxed"],
            }
        ],
    },
}


@pytest.mark.parametrize("task_dict", [single_task, multi_task, multi_data])
def test_task_setup(task_dict):
    other_args = {"run_type": "debug", "model": "m3gnet_dgl", "cli_args": None}
    task_dict.update(other_args)
    setup_datamodule(config=task_dict)
