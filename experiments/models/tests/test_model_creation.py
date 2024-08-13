from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

import experiments
from experiments.utils.configurator import configurator
from experiments.utils.utils import instantiate_arg_dict

base_path = Path(experiments.__file__).parent
model_path = base_path.joinpath("configs", "models")
configurator.configure_models(model_path)

models = list(configurator.models.keys())


@pytest.mark.parametrize("model", models)
def test_instantiate_model_dict(model):
    model_dict = configurator.models[model]
    instantiate_arg_dict(deepcopy(model_dict))
