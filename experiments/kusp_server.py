import sys

sys.path.insert(0, "/store/code/ai4science/kusp")

import argparse
from copy import deepcopy
from pathlib import Path

import matsciml
import numpy as np
import torch
from ase import Atoms
from kusp import KUSPServer
from matsciml.datasets import MaterialsProjectDataset
from pymatgen.io.ase import AseAtomsAdaptor

from experiments.utils.configurator import configurator
from experiments.utils.utils import instantiate_arg_dict


class PyMatGenDataset(MaterialsProjectDataset):
    def data_converter(self, config):
        pymatgen_structure = AseAtomsAdaptor.get_structure(config)
        data = {"structure": pymatgen_structure}
        return_dict = {}
        self._parse_structure(data, return_dict)
        for transform in self.transforms:
            return_dict = transform(return_dict)
        return_dict = self.collate_fn([return_dict])
        return return_dict


def raw_data_to_atoms(species, pos, contributing, cell, elem_map):
    contributing = contributing.astype(int)
    pos_contributing = pos[contributing == 1]
    species = np.array(list(map(lambda x: elem_map[x], species)))
    species = species[contributing == 1]
    atoms = Atoms(species, positions=pos_contributing, cell=cell, pbc=[1, 1, 1])
    return atoms


#########################################################################
#### Server
#########################################################################


class MatSciMLModelServer(KUSPServer):
    def __init__(self, model, dataset, configuration):
        super().__init__(model, configuration)
        self.cutoff = self.global_information.get("cutoff", 6.0)
        self.elem_map = self.global_information.get("elements")
        self.graph_in = None
        self.cell = self.global_information.get(
            "cell",
            np.array([[10, 0.0, 0.0], [0.0, 10, 0.0], [0.0, 0.0, 10]]),
        )
        if not isinstance(self.cell, np.ndarray):
            self.cell = np.array(self.cell)
        self.n_atoms = -1
        self.config = None
        self.dataset = dataset

    def prepare_model_inputs(self, atomic_numbers, positions, contributing_atoms):
        self.n_atoms = atomic_numbers.shape[0]
        config = raw_data_to_atoms(
            atomic_numbers, positions, contributing_atoms, self.cell, self.elem_map
        )
        data = self.dataset.data_converter(config)
        self.batch_in = data
        self.config = config
        self.contributing = torch.tensor(contributing_atoms, dtype=torch.float64)
        return {"batch": self.batch_in}

    def prepare_model_outputs(self, outputs):
        energy = (
            (outputs["energy"] * self.contributing)
            .double()
            .squeeze()
            .detach()
            .numpy()
            .sum()
        )
        force = outputs["force"].double().squeeze().detach().numpy()
        return {"energy": energy, "forces": force}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default=None)
    parser.add_argument("-p", "--checkpoint", default=None)
    parser.add_argument("-t", "--task", default="ForceRegressionTask")
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

    configurator.configure_models(args.model_config)
    configurator.configure_datasets(args.dataset_config)
    configurator.configure_trainer(args.trainer_config)

    model_args = instantiate_arg_dict(deepcopy(configurator.models[args.model]))

    model = getattr(matsciml.models, args.task)
    transforms = model_args.pop("transforms")
    model = model.load_from_checkpoint(
        args.checkpoint,
    )

    generic_dataset = PyMatGenDataset("./empty_lmdb", transforms=transforms)
    server = MatSciMLModelServer(
        model=model, dataset=generic_dataset, configuration="kusp_config.yaml"
    )
    server.serve()
