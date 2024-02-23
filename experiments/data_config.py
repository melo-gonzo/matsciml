from __future__ import annotations

import json
from copy import deepcopy

from matsciml.datasets.transforms import (
    DistancesTransform,
    FrameAveraging,
    GraphVariablesTransform,
    MGLDataTransform,
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
)
from matsciml.lightning.data_utils import MatSciMLDataModule

available_data = {
    "is2re": {
        "dataset": "IS2REDataset",
        "debug": {
            "batch_size": 4,
            "num_workers": 0,
            "train_path": "./matsciml/datasets/dev-is2re",
            "val_split": "./matsciml/datasets/dev-is2re",
        },
        "experiment": {
            "train_path": "/store/code/open-catalyst/data_lmdbs/is2re/all/train",
            "val_split": "/store/code/open-catalyst/data_lmdbs/is2re/all/val",
            "normalize_kwargs": json.load(
                open("./matsciml/datasets/norms/is2re-46032samples-norms.json")
            ),
        },
    },
    "s2ef": {
        "dataset": "S2EFDataset",
        "debug": {
            "batch_size": 4,
            "num_workers": 0,
            "train_path": "./matsciml/datasets/dev-s2ef",
            "val_split": "./matsciml/datasets/dev-s2ef",
        },
        "experiment": {
            "train_path": "/datasets-alt/open-catalyst/s2ef_train_2M/ref_energy_s2ef_train_2M_dgl_munch_edges/",
            "val_split": "/datasets-alt/open-catalyst/s2ef_val_id/ref_energy_munch_s2ef_val_id/",
            "normalize_kwargs": json.load(
                open("./matsciml/datasets/norms/s2ef-200000samples-norms.json")
            ),
        },
    },
    "lips": {
        "dataset": "LiPSDataset",
        "debug": {
            "batch_size": 4,
            "num_workers": 0,
            "train_path": "./matsciml/datasets/lips/devset",
            "val_split": "./matsciml/datasets/lips/devset",
        },
        "experiment": {
            "train_path": "/datasets-alt/molecular-data/lips/train",
            "val_split": "/datasets-alt/molecular-data/lips/val",
            "test_split": "/datasets-alt/molecular-data/lips/test",
            "normalize_kwargs": json.load(
                open("./matsciml/datasets/norms/lips-1750samples-norms.json")
            ),
        },
    },
    "carolina": {
        "dataset": "CMDataset",
        "debug": {
            "batch_size": 4,
            "num_workers": 0,
            "train_path": "./matsciml/datasets/carolina_db/devset/",
            "val_split": "./matsciml/datasets/carolina_db/devset/",
        },
        "experiment": {
            "train_path": "/datasets-alt/molecular-data/carolina_matdb/train",
            "val_split": "/datasets-alt/molecular-data/carolina_matdb/val",
            "test_split": "/datasets-alt/molecular-data/carolina_matdb/test",
            "normalize_kwargs": json.load(
                open("./matsciml/datasets/norms/carolina-16080samples-norms.json")
            ),
        },
    },
    "materials-project": {
        "dataset": "MaterialsProjectDataset",
        "debug": {
            "batch_size": 4,
            "num_workers": 0,
            "train_path": "./matsciml/datasets/materials_project/devset-full/",
            "val_split": "./matsciml/datasets/materials_project/devset-full/",
        },
        "experiment": {
            "train_path": "/datasets-alt/molecular-data/materials_project/train",
            "val_split": "/datasets-alt/molecular-data/materials_project/val",
            "test_split": "/datasets-alt/molecular-data/materials_project/test",
            "normalize_kwargs": json.load(
                open(
                    "./matsciml/datasets/norms/materials-project-11273samples-norms.json"
                )
            ),
        },
    },
    "nomad": {
        "dataset": "NomadDataset",
        "debug": {
            "batch_size": 4,
            "num_workers": 0,
            "train_path": "./matsciml/datasets/nomad/devset/",
            "val_split": "./matsciml/datasets/nomad/devset/",
        },
        "experiment": {
            "train_path": "/datasets-alt/molecular-data/nomad/train",
            "val_split": "/datasets-alt/molecular-data/nomad/val",
            "test_split": "/datasets-alt/molecular-data/nomad/test",
            "normalize_kwargs": json.load(
                open("./matsciml/datasets/norms/nomad-10826samples-norms.json")
            ),
        },
    },
    "oqmd": {
        "dataset": "OQMDDataset",
        "debug": {
            "batch_size": 4,
            "num_workers": 0,
            "train_path": "./matsciml/datasets/oqmd/devset/",
            "val_split": "./matsciml/datasets/oqmd/devset/",
        },
        "experiment": {
            "train_path": "/datasets-alt/molecular-data/oqmd/train",
            "val_split": "/datasets-alt/molecular-data/oqmd/val",
            "test_split": "/datasets-alt/molecular-data/oqmd/test",
            "normalize_kwargs": json.load(
                open("./matsciml/datasets/norms/oqmd-76666samples-norms.json")
            ),
        },
    },
    "symmetry": {
        "dataset": "SyntheticPointGroupDataset",
        "debug": {
            "batch_size": 4,
            "num_workers": 0,
            "train_path": "./matsciml/datasets/symmetry/devset/",
            "val_split": "./matsciml/datasets/symmetry/devset/",
        },
    },
    "mp-traj": {
        "dataset": "MaterialsProjectDataset",
        "debug": {
            "batch_size": 4,
            "num_workers": 0,
            "train_path": "/store/code/open-catalyst/data_lmdbs/mp-traj-full/devset",
            "val_split": "/store/code/open-catalyst/data_lmdbs/mp-traj-full/devset",
        },
        "experiment": {
            "train_path": "/store/code/open-catalyst/data_lmdbs/mp-traj-full/train",
            "val_split": "/store/code/open-catalyst/data_lmdbs/mp-traj-full/val",
            "test_split": "/store/code/open-catalyst/data_lmdbs/mp-traj-full/test",
            "normalize_kwargs": json.load(
                open("./matsciml/datasets/norms/mp-traj-10214samples-norms.json")
            ),
            "batch_size": 4,
        },
    },
    "gnome": {
        "dataset": "MaterialsProjectDataset",
        "debug": {
            "batch_size": 4,
            "num_workers": 0,
            "train_path": "/store/code/open-catalyst/data_lmdbs/gnome/devset",
            "val_split": "/store/code/open-catalyst/data_lmdbs/gnome/devset",
        },
        "experiment": {
            "train_path": "/store/code/open-catalyst/data_lmdbs/gnome/train",
            "val_split": "/store/code/open-catalyst/data_lmdbs/gnome/val",
            "test_split": "/store/code/open-catalyst/data_lmdbs/gnome/test",
            "normalize_kwargs": json.load(
                open("./matsciml/datasets/norms/gnome-26940samples-norms.json")
            ),
        },
    },
    "generic": {"experiment": {"batch_size": 32, "num_workers": 32}},
}

transforms = {
    "egnn": [
        PeriodicPropertiesTransform(cutoff_radius=6.5, adaptive_cutoff=True),
        PeriodicPropertiesTransform(cutoff_radius=10.0),
        PointCloudToGraphTransform(
            "dgl",
            cutoff_dist=20.0,
            node_keys=["pos", "atomic_numbers"],
        ),
    ],
    "faenet": [
        PeriodicPropertiesTransform(cutoff_radius=6.5, adaptive_cutoff=True),
        PointCloudToGraphTransform(
            "pyg",
            cutoff_dist=20.0,
            node_keys=["pos", "atomic_numbers"],
        ),
        FrameAveraging(frame_averaging="3D", fa_method="stochastic"),
    ],
    "megnet": [
        PeriodicPropertiesTransform(cutoff_radius=6.5, adaptive_cutoff=True),
        PointCloudToGraphTransform(
            "dgl",
            cutoff_dist=20.0,
            node_keys=["pos", "atomic_numbers"],
        ),
        DistancesTransform(),
        GraphVariablesTransform(),
    ],
    "m3gnet": [
        PeriodicPropertiesTransform(cutoff_radius=6.5, adaptive_cutoff=True),
        PointCloudToGraphTransform(
            "dgl",
            cutoff_dist=20.0,
            node_keys=["pos", "atomic_numbers"],
        ),
        MGLDataTransform(),
    ],
    "tensornet": [
        PeriodicPropertiesTransform(cutoff_radius=6.5, adaptive_cutoff=True),
        PointCloudToGraphTransform(
            "dgl",
            cutoff_dist=20.0,
            node_keys=["pos", "atomic_numbers"],
        ),
    ],
}


def setup_datamodule(args):
    if len(args.data) > 1:
        raise Exception("Cannot handle more than one dataset at the moment.")
    else:
        data = args.data[0]
    dset = deepcopy(available_data[data])
    dm_kwargs = deepcopy(available_data["generic"]["experiment"])
    dset[args.run_type].pop("normalize_kwargs", None)
    dm_kwargs.update(dset[args.run_type])
    dm = MatSciMLDataModule(
        dataset=dset["dataset"],
        dset_kwargs={"transforms": transforms[args.model]},
        **dm_kwargs,
    )
    return dm


data_targets = {
    "is2re": [
        "energy_init",
        "energy_relaxed",
    ],
    "s2ef": ["energy", "force"],
    "lips": ["energy", "force"],
    "carolina": ["energy", "symmetry_number", "symmetry_symbol"],
    "materials-project": [
        "is_magnetic",
        "is_metal",
        "is_stable",
        "band_gap",
        "efermi",
        "energy_per_atom",
        "formation_energy_per_atom",
        "uncorrected_energy_per_atom",
        "symmetry_number",
        "symmetry_symbol",
        "symmetry_group",
    ],
    "nomad": [
        "spin_polarized",
        "efermi",
        "relative_energy",
        "symmetry_number",
        "symmetry_symbol",
        "symmetry_group",
    ],
    "oqmd": ["band_gap", "energy", "stability"],
    "symmetry": [],
    "mp-traj": [
        "uncorrected_total_energy",
        "corrected_total_energy",
        "energy_per_atom",
        "ef_per_atom",
        "e_per_atom_relaxed",
        "ef_per_atom_relaxed",
        "force",
        "stress",
        "magmom",
        "bandgap",
    ],
    "gnome": ["corrected_total_energy", "force"],
}
