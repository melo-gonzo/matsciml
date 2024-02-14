from __future__ import annotations

from matsciml.datasets import *
from matsciml.datasets.transforms import (
    DistancesTransform,
    FrameAveraging,
    GraphToGraphTransform,
    GraphVariablesTransform,
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
    UnitCellCalculator,
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
            "normalize_kwargs": {},
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
            "normalize_kwargs": {},
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
            "normalize_kwargs": {},
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
            "train_path": "/store/code/open-catalyst/data_lmdbs/carolina_matdb/train",
            "val_split": "/store/code/open-catalyst/data_lmdbs/carolina_matdb/val",
            "test_split": "/store/code/open-catalyst/data_lmdbs/carolina_matdb/test",
            "normalize_kwargs": {},
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
            "train_path": "/store/code/open-catalyst/data_lmdbs/mp-project/train",
            "val_split": "/store/code/open-catalyst/data_lmdbs/mp-project/val",
            "test_split": "/store/code/open-catalyst/data_lmdbs/mp-project/test",
            "normalize_kwargs": {},
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
            "train_path": "/store/code/open-catalyst/data_lmdbs/nomad/train",
            "val_split": "/store/code/open-catalyst/data_lmdbs/nomad/val",
            "test_split": "/store/code/open-catalyst/data_lmdbs/nomad/test",
            "normalize_kwargs": {},
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
            "train_path": "/store/code/open-catalyst/data_lmdbs/oqmd/train",
            "val_split": "/store/code/open-catalyst/data_lmdbs/oqmd/val",
            "test_split": "/store/code/open-catalyst/data_lmdbs/oqmd/test",
            "normalize_kwargs": {},
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
            "normalize_kwargs": {},
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
            "normalize_kwargs": {},
        },
    },
    "generic": {"experiment": {"batch_size": 32, "num_workers": 32}},
}

transforms = {
    "egnn": [
        PeriodicPropertiesTransform(cutoff_radius=10.0),
        PointCloudToGraphTransform(
            "dgl",
            cutoff_dist=20.0,
            node_keys=["pos", "atomic_numbers"],
        ),
    ],
    "faenet": [
        PointCloudToGraphTransform(
            "pyg",
            cutoff_dist=20.0,
            node_keys=["pos", "atomic_numbers"],
        ),
        FrameAveraging(frame_averaging="3D", fa_method="stochastic"),
    ],
    "megnet": [
        PointCloudToGraphTransform(
            "dgl",
            cutoff_dist=20.0,
            node_keys=["pos", "atomic_numbers"],
        ),
        DistancesTransform(),
        GraphVariablesTransform(),
    ],
    "m3gnet": [],
}


def setup_datamodule(args):
    dset = available_data[args.data]
    dm_kwargs = available_data["generic"]["experiment"]
    normalize_kwargs = dset[args.run_type].pop("normalize_kwargs", None)
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
        "energy_total",
        "symmetry_number",
        "symmetry_symbol",
        "symmetry_group",
    ],
    "oqmd": ["band_gap", "energy", "stability"],
    "symmetry": [],
    "mp-traj": [
        "uncorrected_total_energy",
        "energy_per_atom",
        "ef_per_atom",
        "e_per_atom_relaxed",
        "ef_per_atom_relaxed",
        "force",
        "magmom",
        "bandgap",
    ],
    "gnome": ["corrected_total_energy", "force"],
}
