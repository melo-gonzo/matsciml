from __future__ import annotations

import json
import os
import sys
from copy import deepcopy


cg_msl = "/store/code/open-catalyst/public-repo/matsciml"

if os.path.exists(cg_msl):
    sys.path.append(cg_msl)

sm_msl = "/workspace/ai-mat-top/matsciml_top/forks/carmelo_matsciml/"

if os.path.exists(sm_msl):
    sys.path.append(sm_msl)



from matsciml.datasets import *
from matsciml.datasets.transforms import (
    DistancesTransform,
    FrameAveraging,
    GraphVariablesTransform,
    MGLDataTransform,
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
    COMShift,
)
from matsciml.lightning.data_utils import (
    MatSciMLDataModule,
    MultiDataModule,
    MultiDataset,
)

data_keys = [
    "is2re",
    "s2ef",
    "lips",
    "carolina",
    "materials-project",
    "nomad",
    "oqmd",
    "symmetry",
    "mp-traj",
    "gnome",
    "mp-gnome",
    "generic",
]

norm_files = os.listdir("./matsciml/datasets/norms")
norm_dict = {}
for data_name in data_keys:
    norm_dict[data_name] = None
    for file in norm_files:
        if data_name in file:
            norm_dict[data_name] = json.load(
                open(os.path.join("./matsciml/datasets/norms", file))
            )


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
            "train_path": "/datasets-alt/open-catalyst/carmelo_copy_is2re/is2re/all/train",
            "val_split": "/datasets-alt/open-catalyst/carmelo_copy_is2re/is2re/all/val_id",
            "normalize_kwargs": norm_dict["is2re"],
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
            "train_path": "/datasets-alt/open-catalyst/s2ef_train_200K/ref_energy_s2ef_train_200K_dgl_munch_edges/",
            "val_split": "/datasets-alt/open-catalyst/s2ef_val_id/ref_energy_munch_s2ef_val_id/",
            "normalize_kwargs": norm_dict["s2ef"],
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
            "normalize_kwargs": norm_dict["lips"],
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
            "normalize_kwargs": norm_dict["carolina"],
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
            "normalize_kwargs": norm_dict["materials-project"],
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
            "normalize_kwargs": norm_dict["nomad"],
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
            "normalize_kwargs": norm_dict["oqmd"],
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
            "train_path": "/store/nosnap/chem-ai/mp-traj-full/devset",
            "val_split": "/store/nosnap/chem-ai/mp-traj-full/devset",
        },
        "experiment": {
            "train_path": "/datasets-alt/molecular-data/mat_traj/mp-traj-full/train",
            "val_split": "/datasets-alt/molecular-data/mat_traj/mp-traj-full/val",
            "test_split": "/datasets-alt/molecular-data/mat_traj/mp-traj-full/test",
            "normalize_kwargs": norm_dict["mp-traj"],
            "batch_size": 4,
        },
    },
    "gnome": {
        "dataset": "GnomeMaterialsProjectDataset",
        "debug": {
            "batch_size": 4,
            "num_workers": 0,
            "train_path": "/store/code/open-catalyst/data_lmdbs/gnome/devset",
            "val_split": "/store/code/open-catalyst/data_lmdbs/gnome/devset",
        },
        "experiment": {
            "train_path": "/datasets-alt/molecular-data/gnome/train",
            "val_split": "/datasets-alt/molecular-data/gnome/val",
            "test_split": "/datasets-alt/molecular-data/data_lmdbs/gnome/test",
            "normalize_kwargs": norm_dict["gnome"],
        },
    },
    "mp-gnome": {
        "dataset": "MaterialsProjectDataset",
        "debug": {
            "batch_size": 4,
            "num_workers": 0,
            "train_path": "/store/code/open-catalyst/data_lmdbs/gnome/devset",
            "val_split": "/store/code/open-catalyst/data_lmdbs/gnome/devset",
        },
        "experiment": {
            "train_path": "/datasets-alt/molecular-data/mat_traj/mp-traj-gnome-combo/train",
            "val_split": "/datasets-alt/molecular-data/mat_traj/mp-traj-gnome-combo/val",
            "test_split": "/datasets-alt/molecular-data/mat_traj/mp-traj-gnome-combo/test",
            "normalize_kwargs": norm_dict["mp-traj"],
            "batch_size": 4,
        },
    },
    "generic": {"experiment": {"batch_size": 4, "num_workers": 16}},
}


transforms = {
    "egnn": [
        PeriodicPropertiesTransform(cutoff_radius=6.5, adaptive_cutoff=True),
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
    "gala": [
        COMShift(),
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
    if len(args.data) == 1:
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
    else:
        train_dset_list = []
        val_dset_list = []
        for data in args.data:
            dset = deepcopy(available_data[data])
            dm_kwargs = deepcopy(available_data["generic"]["experiment"])
            dset[args.run_type].pop("normalize_kwargs", None)
            dm_kwargs.update(dset[args.run_type])
            dataset_name = dset["dataset"]
            dataset = getattr(sys.modules[__name__], dataset_name)
            model_transforms = transforms[args.model]
            train_dset_list.append(
                dataset(dm_kwargs["train_path"], transforms=model_transforms)
            )
            val_dset_list.append(
                dataset(dm_kwargs["val_split"], transforms=model_transforms)
            )

        train_dset = MultiDataset(train_dset_list)
        val_dset = MultiDataset(val_dset_list)
        dm = MultiDataModule(
            train_dataset=train_dset,
            val_dataset=val_dset,
            batch_size=dm_kwargs["batch_size"],
            num_workers=dm_kwargs["num_workers"],
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
    "mp-gnome": ["corrected_total_energy", "force"],
    "gnome": ["corrected_total_energy", "force"],
}
