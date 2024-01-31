from __future__ import annotations

available_data = {
    "is2re": {
        "debug": {
            "batch_size": 4,
            "num_workers": 0,
            "train_path": "./matsciml/datasets/dev-is2re",
            "val_path": "./matsciml/datasets/dev-is2re",
        },
        "full_train": {
            "train_path": "/store/code/open-catalyst/data_lmdbs/is2re/all/train",
            "val_path": "/store/code/open-catalyst/data_lmdbs/is2re/all/val",
            "normalize_kwargs": {},
            "dset_kwargs": None,
        },
    },
    "s2ef": {
        "debug": {
            "batch_size": 4,
            "num_workers": 0,
            "train_path": "./matsciml/datasets/dev-s2ef",
            "val_path": "./matsciml/datasets/dev-s2ef",
        },
        "full_train": {
            "train_path": "/datasets-alt/open-catalyst/s2ef_train_2M/ref_energy_s2ef_train_2M_dgl_munch_edges/",
            "val_path": "/datasets-alt/open-catalyst/s2ef_val_id/ref_energy_munch_s2ef_val_id/",
            "normalize_kwargs": {},
        },
    },
    "generic": {"full_train": {"batch_size": 32, "num_workers": 32}},
}
