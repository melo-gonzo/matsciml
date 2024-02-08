from __future__ import annotations

from tqdm import tqdm

from matsciml.datasets import MaterialsProjectDataset
from matsciml.datasets.transforms import (
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
)

"""
This example shows how periodic boundary conditions can be wired
into the graphs via the transform pipeline interface.

We chain the `PeriodicPropertiesTransform`, which calculates the
offsets and images using Pymatgen, which provides the edge definitions
that are used by `PointCloudToGraphTransform`.
"""

dset = MaterialsProjectDataset(
    lmdb_root_path="/store/code/open-catalyst/data_lmdbs/gnome/train",
    transforms=[
        PeriodicPropertiesTransform(cutoff_radius=10),
        PointCloudToGraphTransform(backend="dgl"),
    ],
)
idx = 43851
dset.__getitem__(idx)
