from __future__ import annotations

import sys
import os

cg_msl = "/store/code/open-catalyst/public-repo/matsciml"

if os.path.exists(cg_msl):
    sys.path.append(cg_msl)
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
    "/store/code/open-catalyst/data_lmdbs/materials_project/",
    transforms=[
        PeriodicPropertiesTransform(cutoff_radius=6.5, adaptive_cutoff=True),
        PointCloudToGraphTransform(backend="dgl"),
    ],
)

for idx in range(len(dset)):
    print(idx, end="\r")
    dset.__getitem__(idx)
