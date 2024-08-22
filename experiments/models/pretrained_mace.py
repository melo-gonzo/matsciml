import torch
from e3nn.o3 import Irreps
from mace.modules import ScaleShiftMACE
from mace.modules.blocks import RealAgnosticResidualInteractionBlock
from matsciml.common.types import BatchDict
from matsciml.datasets.transforms import (
    PeriodicPropertiesTransform,
    PointCloudToGraphTransform,
)
from matsciml.interfaces.ase import MatSciMLCalculator
from matsciml.models.base import ScalarRegressionTask
from matsciml.models.pyg.mace import MACEWrapper
from matsciml.models.utils.io import *
from torch import nn


class MACEForwardModule(ScalarRegressionTask):
    def forward(self, batch):
        outputs = self.encoder(batch)
        return outputs


class OGMACE(MACEWrapper):
    def _forward(
        self,
        graph,
        node_feats: torch.Tensor,
        pos: torch.Tensor,
        **kwargs,
    ):
        mace_data = {
            "positions": pos,
            "node_attrs": node_feats,
            "ptr": graph.ptr,
            "cell": kwargs["cell"],
            "shifts": kwargs["shifts"],
            "batch": graph.batch,
            "edge_index": graph.edge_index,
        }
        outputs = self.encoder(
            mace_data,
            training=self.training,
            compute_force=True,
            compute_virials=False,
            compute_stress=False,
            compute_displacement=False,
        )
        return outputs

    def forward(self, batch: BatchDict):
        input_data = self.read_batch(batch)
        outputs = self._forward(**input_data)
        return outputs


def return_mace_calc():
    available_models = {
        "mace": {
            "encoder_class": OGMACE,
            "encoder_kwargs": {
                "mace_module": ScaleShiftMACE,
                "num_atom_embedding": 89,
                "r_max": 6.0,
                "num_bessel": 10,
                "num_polynomial_cutoff": 5.0,
                "max_ell": 3,
                "interaction_cls": RealAgnosticResidualInteractionBlock,
                "interaction_cls_first": RealAgnosticResidualInteractionBlock,
                "num_interactions": 2,
                "atom_embedding_dim": 128,
                "MLP_irreps": Irreps("16x0e"),
                "avg_num_neighbors": 61.964672446250916,
                "correlation": 3,
                "radial_type": "bessel",
                "gate": nn.SiLU(),
                "atomic_inter_scale": 0.804153875447809696197509765625,
                "atomic_inter_shift": 0.164096963591873645782470703125,
                "distance_transform": None,
                ###
                # fmt: off
                "atomic_energies": torch.Tensor(
                    [
                        -3.6672,
                        -1.3321,
                        -3.4821,
                        -4.7367,
                        -7.7249,
                        -8.4056,
                        -7.3601,
                        -7.2846,
                        -4.8965,
                        0.0000,
                        -2.7594,
                        -2.8140,
                        -4.8469,
                        -7.6948,
                        -6.9633,
                        -4.6726,
                        -2.8117,
                        -0.0626,
                        -2.6176,
                        -5.3905,
                        -7.8858,
                        -10.2684,
                        -8.6651,
                        -9.2331,
                        -8.3050,
                        -7.0490,
                        -5.5774,
                        -5.1727,
                        -3.2521,
                        -1.2902,
                        -3.5271,
                        -4.7085,
                        -3.9765,
                        -3.8862,
                        -2.5185,
                        6.7669,
                        -2.5635,
                        -4.9380,
                        -10.1498,
                        -11.8469,
                        -12.1389,
                        -8.7917,
                        -8.7869,
                        -7.7809,
                        -6.8500,
                        -4.8910,
                        -2.0634,
                        -0.6396,
                        -2.7887,
                        -3.8186,
                        -3.5871,
                        -2.8804,
                        -1.6356,
                        9.8467,
                        -2.7653,
                        -4.9910,
                        -8.9337,
                        -8.7356,
                        -8.0190,
                        -8.2515,
                        -7.5917,
                        -8.1697,
                        -13.5927,
                        -18.5175,
                        -7.6474,
                        -8.1230,
                        -7.6078,
                        -6.8503,
                        -7.8269,
                        -3.5848,
                        -7.4554,
                        -12.7963,
                        -14.1081,
                        -9.3549,
                        -11.3875,
                        -9.6219,
                        -7.3244,
                        -5.3047,
                        -2.3801,
                        0.2495,
                        -2.3240,
                        -3.7300,
                        -3.4388,
                        -5.0629,
                        -11.0246,
                        -12.2656,
                        -13.8556,
                        -14.9331,
                        -15.2828,
                    ]
                ).to(torch.double),
                "atomic_numbers": [
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                    11,
                    12,
                    13,
                    14,
                    15,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                    22,
                    23,
                    24,
                    25,
                    26,
                    27,
                    28,
                    29,
                    30,
                    31,
                    32,
                    33,
                    34,
                    35,
                    36,
                    37,
                    38,
                    39,
                    40,
                    41,
                    42,
                    43,
                    44,
                    45,
                    46,
                    47,
                    48,
                    49,
                    50,
                    51,
                    52,
                    53,
                    54,
                    55,
                    56,
                    57,
                    58,
                    59,
                    60,
                    61,
                    62,
                    63,
                    64,
                    65,
                    66,
                    67,
                    68,
                    69,
                    70,
                    71,
                    72,
                    73,
                    74,
                    75,
                    76,
                    77,
                    78,
                    79,
                    80,
                    81,
                    82,
                    83,
                    89,
                    90,
                    91,
                    92,
                    93,
                    94,
                ],
                # fmt: on
            },
            "output_kwargs": {"lazy": False, "input_dim": 256, "hidden_dim": 256},
        }
    }

    ckpt = "mp_test_checkpoints/2023-12-10-mace-128-L0_epoch-199.model"

    model = MACEForwardModule(**available_models["mace"])
    model.encoder.encoder.load_state_dict(
        torch.load(ckpt, map_location=torch.device("cpu")).state_dict(), strict=True
    )
    model = model.to(torch.double)
    transforms = [
        PeriodicPropertiesTransform(cutoff_radius=5.0, adaptive_cutoff=True),
        PointCloudToGraphTransform(
            "pyg",
            cutoff_dist=20.0,
            node_keys=["pos", "atomic_numbers"],
        ),
    ]

    calc = MatSciMLCalculator(model, transforms)
    return calc, model


calc, model = return_mace_calc()
