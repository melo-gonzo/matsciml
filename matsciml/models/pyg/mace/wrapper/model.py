from __future__ import annotations

from logging import getLogger
from typing import Any
from functools import cache

import torch
from e3nn.o3 import Irreps
from mace.modules import MACE

from matsciml.models.base import AbstractPyGModel
from matsciml.common.types import BatchDict, DataDict, AbstractGraph, Embeddings
from matsciml.common.registry import registry
from matsciml.common.inspection import get_model_required_args, get_model_all_args


__mace_required_args = get_model_required_args(MACE)
__mace_all_args = get_model_all_args(MACE)


logger = getLogger(__file__)

__all__ = ["MACEWrapper"]


@registry.register_model("MACEWrapper")
class MACEWrapper(AbstractPyGModel):
    def __init__(
        self,
        atom_embedding_dim: int,
        num_atom_embedding: int = 100,
        embedding_kwargs: Any = None,
        encoder_only: bool = True,
        **mace_kwargs,
    ) -> None:
        if embedding_kwargs is not None:
            logger.warning("`embedding_kwargs` is not used for MACE models.")
        super().__init__(atom_embedding_dim, num_atom_embedding, {}, encoder_only)
        for key in mace_kwargs:
            assert (
                key in __mace_all_args
            ), f"{key} was passed as a MACE kwarg but does not match expected arguments."
        # remove the embedding table, as MACE uses e3nn layers
        del self.atom_embedding
        if "num_elements" in mace_kwargs:
            raise KeyError(
                "Please use `num_atom_embedding` instead of passing `num_elements`."
            )
        if "hidden_irreps" in mace_kwargs:
            raise KeyError(
                "Please use `atom_embedding_dim` instead of passing `hidden_irreps`."
            )
        atom_embedding_dim = Irreps(f"{atom_embedding_dim}x0e")
        # pack stuff into the mace kwargs
        mace_kwargs["num_elements"] = num_atom_embedding
        mace_kwargs["hidden_irreps"] = atom_embedding_dim
        # check to make sure all that's required is
        for key in __mace_required_args:
            if key not in mace_kwargs:
                raise KeyError(
                    f"{key} is required by MACE, but was not found in kwargs."
                )
        self.encoder = MACE(**mace_kwargs)
        self.save_hyperparameters()

    @property
    @cache
    def _atom_eye(self) -> torch.Tensor:
        return torch.eye(
            self.hparams.num_atom_embedding, device=self.device, dtype=self.dtype
        )

    def atomic_numbers_to_one_hot(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        """
        Convert discrete atomic numbers into one-hot vectors based
        on some maximum number of elements possible.

        Parameters
        ----------
        atomic_numbers : torch.Tensor
            1D tensor of integers corresponding to atomic numbers.

        Returns
        -------
        torch.Tensor
            2D tensor of one-hot vectors for each node.
        """
        return self._atom_eye[atomic_numbers.long()]

    def read_batch(self, batch: BatchDict) -> DataDict:
        data = super().read_batch(batch)
        # expect a PyG graph already
        graph = batch["graph"]
        atomic_numbers = graph.atomic_numbers
        one_hot_atoms = self.atomic_numbers_to_one_hot(atomic_numbers)
        # check to make sure we have unit cell shifts
        for key in ["cell", "offsets"]:
            if key not in batch:
                raise KeyError(
                    f"Expected periodic property {key} to be in batch."
                    " Please include ``PeriodicPropertiesTransform``."
                )
        # the name of these keys matches up with our `_forward`, and
        # later get remapped to MACE ones
        data.update(
            {
                "pos": graph.pos,
                "edge_index": graph.edge_index,
                "node_feats": one_hot_atoms,
                "ptr": graph.ptr,  # refers to pointers/node segments
                "cell": batch["cell"],
                "shifts": batch["offsets"],
            }
        )
        return data

    def _forward(
        self,
        graph: AbstractGraph,
        node_feats: torch.Tensor,
        pos: torch.Tensor,
        **kwargs,
    ) -> Embeddings:
        """
        Takes arguments in the standardized format, and passes them into MACE
        with some redundant mapping.

        Parameters
        ----------
        graph : AbstractGraph
            Graph structure containing node and graph properties

        node_feats : torch.Tensor
            Tensor containing one-hot node features, shape ``[num_nodes, num_elements]``

        pos : torch.Tensor
            2D tensor containing node positions, shape ``[num_nodes, 3]``

        Returns
        -------
        Embeddings
            MatSciML ``Embeddings`` structure
        """
        # repack data into MACE format
        mace_data = {
            "positions": pos,
            "node_attrs": node_feats,
            "ptr": kwargs["ptr"],
            "cell": kwargs["cell"],
            "shifts": kwargs["shifts"],
            "batch": graph.batch,
            "edge_index": graph.edge_index,
        }
        outputs = self.encoder(
            mace_data,
            training=self.training,
            compute_force=False,
            compute_virials=False,
            compute_stress=False,
            compute_displacement=False,
        )
        # TODO check that these are the correct things to unpack
        node_embeddings = outputs["node_feats"]
        graph_embeddings = outputs["contributions"]
        return Embeddings(graph_embeddings, node_embeddings)
