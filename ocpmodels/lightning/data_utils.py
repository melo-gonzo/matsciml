# Copyright (C) 2022-3 Intel Corporation
# SPDX-License-Identifier: MIT License

from abc import abstractclassmethod
from typing import Any, Union, Optional, Type, List, Callable, Dict
from pathlib import Path
from warnings import warn
from os import getenv

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torch.utils.data import random_split

from ocpmodels.common.registry import registry
from ocpmodels.datasets import MultiDataset


class MatSciMLDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: Optional[Union[str, Type[TorchDataset], TorchDataset]] = None,
        train_path: Optional[Union[str, Path]] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        val_split: Optional[Union[str, Path, float]] = 0.0,
        test_split: Optional[Union[str, Path, float]] = 0.0,
        seed: Optional[int] = None,
        dset_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        # make sure we have something to work with
        assert any(
            [i for i in [dataset, train_path, val_split, test_split]]
        ), f"No splits provided to datamodule."
        # if floats are passed to splits, make sure dataset is provided for inference
        if any([isinstance(i, float) for i in [val_split, test_split]]):
            assert (
                dataset is not None
            ), f"Float passed to split, but no dataset provided to split."
        if isinstance(dataset, Type):
            assert any(
                [
                    isinstance(p, (str, Path))
                    for p in [train_path, val_split, test_split]
                ]
            ), "Dataset type passed, but no paths to construct with."
        self.dataset = dataset
        self.dset_kwargs = dset_kwargs
        self.save_hyperparameters(ignore=["dataset"])

    def _make_dataset(
        self, path: Union[str, Path], dataset: Union[TorchDataset, Type[TorchDataset]]
    ) -> TorchDataset:
        """
        Convert a string or path specification of a dataset into a concrete dataset object.

        Parameters
        ----------
        spec : Union[str, Path]
            String or path to data split
        dataset : Union[Torch.Dataset, Type[TorchDataset]]
            A dataset object or type. If the former, the transforms from this dataset
            will be copied over to be applied to the new split.

        Returns
        -------
        TorchDataset
            Dataset corresponding to the given path
        """
        dset_kwargs = getattr(self, "dset_kwargs", None)
        if not dset_kwargs:
            dset_kwargs = {}
        # try and grab the dataset class from registry
        if isinstance(dataset, str):
            dset_string = dataset
            dataset = registry.get_dataset_class(dataset)
            if not dataset:
                valid_keys = registry.__entries__["datasets"].keys()
                raise KeyError(
                    f"Incorrect dataset specification from string: passed {dset_string}, but not found in registry: {valid_keys}."
                )
        if isinstance(dataset, TorchDataset):
            transforms = getattr(dataset, "transforms", None)
            dset_kwargs["transforms"] = transforms
            # apply same transforms to this split
            new_dset = dataset.__class__(path, **dset_kwargs)
        else:
            new_dset = dataset(path, **dset_kwargs)
        return new_dset

    def setup(self, stage: Optional[str] = None) -> None:
        splits = {}
        # set up the training split, if provided
        if getattr(self.hparams, "train_path", None) is not None:
            assert isinstance(
                self.dataset, (Type, str)
            ), f"Train path provided but no valid dataset class."
            train_dset = self._make_dataset(self.hparams.train_path, self.dataset)
            # set the main dataset to the train split, since it's used for other splits
            self.dataset = train_dset
            splits["train"] = train_dset
        # now make test and validation splits. If both are floats, we'll do a joint split
        if any(
            [
                isinstance(self.hparams[key], float)
                for key in ["val_split", "test_split"]
            ]
        ):
            # in the case that floats are provided for
            if self.hparams.seed is None:
                # try read from PyTorch Lightning, if not use a set seed
                seed = getenv("PL_GLOBAL_SEED", 42)
            else:
                seed = self.hparams.seed
            generator = torch.Generator().manual_seed(int(seed))
            num_points = len(self.dataset)
            # grab the fractional splits, but ignore them if they are not floats
            val_split = getattr(self.hparams, "val_split")
            if not isinstance(val_split, float):
                val_split = 0.0
            test_split = getattr(self.hparams, "test_split")
            if not isinstance(test_split, float):
                test_split = 0.0
            num_val = int(val_split * num_points)
            num_test = int(test_split * num_points)
            # make sure we're not asking for more data than exists
            num_train = num_points - (num_val + num_test)
            assert (
                num_train >= 0
            ), f"More test/validation samples requested than available samples."
            splits_list = random_split(
                self.dataset, [num_train, num_val, num_test], generator
            )
            for split, key in zip(splits_list, ["train", "val", "test"]):
                if split is not None:
                    splits[key] = split
        # otherwise, just assume paths - if they're not we'll ignore them here
        for key in ["val", "test"]:
            split_path = getattr(self.hparams, f"{key}_split", None)
            if isinstance(split_path, (str, Path)):
                dset = self._make_dataset(split_path, self.dataset)
                splits[key] = dset
        # the last case assumes only the dataset is passed, we will treat it as train
        if len(splits) == 0:
            splits["train"] = self.dataset
        self.splits = splits

    def train_dataloader(self):
        split = self.splits.get("train")
        return DataLoader(
            split,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            collate_fn=self.dataset.collate_fn,
        )

    def predict_dataloader(self):
        """
        Predict behavior just assumes the whole dataset is used for inference.
        """
        return DataLoader(
            self.dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.dataset.collate_fn,
        )

    def test_dataloader(self):
        split = self.splits.get("test", None)
        if split is None:
            return None
        return DataLoader(
            split,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.dataset.collate_fn,
        )

    def val_dataloader(self):
        split = self.splits.get("val", None)
        if split is None:
            return None
        return DataLoader(
            split,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=self.dataset.collate_fn,
        )

    @classmethod
    def from_devset(
        cls,
        dataset: str,
        dset_kwargs: Dict[str, Any] = {},
        **kwargs,
    ):
        r"""
        Instantiate a data module from a dataset's devset.

        This is intended mostly for testing and debugging purposes, with the
        bare number of args/kwargs required to get up and running. The behavior
        of this method will replicate the devset for train, validation, and test
        to allow each part of the pipeline to be tested.

        Parameters
        ----------
        dataset : str
            Class name for dataset to use
        dset_kwargs : Dict[str, Any], optional
            Dictionary of keyword arguments to be passed into
            the dataset creation, for example 'transforms', by default {}

        Returns
        -------
        MatSciMLDataModule
            Instance of `MatSciMLDataModule` from devset

        Raises
        ------
        NotImplementedError
            If the dataset specified does not contain a devset path, this
            method will raise 'NotImplementedError'.
        """
        kwargs.setdefault("batch_size", 8)
        kwargs.setdefault("num_workers", 0)
        dset_kwargs.setdefault("transforms", None)
        dset = registry.get_dataset_class(dataset)
        devset_path = getattr(dset, "__devset__", None)
        if not devset_path:
            raise NotImplementedError(
                f"Dataset {dset.__name__} does not contain a '__devset__' attribute, cannot instantiate from devset."
            )
        datamodule = cls(
            dset,
            train_path=devset_path,
            val_split=devset_path,
            test_split=devset_path,
            dset_kwargs=dset_kwargs,
            **kwargs,
        )
        return datamodule

    @property
    def target_keys(self) -> Dict[str, List[str]]:
        return self.dataset.target_keys


class MultiDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        train_dataset: Optional[MultiDataset] = None,
        val_dataset: Optional[MultiDataset] = None,
        test_dataset: Optional[MultiDataset] = None,
        predict_dataset: Optional[MultiDataset] = None,
    ) -> None:
        super().__init__()
        if not any([train_dataset, val_dataset, test_dataset, predict_dataset]):
            raise ValueError(
                f"No datasets were passed for training, validation, testing, or predict."
            )
        self.save_hyperparameters(
            ignore=["train_dataset", "val_dataset", "test_dataset", "predict_dataset"]
        )
        # stash the datasets as an attribute
        self.datasets = {
            key: value
            for key, value in zip(
                ["train", "val", "test", "predict"],
                [train_dataset, val_dataset, test_dataset, predict_dataset],
            )
        }

    @property
    def target_keys(self) -> Dict[str, Dict[str, List[str]]]:
        return self.datasets["train"].target_keys

    def train_dataloader(self) -> Union[DataLoader, None]:
        data = self.datasets.get("train", None)
        if data is None:
            return None
        return DataLoader(
            data,
            self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            collate_fn=data.collate_fn,
        )

    def val_dataloader(self) -> Union[DataLoader, None]:
        data = self.datasets.get("val", None)
        if data is None:
            return None
        return DataLoader(
            data,
            self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=data.collate_fn,
        )

    def test_dataloader(self) -> Union[DataLoader, None]:
        data = self.datasets.get("test", None)
        if data is None:
            return None
        return DataLoader(
            data,
            self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=data.collate_fn,
        )

    def predict_dataloader(self) -> Union[DataLoader, None]:
        data = self.datasets.get("predict", None)
        if data is None:
            return None
        return DataLoader(
            data,
            self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=data.collate_fn,
        )
