import h5py
import json
import random
from copy import copy
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Callable, Optional
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
from src.transform import BaseTransform


class ConfRankDataset(InMemoryDataset):
    """Default dataset for calculations with GFNFF data."""

    def __init__(
        self,
        path_to_hdf5: str | Path | None = None,
        transform: Callable | None = None,
        dtype=torch.float32,
    ):
        super().__init__("./", transform, pre_transform=None, pre_filter=None)

        self.path = (
            Path(path_to_hdf5) if isinstance(path_to_hdf5, str) else path_to_hdf5
        )

        self.dtype = dtype
        # empty dataset
        self.data = Data()
        self.slices = defaultdict(dict, {})

        if path_to_hdf5:
            self.data, self.slices = ConfRankDataset.from_hdf5(self.path)

            # setting everything to the same precision
            for key, val in self.data.items():
                if isinstance(val, torch.Tensor):
                    if val.dtype == torch.float32 or val.dtype == torch.float64:
                        self.data[key] = val.to(self.dtype)

    def to_hdf5(self, fp: Path):
        """Save the data and slices of the dataset to an HDF5 file."""
        with h5py.File(fp, "w") as f:
            data_group = f.create_group("data")
            slices_group = f.create_group("slices")

            for key, value in self._data.items():
                # save data (incl. type casting)
                # NOTE: str need to be stored as byte
                if isinstance(value, list):
                    if isinstance(value[0], str):
                        value = [v.encode("utf-8") for v in value]
                    value = np.array(value)
                if isinstance(value, torch.Tensor):
                    value = value.numpy()
                if value.dtype == np.int64:
                    value = value.astype(np.uint64)
                data_group.create_dataset(key, data=value)

                # save slices
                slice_value = self.slices[key].numpy()
                if slice_value.dtype == np.int64:
                    slice_value = slice_value.astype(np.uint64)
                slices_group.create_dataset(key, data=slice_value)

    @staticmethod
    def from_hdf5(fp: Path) -> tuple[Data, defaultdict]:
        """Load data and slices from HDF5 file."""
        data = {}
        slices = {}
        with h5py.File(fp, "r") as f:
            for key in f["data"].keys():
                np_arrays = {"data": f["data"][key][:], "slices": f["slices"][key][:]}
                # some casting
                for prop, val in np_arrays.items():
                    if val.dtype == np.uint64:
                        np_arrays[prop] = val.astype(np.int64)
                # uids are of dtype string, so we got to handle it seperately
                if key in ["uid", "conf_id", "confid", "ensbid"]:
                    data[key] = np_arrays["data"].tolist()
                    data[key] = [bs.decode("utf-8") for bs in data[key]]
                    slices[key] = torch.from_numpy(np_arrays["slices"])
                else:
                    data[key] = torch.from_numpy(np_arrays["data"])
                    slices[key] = torch.from_numpy(np_arrays["slices"])
        return Data.from_dict(data), defaultdict(dict, slices)

    @staticmethod
    def from_data_slices(data: Data, slices: defaultdict) -> "ConfRankDataset":
        """
        Create a new GFNFF_Dataset instance using the provided data and slices.

        This static method allows for the creation of a GFNFF_Dataset instance from individual
        data slices rather than loading it from a file as is done in the constructor.

        Args:
            data (Data): The data to be used for the new GFNFF_Dataset instance.
            slices (defaultdict): The slices to be used for the new GFNFF_Dataset instance.

        Returns:
            ConfRankDataset: A new GFNFF_Dataset instance initialized with the provided data and slices.
        """

        gd = ConfRankDataset()
        gd.data, gd.slices = data, slices
        return gd

    def merge(self, other: "ConfRankDataset") -> "ConfRankDataset":
        """Combine two datasets. As done in `https://github.com/pyg-team/pytorch_geometric/issues/88`."""
        # NOTE: alternatively use torch.utils.data.ConcatDataset
        if not isinstance(other, ConfRankDataset):
            raise TypeError("Can only merge instances of `ConfRankDataset`.")

        # merge `Data` objects then collate
        print("Merging datasets...")
        self_data = list(tqdm(self))
        other_data = list(tqdm(other))
        data_list = self_data + other_data

        print("Collating datasets...")
        data, slices = ConfRankDataset.collate(data_list)
        return ConfRankDataset.from_data_slices(data, slices)

    def equal(self, other):
        """Compare two instances for equality."""
        if not isinstance(other, ConfRankDataset):
            return False
        if len(self) != len(other):
            return False

        # compare data content
        for data1, data2 in zip(self, other):
            if data1.num_nodes != data2.num_nodes or data1.num_edges != data2.num_edges:
                return False

            for key in data1.keys():
                # val1 = getattr(data1, key, None)
                if key not in data2:
                    return False
                if isinstance(data1[key], torch.Tensor):
                    if not torch.equal(data1[key], data2[key]):
                        return False
                else:
                    if not data1[key] == data2[key]:
                        return False

        return True

    def get_ensembles(self) -> defaultdict[str : list[Data]]:
        """Obtain ensemble-wise sorted data."""
        ensembles = defaultdict(list)
        for sample in tqdm(self, desc="Get ensembles"):
            ensembles[sample.ensbid].append(sample)
        return ensembles


class PairDataset(Dataset):
    """Collect pairs of samples from dataset. Per default only take pairs from same ensemble."""

    def __init__(
        self,
        path_to_hdf5: list[str | Path],
        sample_pairs_randomly: bool = False,
        lowest_k: Optional[int] = None,
        additional_k: Optional[int] = None,
        transform: Callable | BaseTransform | None = None,
        dtype=torch.float64,
    ):
        """
        :param path_to_hdf5: path of hdf5 file storing the data
        :param sample_pairs_randomly: If False, all pairs (i,j) with i<j are computed for each ensemble.
        If True, random sampling is used and lowest_k and additional_k have to be specified.
        :param lowest_k: Number of conformers with the lowest energy in an ensemble,
        only has an effect if sample_pairs_randomly is True; default: None
        :param additional_k: Number of conformers that are samples randomly,
        only has an effect if sample_pairs_randomly is True; default: None
        :param transform: Transformation for on-the-fly post-processing of data points; default: None
        :param dtype: precision that used; default: torch.float64
        """
        self.dtype = dtype
        if isinstance(path_to_hdf5, str) or isinstance(path_to_hdf5, Path):
            path_to_hdf5 = [path_to_hdf5]
        dsets = []
        for p in path_to_hdf5:
            next_dataset = ConfRankDataset(p, transform, dtype=dtype)
            dsets.append(next_dataset)

        self.dataset = torch.utils.data.ConcatDataset(dsets)
        self.sample_pairs_randomly = sample_pairs_randomly
        self.lowest_k = lowest_k
        self.additional_k = additional_k
        if self.sample_pairs_randomly:
            assert self.lowest_k is not None
            assert self.additional_k is not None

        self.ensembles = self.get_ensembles(self.dataset)
        self.pairs = self._setup_pairs()

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            index1, index2 = self.pairs[idx]
            return self.dataset[index1], self.dataset[index2]
        elif isinstance(idx, slice):
            new_pairs = self.pairs[idx]
            new_dataset = copy(self)
            new_dataset.pairs = new_pairs
            return new_dataset
        elif isinstance(idx, (list, np.ndarray, torch.Tensor)):
            new_pairs = [self.pairs[i] for i in idx]
            new_dataset = copy(self)
            new_dataset.pairs = new_pairs
            return new_dataset
        else:
            raise IndexError("Invalid index type")

    def __str__(self):
        return f"PairDataset({len(self)})"

    def split_by_ensemble(self, train_size, val_size, test_size, seed=42):
        assert train_size + val_size + test_size == 1, "train+val+test sizes must be 1"
        random.seed(seed)
        n_train = int(train_size * len(self.ensembles))
        n_val = int(val_size * len(self.ensembles))
        ensemble_uids = list(self.ensembles.keys())
        random.shuffle(ensemble_uids)
        ensemble_uids_train = ensemble_uids[:n_train]
        ensemble_uids_val = ensemble_uids[n_train : n_train + n_val]
        train_idx = []
        val_idx = []
        test_idx = []
        for idx, sample in enumerate(self):
            ensbid = sample[0].ensbid
            assert (
                sample[1].ensbid == ensbid
            ), f"sample[0] has ensemble id {ensbid} and sample[1] has ensemble id {sample[1].ensbid}"
            if ensbid in ensemble_uids_train:
                train_idx.append(idx)
            elif ensbid in ensemble_uids_val:
                val_idx.append(idx)
            else:
                test_idx.append(idx)
        return self[train_idx], self[val_idx], self[test_idx]

    def get_ensembles(self, dataset) -> dict[str, int] | dict[str, list[int]]:
        """Obtain mapping of samples and ensembles."""
        print("Calculating ensembles ...")
        ensembles = {}
        for idx, d in enumerate(dataset):
            euid = d.ensbid
            ensembles.setdefault(euid, [])
            ensembles[euid].append(idx)
        return ensembles

    def save_ensembles(self, path: str | Path):
        """Save ensemble info to file for further usage."""
        with open(path, "w") as json_file:
            json.dump(self.ensembles, json_file)

    def load_ensembles(self, path: str | Path) -> dict[str, int]:
        """Load ensemble info from file."""
        with open(path, "r") as json_file:
            ensembles = json.load(json_file)
        return ensembles

    def _setup_pairs(self) -> list[tuple[int]]:
        """Initialize pairs to draw samples from."""
        if self.sample_pairs_randomly:
            pairs = self.pair_generation_ensemble_random_sampled()
        else:
            pairs = self.pair_generation_ensemble()
        return pairs

    def pair_generation_ensemble(self):
        """Add pairs all pairs up to permutation"""
        pairs = []
        # get tuples (i,j) with i, j stemming from same ensemble and i < j
        for ensbid, idcs in self.ensembles.items():
            combs = combinations(idcs, 2)
            pairs.extend(combs)  # no permutations
        return pairs

    def pair_generation_ensemble_random_sampled(self):
        """Generate randomly sampled pairs up to permutation"""
        pairs = []
        # get tuples (i,j) with i, j stemming from same ensemble
        for ensbid, idcs in self.ensembles.items():
            # get energies
            energies = np.array(
                [self.dataset[i]["total_energy_ref"].item() for i in idcs]
            )
            # sort by energy
            sort_idx = np.argsort(energies).reshape(-1)
            idxs_i = np.array(idcs)[sort_idx]
            idxs_i = np.concatenate(
                [
                    idxs_i[: self.lowest_k],
                    np.random.choice(
                        idxs_i[self.lowest_k :],
                        replace=False,
                        size=(min(self.additional_k, len(idxs_i[self.lowest_k :])),),
                    ),
                ]
            )
            nn_combs = []
            for p1 in idxs_i:
                for p2 in idxs_i:
                    if p2 < p1:
                        nn_combs.append((p1, p2))
            pairs.extend(nn_combs)
        return pairs
