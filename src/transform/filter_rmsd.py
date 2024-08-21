from tqdm import tqdm
from .base_processor import BaseProcessor


class FilterRmsd(BaseProcessor):
    """Filter a dataset based on values of the RMSD between
    sample geometries on GFN-FF and reference level.
    """

    def __init__(self, threshold: float):
        self.thr = threshold  # max allowed value for RMSD

    def __call__(self, dataset):
        idxs = []
        for i, sample in tqdm(enumerate(dataset), desc="Filtering RMSD"):
            if sample.rmsd < self.thr:
                idxs.append(i)
        return dataset.index_select(idxs)
