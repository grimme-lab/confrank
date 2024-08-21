from torch_geometric.data import Data
from .base_transform import BaseTransform


class PipelineTransform(BaseTransform):
    """Combination of multiple transformations applied in consecutive order to a given `Data` object."""

    def __init__(self, tranformations: list[BaseTransform]):
        self.tranformations = tranformations

    def __call__(self, sample: Data) -> Data:
        for trafo in self.tranformations:
            sample = trafo(sample)
        return sample
