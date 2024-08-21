from abc import ABC, abstractmethod

from torch_geometric.data import Data


class BaseTransform(ABC):
    """Base class for transformations applied to a `ConfRankDataset`.
    NOTE: Transformations are only lazily applied, when sample is accessed.

    Example:
    >> trafo = Dummy_Transform()
    >> ds = ConfRankDataset(..., transform=trafo)
    >> for s in pepconf:
    >>    print(s) # apply transformation
    """

    @abstractmethod
    def __call__(self, sample: Data) -> Data:
        """Apply transformation to `Data` object and return transformed `Data` object."""
        pass


class DummyTransform(BaseTransform):
    """Dummy transformation for testing purposes."""

    def __call__(self, sample: Data) -> Data:
        sample.uid = sample.uid + "_edit"
        return sample
