from abc import ABC, abstractmethod

from torch_geometric.data import InMemoryDataset


class BaseProcessor(ABC):
    """Base class for processing applied to a `ConfRankDataset`.
    NOTE: Processor are applied on a dataset object and return a dataset object

    Example:
    >> proc = Dummy_Processor()
    >> ds = ConfRankDataset(...)
    >> ds_new = proc(ds)
    """

    @abstractmethod
    def __call__(self, dataset: InMemoryDataset) -> InMemoryDataset:
        """Apply transformation to `InMemoryDataset` object and return transformed `InMemoryDataset` object."""
        pass


class DummyProcessor(BaseProcessor):
    """Dummy Processor for testing purposes."""

    def __call__(self, dataset: InMemoryDataset) -> InMemoryDataset:
        return dataset[:2]
