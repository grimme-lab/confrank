from torch_geometric.nn import radius_graph
from torch_geometric.data import Data
from .base_transform import BaseTransform


class RadiusGraph(BaseTransform):
    """
    Transformation for setting up the graph topology (specified by 'edge_index' tensor) for a given cutoff radius
    """

    def __init__(self, cutoff: float):
        """
        :param cutoff: Cutoff radius that is used to setup the graph topology
        """
        super(RadiusGraph, self).__init__()
        self.cutoff = cutoff

    def __call__(self, sample: Data) -> Data:
        batch = sample["batch"] if "batch" in sample.keys() else None
        edge_index = radius_graph(
            x=sample["pos"],
            r=self.cutoff,
            batch=batch,
            max_num_neighbors=320,
        ).long()
        sample = sample.update({"edge_index": edge_index})
        return sample
