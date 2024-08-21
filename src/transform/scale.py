from typing import Dict
from warnings import warn
from torch_geometric.data import Data
from .base_transform import BaseTransform


class Scale(BaseTransform):
    """
    Data transformation scaling certain attributes in the data, e.g. multiplying the gradients by (-1).
    """

    def __init__(self, scaling: Dict[str, float]):
        """
        :param scaling: dictionary defining the scaling for certain attributes.
        If a key is not found in the sample, the corresponding key will be skipped.
        """
        super(Scale, self).__init__()
        self.scaling = scaling

    def __call__(self, sample: Data) -> Data:
        update_dict = {}
        for key, factor in self.scaling.items():
            if key in sample.keys():
                update_dict[key] = factor * sample[key]
            else:
                warn(
                    f"Tried to scale '{key}' but '{key}' was not found in the sample. Not doing anything."
                )
        sample.update(update_dict)
        return sample
