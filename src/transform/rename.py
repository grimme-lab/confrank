from typing import Dict
from torch_geometric.data import Data
from warnings import warn
from .base_transform import BaseTransform


class Rename(BaseTransform):
    """
    Data transformation renaming certain keys
    """

    def __init__(self, key_mapping: Dict[str, str]):
        """
        :param key_mapping: dictionary defining the mapping of the keys.
        If <key> is not found in the sample, there will be no error and <key> is skipped.
        """
        super(Rename, self).__init__()
        self.key_mapping = key_mapping

    def __call__(self, sample: Data) -> Data:
        update_dict = {}
        for key, new_key in self.key_mapping.items():
            if key in sample.keys():
                update_dict[new_key] = sample[key]
            else:
                warn(
                    f"Tried to rename '{key}' to {new_key} but '{key}' was not found in the sample. Not doing anything."
                )
        sample.update(update_dict)
        return sample
