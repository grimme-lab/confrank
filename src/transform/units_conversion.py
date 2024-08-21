"""Script to change the units of a given dataset."""

import torch
from torch_geometric.transforms import BaseTransform
from src.util.units import AA2AU, AU2KCAL


class ConvertUnitsToSI(BaseTransform):
    """Change units to SI, namely positions in [Angstrom], energies in [kcal / mol] and gradients in [kcal / mol / Angstrom]."""

    # NOTE: this is a `pyg` Transform object
    # ds = ConfRankDataset(fp, transform=ConvertUnitsToSI())

    energies: list[str] = [  # Eh -> kcal/mol
        "add._restraining",
        "angle_energy",
        "bond_energy",
        "bonded_atm_energy",
        "dispersion_energy",
        "electrostat_energy",
        "etot",
        "external_energy",
        "hb_energy",
        "hbb_e",
        "hbl_e",
        "repulsion_energy",
        "torsion_energy",
        "total_energy",
        "total_energy_ref",
        "xb_e",
        "xb_energy",
    ]

    distances: list[str] = []  # ["pos", "rmsd"]  # Bohr -> Angstrom

    gradients: list[str] = ["grad", "grad_ref"]  # Eh / Bohr -> kcal/mol / Angstrom

    def forward(self, data):
        # NOTE: this function is lazily called when data objects within dataset are accessed.
        for attr, value in data:
            if attr in self.energies:
                data.__setitem__(attr, value * AU2KCAL)
            if attr in self.distances:
                data.__setitem__(attr, value / AA2AU)
            if attr in self.gradients:
                data.__setitem__(attr, value * AU2KCAL * AA2AU)
        return data

    def apply_on_dict(self, data: dict) -> dict:
        """Apply unit conversion on data in a `dict` format."""

        def _set(attr, value, factor):
            """Handle different dimensionalities, i.e. scalars, lists, tensors."""
            value = torch.tensor(value)
            value = value * factor
            data[attr] = value.tolist()

        for attr, value in data.items():
            if attr in self.energies:
                _set(attr, value, AU2KCAL)
            if attr in self.distances:
                _set(attr, value, 1 / AA2AU)
            if attr in self.gradients:
                _set(attr, value, AU2KCAL * AA2AU)
        return data
