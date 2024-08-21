from __future__ import annotations

from typing import Dict

try:
    import mace
    import e3nn.util.jit

except ImportError:
    raise ImportError("mace is not installed. Please install mace-torch!")

from src.models.base import BaseModel
import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
import torch.nn.functional
from e3nn import o3

from mace import modules, tools


class MACE(BaseModel):
    """
    MACE Wrapper that can be used to train custom mace models
    """

    def __init__(
        self,
        r_max: float,
        num_bessel: int,
        num_polynomial_cutoff: int,
        max_ell: int,
        num_interactions: int,
        correlation: int,
        hidden_irreps: str,
        MLP_irreps: str,
        atomic_energies: Dict[str, int],
        gfnff_delta_learning: bool = False,
        compute_forces: bool = False,
    ):
        atomic_numbers = list(atomic_energies.keys())
        table = tools.AtomicNumberTable(atomic_numbers)

        model_config = dict(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            max_ell=max_ell,
            interaction_cls=modules.interaction_classes[
                "RealAgnosticResidualInteractionBlock"
            ],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticResidualInteractionBlock"
            ],
            num_interactions=num_interactions,
            num_elements=len(table),
            hidden_irreps=o3.Irreps(hidden_irreps),
            MLP_irreps=o3.Irreps(MLP_irreps),
            gate=torch.nn.functional.silu,
            avg_num_neighbors=8,
            atomic_numbers=table.zs,
            atomic_energies=np.array(list(atomic_energies.values())),
            correlation=correlation,
            radial_type="bessel",
        )

        model = modules.MACE(**model_config)

        hyperparameters = dict(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            max_ell=max_ell,
            num_interactions=num_interactions,
            correlation=correlation,
            hidden_irreps=hidden_irreps,
            MLP_irreps=MLP_irreps,
            atomic_energies=atomic_energies,
            gfnff_delta_learning=gfnff_delta_learning,
            compute_forces=compute_forces,
        )

        assert isinstance(
            model, modules.MACE
        ), f"Model is not an instance of MACE. Make sure to load a proper MACE model."
        super().__init__(
            hyperparameters=hyperparameters,
            compute_forces=compute_forces,
        )
        onehot_weights = torch.zeros(
            max(model.atomic_numbers) + 1, len(model.atomic_numbers)
        )
        for i, z in enumerate(model.atomic_numbers):
            onehot_weights[z][i] = 1
        self.onehot = torch.nn.Embedding.from_pretrained(onehot_weights)
        self.model = e3nn.util.jit.script(model)
        self.cutoff = model.r_max.item()

    def model_forward(self, data: Data) -> Tensor:
        one_hot = self.onehot(data["z"].long())
        _, counts = torch.unique(data["batch"], return_counts=True)
        _ptr = [torch.tensor([0.0], dtype=counts.dtype, device=counts.device)]
        for c in counts:
            _ptr.append(_ptr[-1] + c)
        ptr = torch.cat(_ptr)

        data = dict(
            positions=data["pos"],
            edge_index=data["edge_index"],
            shifts=torch.zeros(
                data["edge_index"].shape[1],
                3,
                device=data["pos"].device,
                dtype=data["pos"].dtype,
            ),
            cell=torch.zeros(len(counts), 3, 3),  # only for reasons of compatibility
            node_attrs=one_hot,
            batch=data["batch"],
            ptr=ptr,
        )

        out = self.model.forward(
            data,
            training=True,
            compute_force=False,
            # disable forces computation here as it is carried out by the wrapper
        )
        return out["energy"].view(-1)
