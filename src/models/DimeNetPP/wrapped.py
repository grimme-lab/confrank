import torch
from torch import Tensor
import torch_geometric
from torch_geometric.data import Data
from src.models.base import BaseModel


class DimeNetPP(BaseModel):

    def __init__(
        self,
        hidden_channels: int,
        num_blocks: int,
        int_emb_size: int,
        basis_emb_size: int,
        out_emb_channels: int,
        num_spherical: int,
        num_radial: int,
        cutoff: float,
        **kwargs
    ):
        hyperparameters = dict(
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            int_emb_size=int_emb_size,
            basis_emb_size=basis_emb_size,
            out_emb_channels=out_emb_channels,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
        )

        super().__init__(
            hyperparameters=hyperparameters,
        )

        self.dimenetpp = torch_geometric.nn.DimeNetPlusPlus(
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            int_emb_size=int_emb_size,
            basis_emb_size=basis_emb_size,
            out_emb_channels=out_emb_channels,
            num_spherical=num_spherical,
            num_radial=num_radial,
            cutoff=cutoff,
            out_channels=hidden_channels,
        )

        self.linear = torch.nn.Linear(hidden_channels, 1, bias=False)
        self.cutoff = cutoff

    def model_forward(self, data: Data) -> Tensor:
        latent = self.dimenetpp(
            z=data["z"].long(),
            pos=data["pos"],
            batch=data["batch"],
        )
        projection = self.linear(latent)
        return projection
