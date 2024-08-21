from torch import Tensor
import torch_geometric
from torch_geometric.data import Data
from src.models.base import BaseModel


class SchNet(BaseModel):

    def __init__(
        self,
        cutoff: float,
        hidden_channels: int,
        num_filters: int,
        num_interactions: int,
        num_gaussians: int,
        gfnff_delta_learning: bool = False,
        compute_forces: bool = False,
        **kwargs
    ):
        model = torch_geometric.nn.SchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
        )

        hyperparameters = dict(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            gfnff_delta_learning=gfnff_delta_learning,
            compute_forces=compute_forces,
        )

        super().__init__(
            hyperparameters=hyperparameters,
            compute_forces=compute_forces,
        )
        self.model = model
        self.cutoff = cutoff

    def model_forward(self, data: Data) -> Tensor:
        energy = self.model(z=data["z"].long(), pos=data["pos"], batch=data["batch"])
        return energy.view(-1)
