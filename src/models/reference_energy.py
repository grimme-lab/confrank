import torch
from torch import Tensor
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from torch_scatter import scatter_add
from src.data import ConfRankDataset


class ReferenceEnergies(torch.nn.Module):
    """
    PyTorch module for fitting the reference energies (per species)
    """

    def __init__(self, freeze: bool = True, num_embeddings: int = 95) -> None:
        """
        :param freeze: Freeze module weights after init, default: True -> No optimization during training
        :param num_embeddings: Number of expected elements, default: default
        """
        super(ReferenceEnergies, self).__init__()
        # constant energy shifts
        self.constant_shifts = torch.nn.Parameter(
            torch.zeros(num_embeddings, 1), requires_grad=bool(~freeze)
        )

    def loss_fn(self, prediction: Tensor, target: Tensor) -> Tensor:
        """
        Loss function that is minimized during the fit: MSE loss
        :param prediction: Tensor storing predicted energies
        :param target: Tensor with actual energies
        :return: scalar loss Tensor
        """
        out = torch.mean(torch.square(prediction - target))
        return out

    def fit_constant_energies(
        self,
        trainset: ConfRankDataset,
        target_key: str = "energy",
        epochs: int = 10,
        batch_size: int = 50,
        lr: float = 1e-3,
        freeze_after: bool = True,
    ) -> None:
        """
        Fit the energies against number of atom species to obtain coefficients for the constant model using SGD.
        For small dataset it is recommended to perform a fit with sklearn etc.

        :param trainset: Dataset
        :param target_key: Key for target of regression; default: energy
        :param epochs: Number of epochs for optimization, default: 10
        :param batch_size: Size of mini batches, default: 50
        :param lr: Learning rate, default: 1e-3
        :param freeze_after: Whether the weights/reference energies should be frozen after this fit
        :return: None
        """
        initial_device = self.constant_shifts.device
        device = torch.device("cpu")
        train_loader = DataLoader(
            trainset, shuffle=True, drop_last=True, batch_size=batch_size
        )
        # activate gradient computation for constant shifts
        self.constant_shifts.requires_grad_(True)
        self.to(device)
        optim = torch.optim.Adam(lr=lr, params=[self.constant_shifts])
        tqdm_loader_outer = tqdm(range(epochs), desc="Epoch")
        for epoch in tqdm_loader_outer:
            tqdm_loader_outer.set_description(
                f"Epoch {epoch + 1}/{len(tqdm_loader_outer)}"
            )
            summed_loss = 0
            summed_rmse = 0
            tqdm_loader_inner = tqdm(train_loader, leave=False, position=1)
            for step, data in enumerate(tqdm_loader_inner):
                tqdm_loader_inner.set_description(
                    f"Step {step + 1}/{len(tqdm_loader_inner)}"
                )
                batch = data["batch"].to(device)
                energy = data[target_key].to(device)
                species = data["z"].long().to(device)

                energy_shifts = self.constant_shifts[..., 0][species]
                prediction = scatter_add(
                    energy_shifts,
                    index=batch,
                    dim=0,
                    dim_size=len(torch.unique(batch)),
                )
                loss = self.loss_fn(energy, prediction)
                rmse = torch.sqrt(torch.mean(torch.square(prediction - energy)))
                summed_rmse += rmse.item()
                optim.zero_grad()
                loss.backward()
                optim.step()
                summed_loss += loss.item()
                tqdm_loader_inner.set_postfix_str(
                    f"step loss: {loss.item():.4E}, step rmse: {rmse.item():.4E}"
                )
            tqdm_loader_outer.set_postfix_str(
                f"Avg. loss: {summed_loss / len(tqdm_loader_inner):.4E}, "
                f"Avg. rmse: {summed_rmse / len(tqdm_loader_inner):.4E}"
            )
        self.constant_shifts.requires_grad_(freeze_after)
        self.constant_shifts.to(initial_device)
        print("Finished fitting the per-atom-energies.")

    def get_constant_energies(
        self,
    ) -> dict[int, float]:
        output_dict = {}
        for z, el in enumerate(self.constant_shifts):
            if el.abs() > 1e-10:
                output_dict[z] = el.item()
        return output_dict

    def set_constant_energies(
        self, energy_dict: dict[int, float], freeze: bool = True
    ) -> None:
        """
        :param energy_dict: Mapping of {atomic_number : reference_energy}
        :param freeze: Should the model weights be frozen after setting the energies?, default: True
        :return: None
        """
        self.constant_shifts.requires_grad_(False)
        for z in energy_dict.keys():
            self.constant_shifts[z] = energy_dict[z]
        self.constant_shifts.requires_grad_(freeze)

    def forward(
        self,
        species: Tensor,
        batch: Tensor,
    ) -> Tensor:
        """
        :param species: 1D Tensor with atomic numbers of shape (N,)
        :param batch: 1D Tensor with batch indices of shape (N,)
        :return: Contribution from summing up reference energies
        """

        energy_shift_per_atom = self.constant_shifts[..., 0][species.long()]
        constant_contribution = scatter_add(
            energy_shift_per_atom, index=batch, dim=0, dim_size=len(torch.unique(batch))
        )
        return constant_contribution
