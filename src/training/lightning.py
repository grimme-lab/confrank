import torch
from torch import Tensor
import pytorch_lightning as pl
from typing import Optional, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from src.training.metrics import (
    MAE,
    MAX_AE,
    Stddev_AE,
    MAD,
    RMSE,
    R2Score,
    SignFlipPercentage,
    ActualVsPredicted,
    RankingMetrics,
)

sns.set_theme()

# stores tuples (bool, metric), where the bool indicates whether metric is used in force computations as well
metric_dict = {
    "MAE": (True, MAE),
    "MAX_AE": (True, MAX_AE),
    "Stddev_AE": (True, Stddev_AE),
    "MAD": (True, MAD),
    "RMSE": (True, RMSE),
    "R2Score": (False, R2Score),
    "SignFlipPctg": (False, SignFlipPercentage),
}


def huber_loss(x, y):
    return torch.nn.functional.huber_loss(x, y, delta=0.01)


class LightningWrapper(pl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        decay_patience: int = 10,
        decay_factor: float = 0.5,
        min_lr: float = 1e-5,
        energy_key: str = "energy",
        forces_key: Optional[str] = None,
        atomic_numbers_key: str = "z",
        energy_tradeoff: float = 1.0,
        forces_tradeoff: Optional[float] = 1.0,
        energy_loss_fn: Callable = huber_loss,
        forces_loss_fn: Callable = huber_loss,
        xy_lim=None,
        pairwise=True,
    ):
        """
        :param model: Model that should be trained
        :param lr: learning rate, default: 1e-3
        :param weight_decay: l2 regularization, default=0.0
        :param decay_patience: patience when using certain schedulers, e.g. cosine decay (optional), default: None
        :param decay_factor: lr decay factor when using certain schedulers, e.g. cosine decay (optional), default: None
        :param min_lr: minimal learning rate when using certain schedulers, e.g. cosine decay (optional), default: None
        :param energy_key: Allows for custom energy key, default: 'energy'
        :param forces_key: Allows for custom force key (optional), default: 'forces'
        :param energy_tradeoff: Tradeoff in loss for energy, default: 1.0
        :param forces_tradeoff: Tradeoff in loss for forces (optional), default: 1.0
        :param energy_loss_fn: Loss function for energy contributions, default: huber loss with delta=0.01
        :param forces_loss_fn: Loss function for force contributions, default: huber loss with delta=0.01
        :param xy_lim: limit for x- and y-axis in the "actual vs. predicted" plot
        :param pairwise: boolean, changes training/val/test to single or pairwise mode.
         Need to change the dataloaders accordingly
        """
        super().__init__()
        self.model = model
        self.lr = lr
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.decay_patience = decay_patience
        self.decay_factor = decay_factor
        self.energy_key = energy_key
        self.forces_key = forces_key
        self.energy_loss_fn = energy_loss_fn
        self.forces_loss_fn = forces_loss_fn
        self.atomic_numbers_key = atomic_numbers_key
        self.xy_lim = xy_lim
        self.pairwise = pairwise
        # define output properties for training
        self.output_properties = dict(single=["energy"], pairwise=["energy_difference"])
        self.property_tradeoffs = {
            "energy": energy_tradeoff,
            "energy_difference": energy_tradeoff,
            "forces": forces_tradeoff,
        }
        # case handling for training with forces
        if forces_key is not None and forces_tradeoff > 0.0:
            compute_forces = True
        else:
            compute_forces = False
        self.compute_forces(mode=compute_forces)
        self.reset_metrics()
        self.model.train()

    def compute_forces(self, mode: bool = True):
        self._compute_forces = mode
        self.model.compute_forces = mode
        if mode:
            for m in ["single", "pairwise"]:
                if "forces" not in self.output_properties[m]:
                    self.output_properties[m].append("forces")
        # need to reset metrics if compute_forces is changed after training
        # otherwise there would be missing or invalid keys in self.metrics during evaluation or testing
        self.reset_metrics()

    def reset_metrics(self):
        # specify metrics
        metrics = {}
        for mode in ["single", "pairwise"]:
            subset_metrics = {}
            for subset in ["train", "val", "test"]:
                prop_metrics = {}
                for prop in self.output_properties[mode]:
                    metric_metrics = {}
                    for metric_name, bool_metric in metric_dict.items():
                        use_for_forces, metric = bool_metric
                        if prop == "forces" and not use_for_forces:
                            continue
                        else:
                            metric_metrics[metric_name] = metric(
                                forces=(prop == "forces")
                            )
                    prop_metrics[prop] = metric_metrics
                subset_metrics[subset] = prop_metrics
            metrics[mode] = subset_metrics

        self.energy_actual_vs_predicted = {
            subset: ActualVsPredicted(xy_lim=self.xy_lim)
            for subset in ["train", "val", "test"]
        }

        self.ranking_metrics = {
            subset: RankingMetrics() for subset in ["train", "val", "test"]
        }

        self.metrics = metrics

    def forward(self, data):
        if self.pairwise:
            data_1, data_2 = data
            out = self.forward_pairwise(data_1, data_2)
        else:
            out = self.forward_single(data)
        return out

    def forward_single(self, data):
        energy, forces = self.model(data)
        return energy, forces

    def forward_pairwise(self, data_1, data_2):
        # rename keys of cartesian coordinates and atomic numbers:
        if "z" not in data_1.keys():
            temp = dict(z=data_1[self.atomic_numbers_key])
            data_1.update(temp)
            temp = dict(z=data_2[self.atomic_numbers_key])
            data_2.update(temp)
        else:
            temp = dict(z=data_1["z"].long())
            data_1.update(temp)
            temp = dict(z=data_2["z"].long())
            data_2.update(temp)

        energy_1, forces_1 = self.model(data_1)
        energy_2, forces_2 = self.model(data_2)
        energy_difference = energy_1 - energy_2
        forces = (
            torch.cat([forces_1, forces_2], dim=0) if self._compute_forces else None
        )  # concatenate force components
        return energy_difference, forces

    def loss_single(self, prediction, target) -> Tensor:
        loss = torch.tensor(
            0.0,
            device=target["energy"].device,
            dtype=target["energy"].dtype,
        )
        for prop, val in prediction.items():
            if prop == "energy":
                contrib = self.energy_loss_fn(prediction[prop], target["energy"])
            elif prop == "forces" and val is not None:
                contrib = self.forces_loss_fn(prediction[prop], target["forces"])
            else:
                continue
            loss += self.property_tradeoffs[prop] * contrib
        return loss

    def loss_pairwise(self, prediction, target) -> Tensor:
        loss = torch.tensor(
            0.0,
            device=target["energy_difference"].device,
            dtype=target["energy_difference"].dtype,
        )
        for prop, val in prediction.items():
            if prop == "energy_difference":
                contrib = self.energy_loss_fn(
                    prediction[prop], target["energy_difference"]
                )
            elif prop == "forces" and val is not None:
                contrib = self.forces_loss_fn(prediction[prop], target["forces"])
            else:
                continue
            loss += self.property_tradeoffs[prop] * contrib
        return loss

    def create_regression_target_pairwise(self, data_1, data_2):
        energy_difference_target = data_1[self.energy_key] - data_2[self.energy_key]
        assert data_1["ensbid"] == data_2["ensbid"]
        target = dict(
            energy_difference=energy_difference_target,
            ensbid=data_1.ensbid,
            confid_1=data_1["confid"],
            confid_2=data_2["confid"],
        )

        if self._compute_forces:
            forces = torch.cat(
                [data_1[self.forces_key], data_2[self.forces_key]], dim=0
            )
            target["forces"] = forces

        return target

    def training_step(
        self,
        data,
        batch_idx: Tensor,
    ) -> Tensor:
        if self.pairwise:
            data_1, data_2 = data
            energy_difference, forces = self.forward_pairwise(data_1, data_2)
            prediction = dict(energy_difference=energy_difference, forces=forces)
            target = self.create_regression_target_pairwise(data_1, data_2)
            loss = self.loss_pairwise(prediction, target)
            self.log(
                f"ptl/train_loss_pairwise",
                loss,
                batch_size=1,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
            return loss
        else:
            energy, forces = self.forward_single(data)
            prediction = dict(energy=energy, forces=forces)
            target = dict(energy=data[self.energy_key])
            if self._compute_forces:
                assert forces is not None
                target["forces"] = data["forces"]
            loss = self.loss_single(prediction, target)
            self.log(
                f"ptl/train_loss_single",
                loss,
                batch_size=1,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
            return loss

    def validation_step(
        self,
        data,
        batch_idx: Tensor,
        dataloader_idx: int = 0,
    ) -> None:
        if self.pairwise:
            assert len(data) == 2
            data_1, data_2 = data
            self._eval_at_step_pairwise(
                data_1, data_2, subset="val", dataloader_idx=dataloader_idx
            )
        else:
            self._eval_at_step_single(data, subset="val", dataloader_idx=dataloader_idx)

    def test_step(
        self,
        data,
        batch_idx: Tensor,
        dataloader_idx: int = 0,
    ):
        if self.pairwise:
            assert len(data) == 2
            data_1, data_2 = data
            self._eval_at_step_pairwise(
                data_1, data_2, subset="test", dataloader_idx=dataloader_idx
            )
        else:
            self._eval_at_step_single(
                data, subset="test", dataloader_idx=dataloader_idx
            )

    def on_validation_epoch_end(self) -> None:
        self._aggregate_metrics("val", self.pairwise)

    def on_test_epoch_end(self) -> None:
        self._aggregate_metrics("test", self.pairwise)

    def _eval_at_step_single(
        self,
        data,
        subset: str,
        dataloader_idx: int,
    ) -> None:
        """
        :param data: mini-batch
        :param subset: either 'train', 'val' or 'test'
        :return: None
        """

        energy, forces = self.forward_single(data)
        prediction = dict(
            energy=energy.detach().cpu(),
            forces=forces.detach().cpu() if forces is not None else None,
        )
        target = dict(energy=data[self.energy_key].cpu())
        if self._compute_forces:
            assert forces is not None
            target["forces"] = data["forces"].cpu()
        loss = self.loss_single(prediction, target)
        self.log(
            f"ptl/{subset}_loss_single",
            loss,
            batch_size=1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        for prop, metrics in self.metrics["single"][subset].items():
            pred_for_prop = prediction[prop].contiguous()
            target_for_prop = target[prop].contiguous()
            for metric_name, metric in metrics.items():
                _ = metric(pred_for_prop, target_for_prop)

    def _eval_at_step_pairwise(
        self,
        data_1,
        data_2,
        subset: str,
        dataloader_idx: int,
    ) -> None:
        """
        :param data: mini-batch
        :param subset: either 'train', 'val' or 'test'
        :return: None
        """

        energy_difference, forces = self.forward_pairwise(data_1, data_2)
        prediction = dict(
            energy_difference=energy_difference.detach().cpu(),
            forces=forces.detach().cpu() if forces is not None else None,
        )
        target = {
            key: (val.cpu() if isinstance(val, torch.Tensor) else val)
            for key, val in self.create_regression_target_pairwise(
                data_1, data_2
            ).items()
        }

        loss = self.loss_pairwise(prediction, target)
        self.log(
            f"ptl/{subset}_loss_pairwise",
            loss,
            batch_size=1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        for prop, metrics in self.metrics["pairwise"][subset].items():
            pred_for_prop = prediction[prop].contiguous()
            target_for_prop = target[prop].contiguous()
            for metric_name, metric in metrics.items():
                _ = metric(pred_for_prop, target_for_prop)

            if prop == "energy_difference":
                self.energy_actual_vs_predicted[subset].update(
                    preds=pred_for_prop, target=target_for_prop
                )
        self.ranking_metrics[subset].update(target=target, pred=prediction)

    def _aggregate_metrics(self, subset: str, pairwise: bool) -> None:
        """
        Aggregate and then reset all metrics
        :param subset: either 'train', 'val' or 'test'
        :return: None
        """
        suffix = self.metrics_suffix(pairwise)
        # store the computed metrics in addition to logging them
        temp_metrics = {}

        subset_metrics = self.metrics[suffix][subset]
        for prop, prop_metrics in subset_metrics.items():
            for metric_name, metric in prop_metrics.items():
                metric_key = "ptl/" + "_".join([prop, subset, metric_name])
                metric_val = metric.compute()
                self.log(
                    "_".join([metric_key, suffix]),
                    metric_val,
                    on_epoch=True,
                    on_step=False,
                    prog_bar="MAE" in metric_key,
                    batch_size=1,
                )
                metric.reset()
                temp_metrics[metric_key] = metric_val

        # additional metrics in case of pairwise mode
        if pairwise:
            assert "energy_difference" in subset_metrics.keys()
            # top k scores:
            ranking_metrics = self.ranking_metrics[subset].compute()
            self.ranking_metrics[subset].reset()
            for key, val in ranking_metrics.items():
                self.log(
                    f"ptl/{key}_{subset}",
                    val,
                    on_epoch=True,
                    on_step=False,
                    prog_bar=subset == "val",
                    batch_size=1,
                )

            # actual vs predicted plot
            fig = self.energy_actual_vs_predicted[subset].compute(
                R2_score=temp_metrics[f"ptl/energy_difference_{subset}_R2Score"],
                RMSE=temp_metrics[f"ptl/energy_difference_{subset}_RMSE"],
                MAE=temp_metrics[f"ptl/energy_difference_{subset}_MAE"],
                SignFlipPercentage=temp_metrics[
                    f"ptl/energy_difference_{subset}_SignFlipPctg"
                ],
                Top1Score=ranking_metrics["top_1"],
                Top3Score=ranking_metrics["top_3"],
                Top5Score=ranking_metrics["top_5"],
                spearman_ensemble_mean=ranking_metrics["spearman_ensemble_mean"],
                pearson_ensemble_mean=ranking_metrics["pearson_ensemble_mean"],
                kendall_tau_ensemble_mean=ranking_metrics["kendall_tau_ensemble_mean"],
            )
            plt.close()
            self.logger.experiment.log_figure(
                self.logger.run_id,
                fig,
                artifact_file=f"epoch={self.current_epoch:03d}_actual_vs_predicted_{subset}.png",
            )
            self.energy_actual_vs_predicted[subset].reset()

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        _scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=opt,
            mode="min",
            min_lr=self.min_lr,
            patience=self.decay_patience,
            factor=self.decay_factor,
        )

        scheduler = {
            "scheduler": _scheduler,
            "monitor": f"ptl/val_loss_{self.metrics_suffix(self.pairwise)}",
            "name": "lr_scheduler",
        }
        schedulers = [scheduler]

        return [opt], schedulers

    def metrics_suffix(self, pairwise: bool) -> str:
        return "pairwise" if pairwise else "single"
