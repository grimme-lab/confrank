from typing import Optional
import numpy as np
import torch
from torch import Tensor
import torchmetrics
from torchmetrics.utilities import dim_zero_cat
from scipy.stats import spearmanr, pearsonr, kendalltau, rankdata
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
from matplotlib.colors import LogNorm


class MAE(torchmetrics.MeanAbsoluteError):
    def __init__(self, **kwargs):
        super().__init__()

    def update(self, pred, target):
        super().update(pred, target)


class RMSE(torchmetrics.MeanSquaredError):
    def __init__(self, **kwargs):
        super(RMSE, self).__init__(squared=False)

    def update(self, pred, target):
        super().update(pred, target)


class MAX_AE(torchmetrics.MaxMetric):
    def __init__(self, **kwargs):
        super(MAX_AE, self).__init__()

    def update(self, pred, target):
        abs = (pred - target).abs()
        super().update(abs)


class R2Score(torchmetrics.R2Score):
    def __init__(self, **kwargs):
        super().__init__()

    def update(self, pred, target):
        if len(pred.shape) > 1:
            if pred.shape[1] > 1:
                # in case of forces -> Compute norm first
                pred = torch.norm(pred, p=2, dim=-1)
                target = torch.norm(target, p=2, dim=-1)
        super().update(pred.view(-1), target.view(-1))


class ActualVsPredicted:
    def __init__(self, xy_lim: Optional[float] = None):
        self.preds = []
        self.target = []
        self.xy_lim = xy_lim
        self.epoch = 0

    def update(self, preds: Tensor, target: Tensor) -> None:
        self.preds.append(preds.detach().cpu())
        self.target.append(target.detach().cpu())

    def compute(
        self,
        R2_score,
        RMSE,
        MAE,
        SignFlipPercentage,
        Top1Score,
        Top3Score,
        Top5Score,
        spearman_ensemble_mean,
        pearson_ensemble_mean,
        kendall_tau_ensemble_mean,
    ):
        f = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(4, 4)

        ax_main = plt.subplot(gs[1:4, 0:3])
        ax_x_dist = plt.subplot(gs[0, 0:3], sharex=ax_main)
        ax_y_dist = plt.subplot(gs[1:4, 3], sharey=ax_main)
        ax_metrics = plt.subplot(gs[0, 3])  # Additional subplot for metrics

        preds = torch.cat(self.preds)
        target = torch.cat(self.target)

        xy_lim = (
            self.xy_lim
            if self.xy_lim is not None
            else torch.max(torch.abs(torch.cat([preds, target]))).item() * 1.10
        )

        # Generate a heatmap instead of a scatterplot
        heatmap_data, xedges, yedges = np.histogram2d(
            preds.numpy(),
            target.numpy(),
            bins=150,
            range=[[-xy_lim, xy_lim], [-xy_lim, xy_lim]],
        )
        extent = (xedges[0], xedges[-1], yedges[0], yedges[-1])
        ax_main.imshow(
            heatmap_data.T,
            extent=extent,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            norm=LogNorm(),
        )
        ax_main.plot([0, 1], [0, 1], transform=ax_main.transAxes, color="red")

        # Plotting the density on the separate axes
        sns.kdeplot(x=preds.numpy(), ax=ax_x_dist, color="blue", fill=True)
        sns.kdeplot(y=target.numpy(), ax=ax_y_dist, color="blue", fill=True)

        # Hide the density plot labels
        ax_x_dist.tick_params(axis="x", labelbottom=False)
        ax_y_dist.tick_params(axis="y", labelleft=False)

        # Set the limits on the density axes
        ax_x_dist.set_xlim(-xy_lim, xy_lim)
        ax_y_dist.set_ylim(-xy_lim, xy_lim)

        ax_main.set_xlabel("$E_{ML}^{(1)} - E_{ML}^{(2)}$")
        ax_main.set_ylabel("$E_{DFT}^{(1)} - E_{DFT}^{(2)}$")
        ax_main.set_xlim(-xy_lim, xy_lim)
        ax_main.set_ylim(-xy_lim, xy_lim)
        ax_main.set_title(f"epoch: {self.epoch:03d}")
        # Add metrics to the new subplot
        metrics_text = (
            f"MAE: {MAE:.2E}\n"
            f"RMSE: {RMSE:.2E}\n"
            f"R2 score: {R2_score:.2f}\n"
            f"SignFlipPctg: {SignFlipPercentage:.3f}\n"
            f"Top-1 score: {Top1Score:.3f}\n"
            f"Top-3 score: {Top3Score:.3f}\n"
            f"Top-5 score: {Top5Score:.3f}\n"
            f"spearman_ensbl_mean: {spearman_ensemble_mean:.3f}\n"
            f"pearson_ensbl_mean: {pearson_ensemble_mean:.3f}\n"
            f"kendall_tau_ensbl_mean: {kendall_tau_ensemble_mean:.3f}\n"
        )
        ax_metrics.text(
            0.5,
            0.5,
            metrics_text,
            ha="center",
            va="center",
            fontsize=9,
            transform=ax_metrics.transAxes,
        )
        ax_metrics.axis("off")  # Turn off the axis
        plt.tight_layout()

        self.epoch += 1
        return f

    def reset(self):
        self.preds = []
        self.target = []


class SignFlipPercentage(torchmetrics.Metric):
    is_differentiable: Optional[bool] = False
    higher_is_better: Optional[bool] = False

    def __init__(self, **kwargs):
        super(SignFlipPercentage, self).__init__(compute_on_cpu=True)
        self.add_state("q1_count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("q3_count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        self.q1_count += torch.sum((preds > 0.0) & (target < 0.0)).long()
        self.q3_count += torch.sum((preds < 0.0) & (target > 0.0)).long()
        self.total_count += len(preds)

    def compute(self):
        percentage = (
            ((self.q1_count + self.q3_count) / self.total_count)
            if self.total_count > 0
            else 0
        )
        return percentage


class Stddev_AE(torchmetrics.Metric):
    is_differentiable: Optional[bool] = True
    higher_is_better: Optional[bool] = False

    def __init__(self, **kwargs):
        super().__init__(compute_on_cpu=True)
        self.add_state("diffs", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        self.diffs.append(preds - target)

    def compute(self):
        diffs = dim_zero_cat(self.diffs)
        std = torch.std(torch.abs(diffs))
        return std


class MAD(torchmetrics.Metric):
    is_differentiable: Optional[bool] = True
    higher_is_better: Optional[bool] = False

    def __init__(self, forces=False, **kwargs):
        super().__init__(compute_on_cpu=True)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.forces = forces

    def update(self, preds: Tensor, target: Tensor) -> None:
        self.preds.append(preds)
        self.target.append(target)

    def compute(self):
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        if self.forces:
            dataset_mean = torch.mean(preds, dim=1, keepdim=True)
            mad = torch.mean(torch.abs(target - dataset_mean))
        else:
            dataset_mean = torch.mean(preds)
            mad = torch.mean(torch.abs(target - dataset_mean))
        return mad


class RankingMetrics:
    def __init__(self):
        # ks for top-k errors
        self.ks = [1, 3, 5]
        self.ensemble_values = {"target": {}, "pred": {}}

    @staticmethod
    def get_longest_row(ensemble_values):
        result = {}
        # iterate over all ensembles
        for ensbid in ensemble_values.keys():
            longest_row = []
            # each element in ensemble_values[ensbid] stores a row of the upper triangle matrix of pairwise differences
            # look for the longest row, i.e., the row that stores information about ALL pairwise differences
            for confid in ensemble_values[ensbid].keys():
                if len(ensemble_values[ensbid][confid]) >= len(longest_row):
                    longest_row = ensemble_values[ensbid][confid]
            longest_row.append(
                [
                    confid,
                    0.0,
                ]
            )  # add distance to itself (diagonal element of pairwise distance matrix) for completeness
            result[ensbid] = longest_row
        return result

    def update(self, target, pred):
        ensbids = target["ensbid"]
        confids_1 = target["confid_1"]
        confids_2 = target["confid_2"]
        for key, data in zip(["target", "pred"], [target, pred]):
            diffs = data["energy_difference"]
            # store pairwise differences for conf1 in each ensemble
            # amounts to row-wise saving of the upper triangle matrix of pairwise differences for each ensemble
            # instead of saving the actual matrix (or its rows) we save tuples in a dictionary,
            # so extraction by conformer_id is easier and such that we do not rely on a particluar order of conformer ids
            for diff, ensbid, confid_1, confid_2 in zip(
                diffs, ensbids, confids_1, confids_2
            ):
                # create dict for ensemble if it does not exist already
                if ensbid not in self.ensemble_values[key].keys():
                    self.ensemble_values[key][ensbid] = {}
                # create row of upper triangle pairwise difference matrix if it does not exist
                if confid_2 not in self.ensemble_values[key][ensbid].keys():
                    # print(confid_2)
                    self.ensemble_values[key][ensbid][confid_2] = []
                # save
                self.ensemble_values[key][ensbid][confid_2].append(
                    [confid_1, diff.item()]
                )

    def compute(self):
        pairwise_diffs_ref = self.get_longest_row(self.ensemble_values["target"])
        pairwise_diffs = self.get_longest_row(self.ensemble_values["pred"])
        correct = {
            k: 0 for k in self.ks
        }  # found the actual lowest conformer in the k lowest predictions
        spearman_correlation_coeffs = []
        pearson_correlation_coeffs = []
        kendall_tau_dists = []
        total = 0
        for ensbid in pairwise_diffs_ref.keys():
            # relative energies with respect to confid that corresponds to row no. 0 in the pairwise difference matrix
            target_rel_energies = [p[1] for p in pairwise_diffs_ref[ensbid]]
            predicted_rel_energies = [p[1] for p in pairwise_diffs[ensbid]]
            spearman_coff = spearmanr(target_rel_energies, predicted_rel_energies)[0]
            spearman_correlation_coeffs.append(spearman_coff)
            pearson_coff = pearsonr(target_rel_energies, predicted_rel_energies)[0]
            pearson_correlation_coeffs.append(pearson_coff)
            kendall_tau_val = kendalltau(
                rankdata(target_rel_energies),
                rankdata(predicted_rel_energies),
            )[0]
            kendall_tau_dists.append(kendall_tau_val)

            # sort conformer ids for computing top-k accuracies
            sorted_target_confs = [
                p[0] for p in sorted(pairwise_diffs_ref[ensbid], key=lambda x: x[1])
            ]
            sorted_predicted_confs = [
                p[0] for p in sorted(pairwise_diffs[ensbid], key=lambda x: x[1])
            ]
            for k in self.ks:
                if sorted_target_confs[0] in sorted_predicted_confs[:k]:
                    correct[k] += 1
            total += 1
        results = dict(
            spearman_ensemble_mean=np.mean(spearman_correlation_coeffs).item(),
            spearman_ensemble_stddev=np.std(spearman_correlation_coeffs).item(),
            spearman_ensemble_median=np.median(spearman_correlation_coeffs).item(),
            spearman_ensemble_max=np.max(spearman_correlation_coeffs).item(),
            spearman_ensemble_min=np.min(spearman_correlation_coeffs).item(),
            pearson_ensemble_mean=np.mean(pearson_correlation_coeffs).item(),
            pearson_ensemble_stddev=np.std(pearson_correlation_coeffs).item(),
            pearson_ensemble_median=np.median(pearson_correlation_coeffs).item(),
            pearson_ensemble_max=np.max(pearson_correlation_coeffs).item(),
            pearson_ensemble_min=np.min(pearson_correlation_coeffs).item(),
            kendall_tau_ensemble_mean=np.mean(kendall_tau_dists).item(),
            kendall_tau_ensemble_stddev=np.std(kendall_tau_dists).item(),
            kendall_tau_ensemble_median=np.median(kendall_tau_dists).item(),
            kendall_tau_ensemble_max=np.max(kendall_tau_dists).item(),
            kendall_tau_ensemble_min=np.min(kendall_tau_dists).item(),
        )
        top_k_accuracies = {f"top_{k}": c / total for k, c in correct.items()}
        return {**results, **top_k_accuracies}

    def reset(self):
        self.ensemble_values = {"target": {}, "pred": {}}
