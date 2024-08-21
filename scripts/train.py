"""
Example script for training models in either pointwise or pairwise fashion
"""

import sys
import os

sys.path.append("../")

import torch
import mlflow
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger
from torch_geometric.loader import DataLoader
from argparse import ArgumentParser

from src.training.lightning import LightningWrapper
from src.models.DimeNetPP import DimeNetPP
from src.models.SchNet import SchNet
from src.models.MACE import MACE
from src.data import ConfRankDataset, PairDataset
from src.transform import (
    Scale,
    Rename,
    RadiusGraph,
    PipelineTransform,
)
from src.util.deployment import save_model

# parse command line inputs
parser = ArgumentParser()
parser.add_argument(
    "--data_dir",
    type=str,
    required=True,
    help="Absolute path to directory with ConfRank dataset.",
)
parser.add_argument(
    "--model",
    type=str,
    required=True,
    choices=["dimenet", "mace", "schnet", "gemnet-T"],
)
parser.add_argument("--cutoff", type=float, required=True)
parser.add_argument(
    "--pairwise",
    type=lambda x: x.lower() == "true",
    required=False,
    default=False,
    choices=[True, False],
)

args = parser.parse_args()

mlflow.set_experiment(experiment_name=f"Train models")
mlflow.pytorch.autolog()

# compensate that pairwise training results in more model updates per epoch by making some values dependent on args.pairwise
training_hyperparams = {
    "max_epochs": 100 if args.pairwise else 1000,
    "lr": 1e-3,
    "weight_decay": 1e-8,
    "batch_size": 20 if args.pairwise else 40,
    "stopping_patience": 5 if args.pairwise else 15,
    "decay_patience": 3 if args.pairwise else 10,
    "decay_factor": 0.5,
    "energy_key": "total_energy_ref",
    "forces_key": None,
    "forces_tradeoff": 0.0,
    "lowest_k": 1,  # always sample the lowest_k pairs with the lowest energies
    "additional_k": 19,  # in addition, sample additional_k from the remaining datapoints,
    "precision": 64,
    "trainset_path": [f"{args.data_dir}/confrank_train{i}.h5" for i in range(1, 9)],
    "testset_path": [f"{args.data_dir}/confrank_test.h5"],
    "seed": 42,
    "pairwise": args.pairwise,
}

exclude_keys = [
    "add._restraining",
    "tlist",
    "xb",
    "imet",
    "vtors",
    "hbl_e",
    "vbond",
    "bpair",
    "vangl",
    "nb",
    "hbl",
    "blist",
    "alist",
    "total_charge",
    "xb_e",
    "hbb",
    "uid",
    "hbb_e",
    "total_energy_gfn2",
    "dispersion_energy",
    "bonded_atm_energy",
    "repulsion_energy",
    "hb_energy",
    "electrostat_energy",
    "bond_energy",
    "angle_energy",
    "external_energy",
    "torsion_energy",
    "xb_energy",
]

energy_loss_fn = lambda x, y: torch.nn.functional.l1_loss(x, y)

if training_hyperparams["precision"] == 32:
    dtype = torch.float32
elif training_hyperparams["precision"] == 64:
    dtype = torch.float64
else:
    raise Exception("Precision must be either 32 or 64")

with mlflow.start_run() as run:
    pl.seed_everything(seed=training_hyperparams["seed"], workers=True)

    r2scan_atom_refs = {
        1: -312.0427605689065,
        6: -23687.220998505094,
        7: -34221.8360905642,
        8: -47026.572451837295,
        9: -62579.24268115989,
        14: -181528.62693507367,
        15: -214078.44768832004,
        16: -249752.85985328682,
        17: -288725.9515678963,
        35: -1615266.7419546635,
        53: -186814.76788476118,
    }  # in kcal/mol

    # select model and corresponding hyperparameters:
    mlflow.log_param("model", args.model)
    if args.model == "dimenet":
        model_hyperparams = {
            "hidden_channels": 48,
            "num_blocks": 3,
            "int_emb_size": 32,
            "basis_emb_size": 5,
            "out_emb_channels": 32,
            "num_spherical": 5,
            "num_radial": 6,
            "cutoff": args.cutoff,
        }
        gnn = DimeNetPP(**model_hyperparams).to(dtype)
    elif args.model == "mace":
        model_hyperparams = dict(
            r_max=args.cutoff,
            num_bessel=8,
            num_polynomial_cutoff=6,
            max_ell=2,
            num_interactions=3,
            hidden_irreps="32x0e + 32x1o",
            MLP_irreps="32x0e",
            atomic_energies={key: 0.0 for key, val in r2scan_atom_refs.items()},
            correlation=3,
        )
        gnn = MACE(**model_hyperparams).to(dtype)
    elif args.model == "schnet":
        model_hyperparams = dict(
            cutoff=args.cutoff,
            hidden_channels=128,
            num_filters=64,
            num_interactions=3,
            num_gaussians=50,
        )
        gnn = SchNet(**model_hyperparams)
    elif args.model == "gemnet-T":
        raise NotImplementedError(
            "Currently not supported due to License incompatibility."
        )
    else:
        raise Exception

    transform = PipelineTransform(
        [
            Scale(scaling={"grad_ref": -1.0}),
            Rename(key_mapping={"grad_ref": "forces"}),
            RadiusGraph(cutoff=args.cutoff),
        ]
    )
    if training_hyperparams["pairwise"]:
        trainset, valset, _ = PairDataset(
            path_to_hdf5=training_hyperparams["trainset_path"],
            sample_pairs_randomly=True,
            transform=transform,
            lowest_k=training_hyperparams["lowest_k"],
            additional_k=training_hyperparams["additional_k"],
            dtype=dtype,
        ).split_by_ensemble(0.92, 0.08, 0.0)

    else:
        dsets = [
            ConfRankDataset(path_to_hdf5=path, transform=transform, dtype=dtype)
            for path in training_hyperparams["trainset_path"]
        ]
        _trainset = torch.utils.data.ConcatDataset(dsets)
        trainset, valset = torch.utils.data.random_split(_trainset, [0.92, 0.08])

    gnn.set_constant_energies(
        energy_dict={key: val for key, val in r2scan_atom_refs.items()}, freeze=False
    )

    lightning_module = LightningWrapper(
        model=gnn,
        energy_key=training_hyperparams["energy_key"],
        forces_key=training_hyperparams["forces_key"],
        forces_tradeoff=training_hyperparams["forces_tradeoff"],
        atomic_numbers_key="z",
        decay_factor=training_hyperparams["decay_factor"],
        decay_patience=training_hyperparams["decay_patience"],
        energy_loss_fn=energy_loss_fn,
        weight_decay=training_hyperparams["weight_decay"],
        xy_lim=None,
        pairwise=training_hyperparams["pairwise"],
    )

    testset = PairDataset(
        path_to_hdf5=training_hyperparams["testset_path"],
        sample_pairs_randomly=True,
        transform=transform,
        lowest_k=training_hyperparams["lowest_k"],
        additional_k=training_hyperparams["additional_k"],
        dtype=dtype,
    )

    train_loader = DataLoader(
        trainset,
        batch_size=training_hyperparams["batch_size"],
        shuffle=True,
        drop_last=True,
        exclude_keys=exclude_keys,
    )

    val_loader = DataLoader(
        valset,
        batch_size=training_hyperparams["batch_size"],
        exclude_keys=exclude_keys,
        drop_last=False,
    )

    test_loader = DataLoader(
        testset,
        batch_size=training_hyperparams["batch_size"],
        exclude_keys=exclude_keys,
        drop_last=False,
    )

    monitor_metric = f"ptl/val_loss_{'pairwise' if args.pairwise else 'single'}"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=monitor_metric, save_top_k=3
    )

    early_stop_callback = EarlyStopping(
        monitor=monitor_metric,
        min_delta=0.0,
        patience=training_hyperparams["stopping_patience"],
        verbose=True,
        mode="min",
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks = [checkpoint_callback, early_stop_callback, lr_monitor]

    mlf_logger = MLFlowLogger(run_id=run.info.run_id)

    for key, val in training_hyperparams.items():
        mlflow.log_param(key, val)

    for key, val in model_hyperparams.items():
        mlflow.log_param(key, val)

    mlflow.log_param("len_trainset", len(trainset))
    mlflow.log_param("len_valset", len(valset))
    mlflow.log_param("len_testset", len(testset))
    mlflow.log_param("num_params", sum(p.numel() for p in gnn.parameters()))

    trainer = pl.Trainer(
        max_epochs=training_hyperparams["max_epochs"],
        enable_progress_bar=True,
        callbacks=callbacks,
        logger=mlf_logger,
        log_every_n_steps=200,
        accelerator="gpu" if torch.cuda.is_available() else None,
        devices=1 if torch.cuda.is_available() else None,
        precision=training_hyperparams["precision"],
        inference_mode=True if training_hyperparams["forces_key"] is None else False,
        # allow inference mode but only if no force computation is done. For force computation, inference mode must be False,
    )

    trainer.fit(
        lightning_module, train_dataloaders=train_loader, val_dataloaders=val_loader
    )

    # save best model checkpoint
    best_model_path = checkpoint_callback.best_model_path
    ckpt = torch.load(best_model_path)
    lightning_module.load_state_dict(ckpt["state_dict"])
    best_model = lightning_module.model
    run_id = mlflow.active_run().info.run_id
    experiment_id = mlflow.active_run().info.experiment_id
    default_root_dir = f"mlruns/{experiment_id}/{run_id}"
    model_path = os.path.join(default_root_dir, f"best_model.{args.model}")
    save_model(best_model, model_path)
    mlflow.log_artifact(model_path)

    # always run tests in pairwise mode
    lightning_module.pairwise = True
    trainer.test(lightning_module, ckpt_path="best", dataloaders=test_loader)
