"""Training entry point and experiment setup utilities."""

import os
import time
from typing import Any, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from data_handler.CreateDataset import (
    create_train_dataset,
    move_tensordataset_to_device,
)
from tools.filename_handler import format_scientific_notation
from tools.logger import load_logger
from tools.model_config import model_config
from tools.seed import setup_seed
from train_solver import TrainSolver


RESULTS_ROOT = r"D:\train_result"
ANGLE_LOWER_BOUND = -60
ANGLE_UPPER_BOUND = 60

TEMPERATURE_MODELS = (
    "CDSP-Net",
    "CDSP-Net-SD",
    "CDSP-Net-MCV",
    "CDSP-Net-VSCM",
    "CDSP-Net-Top-M",
)
GRID_MODELS = (
    "CDSP-Net-HS",
    "DeepCNN",
)
SUBSPACE_MODELS = ("SubspaceNet",)


def _format_sequence(values: Sequence[Any]) -> str:
    """Format a sequence for use in a file or directory name."""
    return str(list(values)).replace(" ", "")


def _format_snr_range(snr_values: Sequence[float]) -> str:
    """Format an SNR sequence as start-step-end."""
    if not snr_values:
        raise ValueError("snr must contain at least one value.")

    if len(snr_values) == 1:
        return str(snr_values[0])

    step = snr_values[1] - snr_values[0]
    return f"{snr_values[0]}-{step}-{snr_values[-1]}"


def _create_array_positions(config: Any) -> torch.Tensor:
    """Create sensor positions for the configured array geometry."""
    if config.array_type != "ULA":
        raise ValueError(f"Unsupported array type: {config.array_type}")

    return torch.linspace(
        start=0,
        end=(config.N - 1) * config.d_init,
        steps=config.N,
        dtype=torch.float32,
        device=config.device,
    )


def _build_model_directory_name(
    config: Any,
    temp_string: str,
) -> str:
    """Build the experiment directory name."""
    snr_string = _format_snr_range(config.snr)
    common_prefix = (
        f"{config.model_type}_{config.array_type}_"
        f"M{config.M}_N{config.N}_T{_format_sequence(config.T)}_"
        f"Gap{config.Gap}_"
    )
    common_suffix = (
        f"_CoupLevel{_format_sequence(config.coup_level)}"
        f"_PosErrRatio{_format_sequence(config.eta_Ratio)}"
        f"_{config.band_type}_{config.coh_type}_"
    )

    if config.model_type in TEMPERATURE_MODELS:
        return (
            f"{common_prefix}"
            f"grid{config.grid_step}_temp{temp_string}_"
            f"snr{snr_string}"
            f"{common_suffix}"
        )

    if config.model_type in GRID_MODELS:
        return f"{common_prefix}" f"grid{config.grid_step}_" f"snr{snr_string}" f"{common_suffix}"

    if config.model_type in SUBSPACE_MODELS:
        return f"{common_prefix}" f"snr{snr_string}" f"{common_suffix}"

    raise ValueError(f"Unsupported model type: {config.model_type}")


def _build_dataset_filenames(
    config: Any,
) -> Tuple[str, str]:
    """Build training and validation dataset filenames."""
    c_string = format_scientific_notation(config.c, 0)
    fc_string = format_scientific_notation(config.fc, 0)
    snr_string = _format_snr_range(config.snr)

    train_size = int(config.num_sample * config.datasets_ratio[0])
    valid_size = config.num_sample - train_size

    common_name = (
        f"{config.array_type}_M{config.M}_N{config.N}_"
        f"Gap{config.Gap}_T{_format_sequence(config.T)}_"
        f"_tau{config.tau}_snr{snr_string}_"
        f"C{c_string}_fc{fc_string}_"
        f"CoupLevel{_format_sequence(config.coup_level)}_"
        f"PosErrRatio{_format_sequence(config.eta_Ratio)}_"
        f"{config.band_type}_{config.coh_type}"
    )

    train_filename = f"train_dataset_{common_name}_samples{train_size}.pt"
    valid_filename = f"valid_dataset_{common_name}_samples{valid_size}.pt"
    return train_filename, valid_filename


def _log_configuration(
    config: Any,
    logger: Any,
) -> None:
    """Write selected configuration values to the experiment log."""
    print_order = [
        "num_epochs",
        "batch_size",
        "warm_up_epochs",
        "learning_rate",
        "step_size",
        "gamma",
        "weight_decay",
        "checkpoint_path",
        "datasets_save_dir",
        "resume",
        "parallel",
        "random_seed",
        "gpu",
        "model_name",
        "M",
        "N",
        "tau",
        "c",
        "fc",
        "T",
        "snr",
        "lamda",
        "d_init",
        "delay_max",
        "grid_step",
        "temp",
        "Gap",
        "coup_level",
        "eta_Ratio",
        "eta",
        "num_sample",
        "datasets_ratio",
        "is_creat",
        "array_type",
        "band_type",
        "coh_type",
        "model_type",
        "in_type",
        "device",
    ]

    all_variables = vars(config)
    logger.info(
        "%s Training Configuration %s",
        "=" * 25,
        "=" * 25,
    )

    for key in print_order:
        if key in all_variables:
            logger.info("  %s: %s", key, all_variables[key])

    logger.info("=" * 60)


def _create_or_load_datasets(
    config: Any,
    array_positions: torch.Tensor,
    logger: Any,
    device: torch.device,
    train_filename: str,
    valid_filename: str,
):
    """Create new datasets or load existing datasets from disk."""
    train_directory = os.path.join(
        config.datasets_save_dir,
        "train",
    )
    valid_directory = os.path.join(
        config.datasets_save_dir,
        "valid",
    )
    os.makedirs(train_directory, exist_ok=True)
    os.makedirs(valid_directory, exist_ok=True)

    train_path = os.path.join(
        train_directory,
        train_filename,
    )
    valid_path = os.path.join(
        valid_directory,
        valid_filename,
    )

    if config.is_creat == "True":
        train_dataset, valid_dataset = create_train_dataset(
            num_sample=config.num_sample,
            N=config.N,
            M=config.M,
            T_list=config.T,
            snr=config.snr,
            tau=config.tau,
            lamda=config.lamda,
            coup_level=config.coup_level,
            eta=config.eta,
            logger=logger,
            Gap=config.Gap,
            array_pos=array_positions,
            d_init=config.d_init,
            coh_type=config.coh_type,
            band_type=config.band_type,
            datasets_ratio=config.datasets_ratio,
            device=device,
            deg_l=ANGLE_LOWER_BOUND,
            deg_u=ANGLE_UPPER_BOUND,
            deg_p=config.grid_step,
        )

        train_dataset_cpu = move_tensordataset_to_device(
            train_dataset,
            "cpu",
        )
        valid_dataset_cpu = move_tensordataset_to_device(
            valid_dataset,
            "cpu",
        )

        torch.save(train_dataset_cpu, train_path)
        torch.save(valid_dataset_cpu, valid_path)

        return train_dataset, valid_dataset

    if config.is_creat == "False":
        torch.serialization.add_safe_globals([TensorDataset])
        train_dataset = torch.load(
            train_path,
            map_location="cpu",
            weights_only=True,
        )
        valid_dataset = torch.load(
            valid_path,
            map_location="cpu",
            weights_only=True,
        )

        train_dataset = move_tensordataset_to_device(
            train_dataset,
            device,
        )
        valid_dataset = move_tensordataset_to_device(
            valid_dataset,
            device,
        )
        return train_dataset, valid_dataset

    raise ValueError("is_creat must be either 'True' or 'False', " f"got {config.is_creat!r}.")


def run_train(config: Any) -> None:
    """Configure and execute one training experiment."""
    device = torch.device(config.device)

    # Set the visible GPU before the first CUDA operation.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
    setup_seed(config.random_seed)

    array_positions = _create_array_positions(config)
    temp_string = format_scientific_notation(
        config.temp,
        1,
    )
    model_directory = _build_model_directory_name(
        config,
        temp_string,
    )

    timestamp = time.strftime(
        "%Y-%m-%d_%H-%M-%S",
        time.localtime(),
    )
    experiment_directory = os.path.join(
        RESULTS_ROOT,
        config.model_name,
        model_directory + timestamp,
    )
    os.makedirs(
        experiment_directory,
        exist_ok=True,
    )

    print(f"Using GPU: {1 if device.type == 'cuda' else 0}")

    logger = load_logger(experiment_directory)
    _log_configuration(config, logger)

    train_filename, valid_filename = _build_dataset_filenames(config)
    train_dataset, valid_dataset = _create_or_load_datasets(
        config=config,
        array_positions=array_positions,
        logger=logger,
        device=device,
        train_filename=train_filename,
        valid_filename=valid_filename,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    model, criterion = model_config(
        config.model_type,
        logger,
        config.N,
        config.M,
        config.fc,
        config.lamda,
        config.tau,
        array_positions,
        config.array_type,
        config.grid_step,
        config.temp,
        config.in_type,
        device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.step_size,
        gamma=config.gamma,
        last_epoch=-1,
    )

    solver = TrainSolver(
        config.num_epochs,
        config.checkpoint_path,
        config.resume,
        config.parallel,
        array_positions,
        train_loader,
        valid_loader,
        model,
        criterion,
        optimizer,
        experiment_directory,
        logger,
        scheduler,
        config.model_type,
        device,
    )
    solver.train()
