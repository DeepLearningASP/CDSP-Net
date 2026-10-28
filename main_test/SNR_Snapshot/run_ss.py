"""Entry point for SNR and snapshot evaluation."""

import os
import time
from typing import Any, Sequence, Tuple

import torch

from data_handler.CreateDataset import create_test_dataset
from ss_solver import SnrSnapshot_TestSolver
from tools.criterions import RMSPELoss_Vectorized
from tools.filename_handler import format_scientific_notation
from tools.logger import load_test_logger
from tools.seed import setup_seed


CONFIG_PRINT_ORDER = [
    "batch_size",
    "checkpoint_path",
    "datasets_save_dir",
    "save_dir1",
    "random_seed",
    "gpu",
    "M",
    "N",
    "tau",
    "c",
    "fc",
    "lamda",
    "d_init",
    "grid_step",
    "temp",
    "spectrum_step",
    "Gap",
    "success_tol_deg",
    "deta",
    "fix",
    "coup_level",
    "eta_Ratio",
    "eta",
    "num_sample",
    "is_creat",
    "test_type",
    "test_algorithm",
    "array_type",
    "band_type",
    "coh_type",
    "angle_type",
    "device",
    "T",
    "snr",
]


def _format_range(values: Sequence[Any]) -> str:
    if len(values) < 2:
        return str(values[0])

    step = values[1] - values[0]
    return f"{values[0]}-{step}-{values[-1]}"


def _build_experiment_names(
    config: Any,
) -> Tuple[Sequence[Any], Sequence[Any], str, str]:
    c_string = format_scientific_notation(
        config.c,
        0,
    )
    fc_string = format_scientific_notation(
        config.fc,
        0,
    )
    eta_string = format_scientific_notation(
        config.eta_Ratio,
        0,
    )

    if config.test_type == "SNR":
        snr_values = config.snr
        snapshot_values = [config.T]
        snr_range = _format_range(snr_values)

        result_name = (
            f"_{config.array_type}_M{config.M}_N{config.N}_"
            f"T{snapshot_values}_Gap{config.Gap}_snr{snr_range}_"
            f"CoupLevel{config.coup_level}_PosErrRatio{eta_string}_"
            f"{config.band_type}_{config.coh_type}"
        )
        test_filename = (
            f"test_dataset_{config.array_type}_M{config.M}_N{config.N}_"
            f"{config.angle_type}_Gap{config.Gap}_T{snapshot_values}_"
            f"tau{config.tau}_snr{snr_range}_C{c_string}_fc{fc_string}_"
            f"CoupLevel{config.coup_level}_PosErrRatio{eta_string}_"
            f"{config.band_type}_{config.coh_type}_samples{config.num_sample}.pt"
        )
        return (
            snr_values,
            snapshot_values,
            result_name,
            test_filename,
        )

    if config.test_type == "Snapshot":
        snr_values = [config.snr]
        snapshot_values = config.T

        result_name = (
            f"_{config.array_type}_M{config.M}_N{config.N}_"
            f"T{snapshot_values[0]}-{snapshot_values[-1]}_Gap{config.Gap}_"
            f"snr{snr_values[0]}_CoupLevel{config.coup_level}_"
            f"PosErrRatio{eta_string}_{config.band_type}_{config.coh_type}"
        )
        test_filename = (
            f"test_dataset_{config.array_type}_M{config.M}_N{config.N}_"
            f"{config.angle_type}_Gap{config.Gap}_"
            f"T{snapshot_values[0]}-{snapshot_values[-1]}_"
            f"tau{config.tau}_snr{snr_values[0]}_C{c_string}_fc{fc_string}_"
            f"CoupLevel{config.coup_level}_PosErrRatio{eta_string}_"
            f"{config.band_type}_{config.coh_type}_samples{config.num_sample}.pt"
        )
        return (
            snr_values,
            snapshot_values,
            result_name,
            test_filename,
        )

    raise ValueError(
        f"Unsupported test type: {config.test_type}"
    )


def _create_array_positions(
    config: Any,
) -> torch.Tensor:
    return torch.linspace(
        start=0,
        end=(config.N - 1) * config.d_init,
        steps=config.N,
        dtype=torch.float32,
        device=config.device,
    )


def _create_result_path(
    config: Any,
    result_name: str,
) -> str:
    timestamp = time.strftime(
        "%Y%m%d-%H%M%S",
        time.localtime(),
    )
    result_path = os.path.join(
        config.save_dir1,
        config.test_type,
        config.test_algorithm,
        timestamp + result_name,
    )
    os.makedirs(
        result_path,
        exist_ok=True,
    )
    return result_path


def _log_configuration(
    config: Any,
    logger: Any,
) -> None:
    all_variables = vars(config)

    logger.info(
        "%s Test Configuration %s",
        "=" * 25,
        "=" * 25,
    )
    for key in CONFIG_PRINT_ORDER:
        if key not in all_variables:
            continue

        value = all_variables[key]
        if (
            key == "checkpoint_path"
            and isinstance(value, (list, tuple))
        ):
            for index, item in enumerate(value):
                logger.info(
                    "  %s[%d]: %s",
                    key,
                    index,
                    item,
                )
        else:
            logger.info(
                "  %s: %s",
                key,
                value,
            )

    logger.info("=" * 60)


def _load_or_create_dataset(
    config: Any,
    snr_values: Sequence[Any],
    snapshot_values: Sequence[Any],
    array_pos: torch.Tensor,
    logger: Any,
    test_filename: str,
):
    test_directory = os.path.join(
        config.datasets_save_dir,
        "test",
    )
    test_path = os.path.join(
        test_directory,
        test_filename,
    )

    if config.is_creat == "True":
        os.makedirs(
            test_directory,
            exist_ok=True,
        )
        parameter = None
        test_dataset = create_test_dataset(
            config.num_sample,
            config.N,
            config.M,
            snapshot_values,
            snr_values,
            parameter,
            config.tau,
            config.lamda,
            config.coup_level,
            config.eta,
            logger,
            config.Gap,
            config.deta,
            config.fix,
            array_pos,
            config.d_init,
            config.coh_type,
            config.band_type,
            config.test_type,
            config.angle_type,
            config.device,
            -60,
            60,
            config.grid_step,
        )
        torch.save(
            test_dataset,
            test_path,
        )
        return test_dataset

    if config.is_creat == "False":
        return torch.load(
            test_path,
            weights_only=True,
        )

    raise ValueError(
        "is_creat must be either 'True' or 'False', "
        f"got {config.is_creat!r}."
    )


def run_snr_snapshot_test(
    config: Any,
) -> None:
    """Run one SNR or snapshot evaluation experiment."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        config.gpu
    )
    setup_seed(config.random_seed)

    (
        snr_values,
        snapshot_values,
        result_name,
        test_filename,
    ) = _build_experiment_names(config)

    array_pos = _create_array_positions(config)
    result_path = _create_result_path(
        config,
        result_name,
    )

    logger = load_test_logger(result_path)
    _log_configuration(
        config,
        logger,
    )

    test_dataset = _load_or_create_dataset(
        config=config,
        snr_values=snr_values,
        snapshot_values=snapshot_values,
        array_pos=array_pos,
        logger=logger,
        test_filename=test_filename,
    )

    criterion = RMSPELoss_Vectorized(
        config.M,
        reduction="mean",
    )
    evaluator = SnrSnapshot_TestSolver(
        config.lamda,
        config.tau,
        config.M,
        config.N,
        snapshot_values,
        config.fc,
        config.grid_step,
        config.temp,
        config.spectrum_step,
        array_pos,
        snr_values,
        config.test_type,
        config.coh_type,
        config.test_algorithm,
        test_dataset,
        logger,
        criterion,
        result_path,
        config.checkpoint_path,
        config.batch_size,
        config.device,
        config.success_tol_deg,
    )
    evaluator.eva()


def run_ss(
    config: Any,
) -> None:
    """Backward-compatible entry point."""
    run_snr_snapshot_test(config)
