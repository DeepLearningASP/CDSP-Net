"""Ablation-study execution utilities."""

import os
import time
from typing import Any, Sequence, Tuple

import torch

from ablation_solver import AblationTestSolver
from data_handler.CreateDataset import create_test_dataset
from tools.criterions import RMSPELoss_Vectorized
from tools.filename_handler import format_scientific_notation
from tools.logger import load_test_logger
from tools.seed import setup_seed


ANGLE_LOWER_BOUND = -60
ANGLE_UPPER_BOUND = 60

_CONFIG_LOG_ORDER = [
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


def _format_snr_range(values: Sequence[Any]) -> str:
    if len(values) < 2:
        return str(values[0])

    step = values[1] - values[0]
    return f"{values[0]}-{step}-{values[-1]}"


def _create_array_positions(config):
    if config.array_type != "ULA":
        raise ValueError(f"Unsupported array type: {config.array_type}")

    return torch.linspace(
        start=0,
        end=(config.N - 1) * config.d_init,
        steps=config.N,
        dtype=torch.float32,
        device=config.device,
    )


def _build_test_metadata(
    config,
) -> Tuple[Any, Any, str, str]:
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

    if config.test_type == "Ablation_SNR":
        snr_values = config.snr
        snapshots = [config.T]
        snr_string = _format_snr_range(snr_values)

        result_name = (
            f"_{config.array_type}_M{config.M}_N{config.N}_"
            f"T{snapshots}_Gap{config.Gap}_snr{snr_string}_"
            f"CoupLevel{config.coup_level}_"
            f"PosErrRatio{eta_string}_{config.band_type}_"
            f"{config.coh_type}"
        )
        test_filename = (
            f"test_dataset_{config.array_type}_M{config.M}_"
            f"N{config.N}_{config.angle_type}_Gap{config.Gap}_"
            f"T{snapshots}_tau{config.tau}_snr{snr_string}_"
            f"C{c_string}_fc{fc_string}_"
            f"CoupLevel{config.coup_level}_"
            f"PosErrRatio{eta_string}_{config.band_type}_"
            f"{config.coh_type}_samples{config.num_sample}.pt"
        )
        return snr_values, snapshots, result_name, test_filename

    if config.test_type == "Ablation_T":
        snr_values = [config.snr]
        snapshots = config.T

        result_name = (
            f"_{config.array_type}_M{config.M}_N{config.N}_"
            f"T{snapshots[0]}-{snapshots[-1]}_Gap{config.Gap}_"
            f"snr{snr_values[0]}_CoupLevel{config.coup_level}_"
            f"PosErrRatio{eta_string}_{config.band_type}_"
            f"{config.coh_type}"
        )
        test_filename = (
            f"test_dataset_{config.array_type}_M{config.M}_"
            f"N{config.N}_{config.angle_type}_Gap{config.Gap}_"
            f"T{snapshots[0]}-{snapshots[-1]}_tau{config.tau}_"
            f"snr{snr_values[0]}_C{c_string}_fc{fc_string}_"
            f"CoupLevel{config.coup_level}_"
            f"PosErrRatio{eta_string}_{config.band_type}_"
            f"{config.coh_type}_samples{config.num_sample}.pt"
        )
        return snr_values, snapshots, result_name, test_filename

    raise ValueError(f"Unsupported ablation test type: {config.test_type}")


def _log_configuration(config, logger):
    all_variables = vars(config)

    logger.info(
        "%s Evaluation Configuration %s",
        "=" * 25,
        "=" * 25,
    )
    for key in _CONFIG_LOG_ORDER:
        if key not in all_variables:
            continue

        value = all_variables[key]
        if key == "checkpoint_path" and isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                logger.info(
                    "  %s[%d]: %s",
                    key,
                    index,
                    item,
                )
        else:
            logger.info("  %s: %s", key, value)

    logger.info("=" * 60)


def _create_or_load_dataset(
    config,
    snapshots,
    snr_values,
    array_positions,
    logger,
    test_filename,
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
        test_dataset = create_test_dataset(
            config.num_sample,
            config.N,
            config.M,
            snapshots,
            snr_values,
            None,
            config.tau,
            config.lamda,
            config.coup_level,
            config.eta,
            logger,
            config.Gap,
            config.deta,
            config.fix,
            array_positions,
            config.d_init,
            config.coh_type,
            config.band_type,
            config.test_type,
            config.angle_type,
            config.device,
            ANGLE_LOWER_BOUND,
            ANGLE_UPPER_BOUND,
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

    raise ValueError("is_creat must be either 'True' or 'False', " f"got {config.is_creat!r}.")


def run_ablation(config):
    """Run one ablation-study evaluation."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
    setup_seed(config.random_seed)

    array_positions = _create_array_positions(config)
    (
        snr_values,
        snapshots,
        result_name,
        test_filename,
    ) = _build_test_metadata(config)

    timestamp = time.strftime(
        "%Y%m%d-%H%M%S",
        time.localtime(),
    )
    result_directory = os.path.join(
        config.save_dir1,
        config.test_type,
        config.test_algorithm,
        timestamp + result_name,
    )
    os.makedirs(
        result_directory,
        exist_ok=True,
    )

    logger = load_test_logger(result_directory)
    _log_configuration(config, logger)

    test_dataset = _create_or_load_dataset(
        config=config,
        snapshots=snapshots,
        snr_values=snr_values,
        array_positions=array_positions,
        logger=logger,
        test_filename=test_filename,
    )

    criterion = RMSPELoss_Vectorized(
        config.M,
        reduction="mean",
    )
    evaluator = AblationTestSolver(
        config.lamda,
        config.M,
        config.N,
        snapshots,
        config.grid_step,
        config.temp,
        array_positions,
        snr_values,
        config.test_type,
        config.test_algorithm,
        test_dataset,
        logger,
        criterion,
        result_directory,
        config.checkpoint_path,
        config.batch_size,
        config.device,
    )
    evaluator.eva()
