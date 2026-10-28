"""CDF evaluation entry point."""

import os
import time
from typing import Any, Tuple

import torch

from cdf_solver import CDFTestSolver
from data_handler.CreateDataset import create_test_dataset
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
    "T",
    "snr",
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
    "cdf_x_deg",
    "device",
]


def _build_names(
    config: Any,
) -> Tuple[str, str]:
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

    result_name = (
        f"_{config.array_type}_M{config.M}_N{config.N}_"
        f"T{config.T}_Gap{config.Gap}_snr{config.snr}"
        f"_CoupLevel{config.coup_level}_PosErrRatio{eta_string}_"
        f"{config.band_type}_{config.coh_type}"
    )
    test_filename = (
        f"test_dataset_{config.array_type}_M{config.M}_N{config.N}_"
        f"{config.angle_type}_Gap{config.Gap}_T{config.T}_"
        f"tau{config.tau}_snr{config.snr}_C{c_string}_fc{fc_string}_"
        f"CoupLevel{config.coup_level}_PosErrRatio{eta_string}_"
        f"{config.band_type}_{config.coh_type}_samples{config.num_sample}.pt"
    )
    return result_name, test_filename


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
    result_directory = timestamp + result_name
    result_path = os.path.join(
        config.save_dir1,
        config.test_type,
        config.test_algorithm,
        result_directory,
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
        snr = [config.snr]
        param = None
        test_dataset = create_test_dataset(
            config.num_sample,
            config.N,
            config.M,
            config.T,
            snr,
            param,
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


def run_cdf(config: Any) -> None:
    """Run one CDF evaluation experiment."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        config.gpu
    )
    setup_seed(config.random_seed)

    array_pos = _create_array_positions(config)
    result_name, test_filename = _build_names(
        config
    )
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
        config,
        array_pos,
        logger,
        test_filename,
    )

    criterion = RMSPELoss_Vectorized(
        config.M,
        reduction="batch",
    )
    evaluator = CDFTestSolver(
        config.lamda,
        config.tau,
        config.M,
        config.N,
        config.T,
        config.snr,
        config.fc,
        config.grid_step,
        config.temp,
        config.spectrum_step,
        array_pos,
        config.cdf_x_deg,
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
    )
    evaluator.eva()


def run_cdf_test(config: Any) -> None:
    """Backward-compatible alias for the original entry point."""
    run_cdf(config)
