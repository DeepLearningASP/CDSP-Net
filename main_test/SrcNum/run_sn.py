"""Source-number evaluation entry point."""

import os
import time
from typing import Any, Sequence, Tuple

import torch

from data_handler.CreateDataset import create_test_dataset
from sn_solver import SrcNum_TestSolver
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
    "M_list",
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
    "device",
]


def _build_names(
    config: Any,
    source_counts: Sequence[int],
    snapshots: Sequence[int],
    snrs: Sequence[float],
) -> Tuple[str, str]:
    c_string = format_scientific_notation(config.c, 0)
    fc_string = format_scientific_notation(config.fc, 0)
    eta_string = format_scientific_notation(
        config.eta_Ratio,
        0,
    )

    source_range = (
        f"{source_counts[0]}-{source_counts[-1]}"
    )
    result_name = (
        f"_{config.array_type}_M{source_range}_N{config.N}_"
        f"T{snapshots[0]}_Gap{config.Gap}_snr{snrs[0]}_"
        f"CoupLevel{config.coup_level}_PosErrRatio{eta_string}_"
        f"{config.band_type}_{config.coh_type}"
    )
    test_filename = (
        f"test_dataset_{config.array_type}_M{source_range}_N{config.N}_"
        f"{config.angle_type}_Gap{config.Gap}_T{snapshots[0]}_"
        f"tau{config.tau}_snr{snrs[0]}_C{c_string}_fc{fc_string}_"
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
    config_values = vars(config)

    logger.info(
        "%s Test Configuration %s",
        "=" * 25,
        "=" * 25,
    )
    for key in CONFIG_PRINT_ORDER:
        if key not in config_values:
            continue

        value = config_values[key]
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
    source_counts: Sequence[int],
    snapshots: Sequence[int],
    snrs: Sequence[float],
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
        test_dataset = create_test_dataset(
            config.num_sample,
            config.N,
            source_counts,
            snapshots,
            snrs,
            None,
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

    return torch.load(
        test_path,
        weights_only=True,
    )


def run_src_num_test(
    config: Any,
) -> None:
    """Run a source-number evaluation experiment."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        config.gpu
    )
    setup_seed(config.random_seed)

    source_counts = list(config.M_list)
    snapshots = [config.T]
    snrs = [config.snr]

    array_pos = _create_array_positions(config)
    result_name, test_filename = _build_names(
        config,
        source_counts,
        snapshots,
        snrs,
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
        source_counts,
        snapshots,
        snrs,
        array_pos,
        logger,
        test_filename,
    )

    evaluator = SrcNum_TestSolver(
        config.lamda,
        config.tau,
        source_counts,
        config.N,
        snapshots,
        config.fc,
        config.grid_step,
        config.temp,
        config.spectrum_step,
        array_pos,
        snrs,
        config.test_type,
        config.coh_type,
        config.test_algorithm,
        test_dataset,
        logger,
        result_path,
        config.checkpoint_path,
        config.batch_size,
        config.device,
    )
    evaluator.eva()


def run_sn(
    config: Any,
) -> None:
    """Backward-compatible entry point."""
    run_src_num_test(config)
