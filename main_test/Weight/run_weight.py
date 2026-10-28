"""Generate one shared fixed-angle observation and record pursuit weights."""

import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from data_handler.CreateDataset import create_test_dataset
from main_test.Weight.weight_solver import WeightTestSolver
from tools.seed import setup_seed


def _validate_config(config):
    """Fail before generating data or writing files if configuration is incomplete."""
    if config.test_type != "Weight" or config.test_algorithm != "CDSP-Net":
        raise ValueError("Weight testing requires test_type='Weight' and CDSP-Net.")
    if config.num_sample != 1 or config.batch_size != 1:
        raise ValueError("Weight testing requires num_sample=1 and batch_size=1.")
    if config.M != 2 or config.angle_type != "fix_angle":
        raise ValueError("This test requires M=2 and angle_type='fix_angle'.")
    if config.array_type != "ULA":
        raise ValueError("This weight test currently supports ULA only.")

    angles = np.asarray(config.fix, dtype=float)
    if angles.shape != (config.M,) or not np.all(np.isfinite(angles)):
        raise ValueError("fix must contain one finite angle in degrees per source.")
    if np.any(np.abs(angles) > 60) or np.any(np.diff(np.sort(angles)) < config.Gap):
        raise ValueError("Fixed angles must lie in [-60, 60] and satisfy Gap.")

    temperatures = np.asarray(config.temp, dtype=float)
    if (temperatures.ndim != 1 or temperatures.size == 0
            or not np.all(np.isfinite(temperatures)) or np.any(temperatures <= 0)
            or len(np.unique(temperatures)) != len(temperatures)):
        raise ValueError("temp must contain distinct finite positive temperatures.")
    if len(config.checkpoint_path) != len(temperatures):
        raise ValueError("checkpoint_path and temp must have the same length.")

    missing = [f"t_p={temp:g}: {path or '<empty>'}"
               for temp, path in zip(temperatures, config.checkpoint_path)
               if not path or not Path(path).is_file()]
    if missing:
        raise ValueError("Fill in valid temperature-matched checkpoints:\n  "
                         + "\n  ".join(missing))


def _create_shared_observation(config, array_pos, logger):
    # Weight is a singleton protocol: create_test_dataset generates only one scene,
    # rather than regenerating signals/noise once for each temperature.
    corr_all, _, doa_all, _ = create_test_dataset(
        num_sample=config.num_sample,
        N=config.N,
        M=config.M,
        Ts=config.T,
        snrs=[config.snr],
        params=None,
        tau=config.tau,
        lamda=config.lamda,
        coup_levels=config.coup_level,
        etas=config.eta,
        logger=logger,
        Gap=config.Gap,
        deta=config.deta,
        fix=config.fix,
        array_pos=array_pos,
        d_init=config.d_init,
        coh_type=config.coh_type,
        band_type=config.band_type,
        test_type=config.test_type,
        angle_type=config.angle_type,
        device=config.device,
        deg_l=-60,
        deg_u=60,
        deg_p=config.grid_step,
    )
    # corr_all: [singleton_condition, sample=1, tau, 2*N, N]
    expected = (1, 1, config.tau, 2 * config.N, config.N)
    if tuple(corr_all.shape) != expected:
        raise ValueError(f"Unexpected observation shape: {tuple(corr_all.shape)}")
    torch.testing.assert_close(
        torch.rad2deg(doa_all[0, 0]),
        torch.tensor(config.fix, dtype=torch.float32, device=config.device),
        atol=1e-5,
        rtol=0,
    )
    return corr_all[0]  # [sample=1, tau, 2*N, N]


def run_weight(config):
    """Save weights only; do not persist the dataset, predictions, or metrics."""
    _validate_config(config)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
    setup_seed(config.random_seed)

    logger = logging.getLogger(__name__)
    array_pos = torch.linspace(
        0, (config.N - 1) * config.d_init, config.N,
        dtype=torch.float32, device=config.device,
    )
    print(f"Weight test: fixed DOAs {config.fix} deg; seed={config.random_seed}")
    print(f"One shared scene: N={config.N}, T={config.T}, SNR={config.snr} dB")
    observation = _create_shared_observation(config, array_pos, logger)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    angle_tag = "_".join(f"{angle:g}" for angle in config.fix)
    result_name = (
        f"{stamp}_{config.array_type}_M{config.M}_N{config.N}_T{config.T}"
        f"_snr{config.snr}_DOA{angle_tag}_seed{config.random_seed}"
    )
    result_path = Path(config.save_dir1) / config.test_type / config.test_algorithm / result_name

    solver = WeightTestSolver(config, array_pos, observation, result_path)
    return solver.eva()
