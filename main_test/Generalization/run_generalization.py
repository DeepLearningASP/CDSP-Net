"""Run the fixed-checkpoint generalization experiments."""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from main_test.Generalization.generalization_solver import GeneralizationTestSolver


def _validate_config(config):
    """Validate the supported signal model and checkpoints before writing output."""
    if config.test_type != "Generalization" or config.test_algorithm != "CDSP-Net":
        raise ValueError("Generalization testing requires CDSP-Net.")
    fixed = {"M": 2, "N": 8, "tau": 8, "lamda": 1.0, "d_init": 0.5,
             "Gap": 10, "grid_step": 1, "temp": 1}
    for name, value in fixed.items():
        if getattr(config, name) != value:
            raise ValueError(f"This evaluation requires {name}={value}.")
    for name in ("batch_size", "num_sample", "T"):
        value = getattr(config, name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if config.T < config.tau:
        raise ValueError("T must be at least tau.")
    if not isinstance(config.random_seed, int) or not 0 <= config.random_seed < 2**32:
        raise ValueError("random_seed must be an integer in [0, 2**32).")
    if config.model_labels != ["UnifiedMixed", "NoCoup", "Unified"]:
        raise ValueError("Expected UnifiedMixed, NoCoup, and Unified in this order.")
    scenes = config.scenarios()
    if not scenes:
        raise ValueError("No test conditions configured.")
    for scene in scenes:
        if (scene.coherence not in {"NonCoherent", "Coherent"}
                or scene.T != config.T or scene.eta < 0
                or not np.isfinite([scene.alpha, scene.eta, scene.snr]).all()
                or scene.gain_db != 0 or scene.phase_deg != 0
                or scene.noise_rho != 0 or scene.source_rho != -1):
            raise ValueError(f"Unsupported condition: {scene}")
    missing = [
        f"{label}: {config.checkpoint_path.get(label, '<unset>')}"
        for label in config.model_labels
        if not config.checkpoint_path.get(label)
        or not Path(config.checkpoint_path[label]).is_file()
    ]
    if missing:
        raise ValueError("Set the three trained checkpoint paths in config_generalization.py:\n  "
                         + "\n  ".join(missing))
    return scenes


def run_generalization(config):
    """Evaluate shared observations and return the summary CSV path."""
    scenes = _validate_config(config)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    result_path = Path(config.save_dir1) / config.test_type / config.test_algorithm / stamp
    print(f"Generalization: {len(scenes)} conditions, {len(config.model_labels)} models, "
          f"{config.num_sample} trials per condition; seed={config.random_seed}")
    solver = GeneralizationTestSolver(config, scenes, result_path)
    return solver.eva()

