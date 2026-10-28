"""Training configuration and sequential task scheduler.

Usage:
    1. Set shared defaults in the base configuration section.
    2. Add task-specific overrides to ``TASK_LIST``.
    3. Run the scheduler entry script to execute each task in an isolated process.
"""

import sys
from types import ModuleType
from typing import Any, Optional, Tuple
import torch


# =========================== Base Configuration ===========================
# Training hyperparameters
num_epochs = 120
batch_size = 64
warm_up_epochs = 3
learning_rate = 0.0005
step_size = 40
gamma = 0.2
weight_decay = 0.0001
checkpoint_path = r"D:\train_result\CDSP_Net_Model\CDSP-Net_ULA_M2_N8_T[200]_Gap10_grid1_temp1.0e-2_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-06-29_11-22-52\Model_epoch12.pth"  # Set a checkpoint path when resume=True
datasets_save_dir = r"D:\experiment\CDSP-Net\datasets"
resume = False
parallel = False
random_seed = 20260513
gpu = "0"
model_name = "CDSP-Net"

# Array and signal configuration
M = 2                                   # Number of sources
N = 8                                   # Number of array elements
tau = 8                                 # Number of covariance lags
c = 3e8                                 # Speed of light
fc = 3e8                                # Carrier frequency
T = [200]                               # Number of snapshots
snr = [-10, -5, 0]                      # Signal-to-noise ratios
lamda = c / fc                          # Wavelength (derived)
d_init = 0.5 * lamda                    # Initial element spacing (derived)
delay_max = (N - 1) * d_init / c        # Maximum propagation delay (derived)
grid_step = 1                           # Grid spacing of the overcomplete OMP dictionary
temp = 1                              # Soft-sorting temperature
Gap = 10                                # Minimum DOA separation
coup_level = [-999]                     # Mutual-coupling level; -999 disables coupling
eta_Ratio = [0.0]                       # Relative array-position error
eta = [ratio * d_init for ratio in eta_Ratio]  # Position error (derived)
num_sample = 100000                     # Number of samples
datasets_ratio = [0.8, 0.2]             # Training/validation split
is_creat = "True"                       # "True": generate data; "False": load existing data
array_type = "ULA"                      # Array geometry: ULA
band_type = "Narrow"                    # Signal bandwidth: Narrow
coh_type = "NonCoherent"                # Source coherence: Coherent / NonCoherent / Mixed (50:50)
model_type = "CDSP-Net"                  # Model variant, e.g. CDSP-Net-Top-M
in_type = "VSCM"                        # Input representation
device = torch.device("cuda")


# =========================== Previous Task Examples (Not Scheduled) ===========================
TASK_LIST = [
    # Temp
    # {'model_type': 'CDSP-Net', 'T': [200], 'snr': [10, 10], 'temp': 0.001, 'coh_type': 'NonCoherent', 'is_creat': 'True'},
    # {'model_type': 'CDSP-Net', 'T': [200], 'snr': [10, 10], 'temp': 0.01, 'coh_type': 'NonCoherent', 'is_creat': 'False'},
    # {'model_type': 'CDSP-Net', 'T': [200], 'snr': [10, 10], 'temp': 0.1, 'coh_type': 'NonCoherent', 'is_creat': 'False'},
    # {'model_type': 'CDSP-Net', 'T': [200], 'snr': [10, 10], 'temp': 1, 'coh_type': 'NonCoherent', 'is_creat': 'False'},
    # {'model_type': 'CDSP-Net', 'T': [200], 'snr': [10, 10], 'temp': 10, 'coh_type': 'NonCoherent', 'is_creat': 'False'},
    # {'model_type': 'CDSP-Net', 'T': [200], 'snr': [10, 10], 'temp': 100, 'coh_type': 'NonCoherent', 'is_creat': 'False'},

    # # Ablation_Input
    # {'model_type': 'CDSP-Net', 'T': [10], 'snr': [10, 10], 'coh_type': 'NonCoherent', 'is_creat': 'True'},
    # {'model_type': 'CDSP-Net-VSCM', 'T': [10], 'snr': [10, 10], 'in_type': 'VSCM', 'coh_type': 'NonCoherent', 'is_creat': 'False'},
    # {'model_type': 'CDSP-Net-MCV', 'T': [10], 'snr': [10, 10], 'in_type': 'MCV', 'coh_type': 'NonCoherent', 'is_creat': 'False'},

    # Ablation_test
    # {'model_type': 'CDSP-Net-Top-M', 'T': [200], 'snr': [10, 10], 'coh_type': 'NonCoherent', 'is_creat': 'False'},
    # {'model_type': 'CDSP-Net-SD', 'T': [200], 'snr': [10, 10], 'coh_type': 'NonCoherent', 'is_creat': 'False'},
    # {'model_type': 'CDSP-Net-HS', 'T': [200], 'snr': [10, 10], 'coh_type': 'NonCoherent', 'is_creat': 'False'},
    #
    # # Grid
    # {'model_type': 'CDSP-Net', 'T': [200], 'snr': [10, 10], 'grid_step': 0.1, 'coh_type': 'NonCoherent', 'is_creat': 'False'},
    # {'model_type': 'CDSP-Net', 'T': [200], 'snr': [10, 10], 'grid_step': 2, 'coh_type': 'NonCoherent', 'is_creat': 'False'},
    # {'model_type': 'CDSP-Net', 'T': [200], 'snr': [10, 10], 'grid_step': 5, 'coh_type': 'NonCoherent', 'is_creat': 'False'},
    # {"model_type": "CDSP-Net", "T": [200], "snr": [10, 10], "grid_step": 10, "coh_type": "NonCoherent", "is_creat": "False"},
    # {"model_type": "CDSP-Net", "T": [200], "snr": [10, 10], "grid_step": 20, "coh_type": "NonCoherent", "is_creat": "False"},
    # {"model_type": "CDSP-Net", "T": [200], "snr": [10, 10], "grid_step": 30, "coh_type": "NonCoherent", "is_creat": "False"},
    # {'model_type': 'CDSP-Net-SD', 'T': [200], 'snr': [10, 10], 'grid_step': 0.1, 'coh_type': 'NonCoherent', 'is_creat': 'False'},
    # {'model_type': 'CDSP-Net-SD', 'T': [200], 'snr': [10, 10], 'grid_step': 2, 'coh_type': 'NonCoherent', 'is_creat': 'False'},
    # {'model_type': 'CDSP-Net-SD', 'T': [200], 'snr': [10, 10], 'grid_step': 5, 'coh_type': 'NonCoherent', 'is_creat': 'False'},
    # {"model_type": "CDSP-Net-SD", "T": [200], "snr": [10, 10], "grid_step": 10, "coh_type": "NonCoherent", "is_creat": "False"},
    # {'model_type': 'CDSP-Net-SD', 'T': [200], 'snr': [10, 10], 'grid_step': 20, 'coh_type': 'NonCoherent', 'is_creat': 'False'},
    # {"model_type": "CDSP-Net-SD", "T": [200], "snr": [10, 10], "grid_step": 30, "coh_type": "NonCoherent", "is_creat": "False"},

    # # Gap
    # {"model_type": "CDSP-Net", "T": [200], "snr": [10, 10], "Gap": 0, "coh_type": "NonCoherent", "is_creat": "True"},
    # {"model_type": "SubspaceNet", "T": [200], "snr": [10, 10], "Gap": 0, "coh_type": "NonCoherent", "is_creat": "False"},
    # {"model_type": "DeepCNN", "T": [200], "snr": [10, 10], "Gap": 0, "coh_type": "NonCoherent", "is_creat": "False"},
    #
    # # low snr & few snapshots
    # {'model_type': 'CDSP-Net', 'T': [2, 6, 10], 'snr': [10, 10], 'coh_type': 'NonCoherent', 'is_creat': 'True'},
    # {'model_type': 'SubspaceNet', 'T': [2, 6, 10], 'snr': [10, 10], 'coh_type': 'NonCoherent', 'is_creat': 'False'},
    # {'model_type': 'DeepCNN', 'T': [2, 6, 10], 'snr': [10, 10], 'coh_type': 'NonCoherent', 'is_creat': 'False'},
    # {'model_type': 'CDSP-Net', 'T': [200], 'snr': [-10, -5, 0], 'coh_type': 'NonCoherent', 'is_creat': 'True'},
    # {'model_type': 'SubspaceNet', 'T': [200], 'snr': [-10, -5, 0], 'coh_type': 'NonCoherent', 'is_creat': 'False'},
    # {'model_type': 'DeepCNN', 'T': [200], 'snr': [-10, -5, 0], 'coh_type': 'NonCoherent', 'is_creat': 'False'},

    # Coherent
    # {'model_type': 'CDSP-Net', 'T': [10, 50, 200], 'snr': [10, 10], 'coh_type': 'Coherent', 'is_creat': 'True'},
    # {'model_type': 'SubspaceNet', 'T': [10, 50, 200], 'snr': [10, 10], 'coh_type': 'Coherent', 'is_creat': 'False'},
    # {'model_type': 'DeepCNN', 'T': [10, 50, 200], 'snr': [10, 10], 'coh_type': 'Coherent', 'is_creat': 'False'},
    # {'model_type': 'CDSP-Net', 'T': [200], 'snr': [10, 10], 'coh_type': 'Coherent', 'M': 2, 'Gap': 0, 'is_creat': 'True'},
    # {'model_type': 'CDSP-Net', 'T': [200], 'snr': [10, 10], 'coh_type': 'Coherent', 'M': 3, 'Gap': 0, 'is_creat': 'True'},
    # {'model_type': 'CDSP-Net', 'T': [200], 'snr': [10, 10], 'coh_type': 'Coherent', 'M': 4, 'Gap': 0, 'is_creat': 'True'},
    # {'model_type': 'CDSP-Net', 'T': [200], 'snr': [10, 10], 'coh_type': 'Coherent', 'M': 5, 'Gap': 0, 'is_creat': 'True'},
    # {'model_type': 'SubspaceNet', 'T': [200], 'snr': [10, 10], 'coh_type': 'Coherent', 'M': 2, 'Gap': 0, 'is_creat': 'False'},
    # {'model_type': 'SubspaceNet', 'T': [200], 'snr': [10, 10], 'coh_type': 'Coherent', 'M': 3, 'Gap': 0, 'is_creat': 'False'},
    # {'model_type': 'SubspaceNet', 'T': [200], 'snr': [10, 10], 'coh_type': 'Coherent', 'M': 4, 'Gap': 0, 'is_creat': 'False'},
    # {'model_type': 'SubspaceNet', 'T': [200], 'snr': [10, 10], 'coh_type': 'Coherent', 'M': 5, 'Gap': 0, 'is_creat': 'False'},
    # {'model_type': 'DeepCNN', 'T': [200], 'snr': [10, 10], 'coh_type': 'Coherent', 'M': 2, 'Gap': 0, 'is_creat': 'False'},
    # {'model_type': 'DeepCNN', 'T': [200], 'snr': [10, 10], 'coh_type': 'Coherent', 'M': 3, 'Gap': 0, 'is_creat': 'False'},
    # {'model_type': 'DeepCNN', 'T': [200], 'snr': [10, 10], 'coh_type': 'Coherent', 'M': 4, 'Gap': 0, 'is_creat': 'False'},
    # {'model_type': 'DeepCNN', 'T': [200], 'snr': [10, 10], 'coh_type': 'Coherent', 'M': 5, 'Gap': 0, 'is_creat': 'False'},
    # {'model_type': 'CDSP-Net', 'T': [200], 'snr': [-10, -5, 0], 'coh_type': 'Coherent', 'is_creat': 'True'},
    # {'model_type': 'SubspaceNet', 'T': [200], 'snr': [-10, -5, 0], 'coh_type': 'Coherent', 'is_creat': 'False'},
    # {'model_type': 'DeepCNN', 'T': [200], 'snr': [-10, -5, 0], 'coh_type': 'Coherent', 'is_creat': 'False'},
    #
    # # Coup
    # {'model_type': 'CDSP-Net', 'T': [200], 'snr': [10, 10], 'coh_type': 'NonCoherent', 'coup_level': [0, -5, -10, -15, -20, -25, -30], 'is_creat': 'True'},
    # {'model_type': 'SubspaceNet', 'T': [200], 'snr': [10, 10], 'coh_type': 'NonCoherent', 'coup_level': [0, -5, -10, -15, -20, -25, -30], 'is_creat': 'False'},
    # {'model_type': 'DeepCNN', 'T': [200], 'snr': [10, 10], 'coh_type': 'NonCoherent', 'coup_level': [0, -5, -10, -15, -20, -25, -30], 'is_creat': 'False'},
    # {'model_type': 'CDSP-Net', 'T': [200], 'snr': [-10, -10], 'coh_type': 'Coherent', 'coup_level': [0, -5, -10, -15, -20, -25, -30], 'is_creat': 'True'},
    # {'model_type': 'SubspaceNet', 'T': [200], 'snr': [-10, -10], 'coh_type': 'Coherent', 'coup_level': [0, -5, -10, -15, -20, -25, -30], 'is_creat': 'False'},
    # {'model_type': 'DeepCNN', 'T': [200], 'snr': [-10, -10], 'coh_type': 'Coherent', 'coup_level': [0, -5, -10, -15, -20, -25, -30], 'is_creat': 'False'},

    # # Pos Err
    # {'model_type': 'CDSP-Net', 'T': [200], 'snr': [10, 10], 'coh_type': 'NonCoherent', 'eta_Ratio': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25], 'is_creat': 'True'},
    # {'model_type': 'SubspaceNet', 'T': [200], 'snr': [10, 10], 'coh_type': 'NonCoherent', 'eta_Ratio': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25], 'is_creat': 'False'},
    # {'model_type': 'DeepCNN', 'T': [200], 'snr': [10, 10], 'coh_type': 'NonCoherent', 'eta_Ratio': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25], 'is_creat': 'False'},
    # {'model_type': 'CDSP-Net', 'T': [2], 'snr': [10, 10], 'coh_type': 'Coherent', 'coup_level': [-10], 'eta_Ratio': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25], 'is_creat': 'True'},
    # {'model_type': 'SubspaceNet', 'T': [2], 'snr': [10, 10], 'coh_type': 'Coherent', 'coup_level': [-10], 'eta_Ratio': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25], 'is_creat': 'False'},
    # {'model_type': 'DeepCNN', 'T': [2], 'snr': [10, 10], 'coh_type': 'Coherent', 'coup_level': [-10], 'eta_Ratio': [0.0, 0.05, 0.1, 0.15, 0.2, 0.25], 'is_creat': 'False'},
    # {'model_type': 'CDSP-Net', 'T': [2], 'snr': [10, 10], 'coh_type': 'Coherent', 'coup_level': [-10], 'eta_Ratio': [0.15], 'is_creat': 'True'},
    # {'model_type': 'SubspaceNet', 'T': [2], 'snr': [10, 10], 'coh_type': 'Coherent', 'coup_level': [-10], 'eta_Ratio': [0.15], 'is_creat': 'False'},
    # {'model_type': 'DeepCNN', 'T': [2], 'snr': [10, 10], 'coh_type': 'Coherent', 'coup_level': [-10], 'eta_Ratio': [0.15], 'is_creat': 'False'},

    # Generalization
    # {'model_type': 'CDSP-Net', 'model_name': 'Gen_Unified', 'T': [10, 50, 200], 'snr': [-5, 0, 5, 10], 'coup_level': [-999, -30, -20, -10], 'eta_Ratio': [0.0, 0.05, 0.1, 0.15], 'is_creat': 'True', 'datasets_save_dir': 'D:\\experiment\\CDSP-Net\\datasets\\Generalization'},
    # {'model_type': 'CDSP-Net', 'model_name': 'Gen_NoCoup', 'T': [10, 50, 200], 'snr': [-5, 0, 5, 10], 'eta_Ratio': [0.0, 0.05, 0.1, 0.15], 'is_creat': 'True', 'datasets_save_dir': 'D:\\experiment\\CDSP-Net\\datasets\\Generalization'},
    {'model_type': 'CDSP-Net', 'model_name': 'Gen_Nominal', 'snr': [10], 'is_creat': 'True', 'datasets_save_dir': 'D:\\experiment\\CDSP-Net\\datasets\\GeneralizationV2'},
    {'model_type': 'CDSP-Net', 'model_name': 'Gen_ObsOnly', 'T': [10, 50, 200], 'snr': [-5, 0, 5, 10], 'is_creat': 'True', 'datasets_save_dir': 'D:\\experiment\\CDSP-Net\\datasets\\GeneralizationV2'},
    {'model_type': 'CDSP-Net', 'model_name': 'Gen_NoPos', 'T': [10, 50, 200], 'snr': [-5, 0, 5, 10], 'coup_level': [-999, -30, -20, -10], 'is_creat': 'True', 'datasets_save_dir': 'D:\\experiment\\CDSP-Net\\datasets\\GeneralizationV2'},
    {'model_type': 'CDSP-Net', 'model_name': 'Gen_UnifiedMixed', 'T': [10, 50, 200], 'snr': [-5, 0, 5, 10], 'coup_level': [-999, -30, -20, -10], 'eta_Ratio': [0.0, 0.05, 0.1, 0.15], 'coh_type': 'Mixed', 'is_creat': 'True', 'datasets_save_dir': 'D:\\experiment\\CDSP-Net\\datasets\\GeneralizationV2'},

    # Complexity
    {'num_epochs': 1, 'model_type': 'CDSP-Net', 'T': [200], 'snr': [10, 10], 'coh_type': 'NonCoherent', 'is_creat': 'True'},
    {'num_epochs': 1, 'model_type': 'SubspaceNet', 'T': [200], 'snr': [10, 10], 'coh_type': 'NonCoherent', 'is_creat': 'False'},
    {'num_epochs': 1, 'model_type': 'DeepCNN', 'T': [200], 'snr': [10, 10], 'coh_type': 'NonCoherent', 'is_creat': 'False'},
    {'num_epochs': 1, 'model_type': 'CDSP-Net', 'M': 3, 'N': 8, 'T': [200], 'snr': [10, 10], 'coh_type': 'NonCoherent', 'is_creat': 'True'},
    {'num_epochs': 1, 'model_type': 'CDSP-Net', 'M': 4, 'N': 8, 'T': [200], 'snr': [10, 10], 'coh_type': 'NonCoherent', 'is_creat': 'True'},
    {'num_epochs': 1, 'model_type': 'CDSP-Net', 'M': 5, 'N': 8, 'T': [200], 'snr': [10, 10], 'coh_type': 'NonCoherent', 'is_creat': 'True'},
    {'num_epochs': 1, 'model_type': 'CDSP-Net', 'M': 2, 'N': 9, 'T': [200], 'snr': [10, 10], 'coh_type': 'NonCoherent', 'is_creat': 'True'},
    {'num_epochs': 1, 'model_type': 'CDSP-Net', 'M': 2, 'N': 10, 'T': [200], 'snr': [10, 10], 'coh_type': 'NonCoherent', 'is_creat': 'True'},
    {'num_epochs': 1, 'model_type': 'CDSP-Net', 'M': 2, 'N': 11, 'T': [200], 'snr': [10, 10], 'coh_type': 'NonCoherent', 'is_creat': 'True'},
]


# Adjacent tasks reuse a dataset when all parameters below are identical.
# An explicit is_creat value always overrides automatic resolution.
DATASET_KEYS = [
    "M",
    "N",
    "T",
    "snr",
    "Gap",
    "coup_level",
    "eta_Ratio",
    "num_sample",
    "array_type",
    "band_type",
    "coh_type",
]

COOLDOWN_SEC = 10  # Delay after each task to allow GPU memory cleanup
START_DELAY_SEC = 2  # Delay before starting each subprocess


# Base-configuration fields that TASK_LIST entries may override.
CONFIG_KEYS = frozenset(
    {
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
        "B",
        "fs",
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
    }
)


# =========================== Derived-Parameter Utilities ===========================
def _recompute_derived(module: ModuleType) -> None:
    """Recompute all derived values from the current base parameters."""
    module.lamda = module.c / module.fc
    module.d_init = 0.5 * module.lamda
    module.delay_max = (module.N - 1) * module.d_init / module.c

    if isinstance(module.eta_Ratio, (list, tuple)):
        module.eta = [ratio * module.d_init for ratio in module.eta_Ratio]
    else:
        module.eta = module.eta_Ratio * module.d_init


# =========================== Isolated Training Process ===========================
def _isolated_train_process(
    task_overrides: dict[str, Any],
    task_idx: int,
    total_tasks: int,
) -> None:
    """Apply one task configuration and run training in an isolated process."""
    from run_train import run_train

    current_module = sys.modules[__name__]

    for key, value in task_overrides.items():
        setattr(current_module, key, value)

    # Recompute dependent values after parameters such as fc, N, or eta_Ratio change.
    _recompute_derived(current_module)

    print(f" >>> Starting task [{task_idx}/{total_tasks}]")
    print(" >>> Task overrides:")
    for key, value in task_overrides.items():
        print(f"        {key:<16} = {value}")
    print(f" >>> Model type       : {current_module.model_type}")
    print(f" >>> Generate dataset : {current_module.is_creat}")

    run_train(config=current_module)


# =========================== Dataset-Reuse Resolution ===========================
def _resolve_is_creat(task_list: list[dict[str, Any]]) -> list[str]:
    """Resolve is_creat by comparing dataset-related parameters of adjacent tasks."""
    base_module = sys.modules[__name__]

    def signature(task: dict[str, Any]) -> tuple[str, ...]:
        """Build a stable signature for dataset-related task parameters."""
        return tuple(repr(task.get(key, getattr(base_module, key))) for key in DATASET_KEYS)

    resolved: list[str] = []
    previous_signature: Optional[Tuple[str, ...]] = None

    for task in task_list:
        current_signature = signature(task)

        if "is_creat" in task:
            resolved.append(task["is_creat"])
        else:
            resolved.append("False" if current_signature == previous_signature else "True")

        previous_signature = current_signature

    return resolved


# =========================== Task Validation ===========================
def _validate_tasks(task_list: list[dict[str, Any]]) -> None:
    """Validate task types, parameter names, and is_creat values."""
    for task_index, task in enumerate(task_list, start=1):
        if not isinstance(task, dict):
            raise TypeError(f"Task {task_index} must be a dictionary, got {task!r}.")

        unknown_keys = set(task) - CONFIG_KEYS
        if unknown_keys:
            unknown_text = ", ".join(sorted(repr(key) for key in unknown_keys))
            raise KeyError(
                f"Task {task_index} contains unknown parameters: {unknown_text}. "
                "Keys must match names from the base configuration."
            )

        if "is_creat" in task and task["is_creat"] not in {"True", "False"}:
            raise ValueError(
                f"Task {task_index}: is_creat must be "
                f"'True' or 'False', got {task['is_creat']!r}."
            )


# =========================== Entry Point ===========================
if __name__ == "__main__":
    import multiprocessing
    import time

    # Validate all task entries before launching any subprocess.
    _validate_tasks(TASK_LIST)

    total_tasks = len(TASK_LIST)
    is_creat_plan = _resolve_is_creat(TASK_LIST)

    # CUDA subprocesses require spawn for reliable process isolation.
    multiprocessing.set_start_method("spawn", force=True)

    # Print a summary of all scheduled tasks.
    print("=" * 60)
    print(f"Scheduled {total_tasks} task(s) for sequential execution:")
    for task_index, (task, is_creat_value) in enumerate(
        zip(TASK_LIST, is_creat_plan),
        start=1,
    ):
        changed_parameters = (
            ", ".join(f"{key}={value}" for key, value in task.items()) or "(all default values)"
        )
        print(f"   [{task_index:>2}] " f"is_creat={is_creat_value:<5} | " f"{changed_parameters}")
    print("=" * 60)
    print("The task queue is ready. Training will start shortly.")
    time.sleep(START_DELAY_SEC)

    # Run tasks sequentially in isolated subprocesses.
    success_count = 0
    failure_count = 0

    for task_index, task in enumerate(TASK_LIST, start=1):
        # Inject the resolved value without overriding an explicit task setting.
        run_task = dict(task)
        run_task.setdefault(
            "is_creat",
            is_creat_plan[task_index - 1],
        )

        process = multiprocessing.Process(
            target=_isolated_train_process,
            args=(run_task, task_index, total_tasks),
        )
        process.start()
        process.join()

        if process.exitcode != 0:
            failure_count += 1
            print(
                f"\nTask [{task_index}/{total_tasks}] failed "
                f"with exit code {process.exitcode}. "
                "The scheduler will continue with the next task.\n"
            )
        else:
            success_count += 1
            print(
                f"\nTask [{task_index}/{total_tasks}] completed successfully. "
                "The subprocess has exited and its GPU memory can be released.\n"
            )

        if task_index < total_tasks:
            time.sleep(COOLDOWN_SEC)

    # Print the final execution summary.
    print("\n" + "=" * 60)
    print(
        "All tasks finished: "
        f"{success_count} succeeded, "
        f"{failure_count} failed, "
        f"{total_tasks} total."
    )
    print("=" * 60)
