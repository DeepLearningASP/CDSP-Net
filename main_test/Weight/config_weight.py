"""Record two pursuit-stage weight distributions for one fixed scene."""

import sys
from pathlib import Path

import torch


batch_size = 1

# Each path must point to a model trained at the corresponding temperature.
temp = [0.001, 1, 100]
checkpoint_path = [
    r"D:\train_result\CDSP-Net\CDSP-Net_ULA_M2_N8_T[200]_Gap10_grid1_temp1e-3_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-09_16-44-08\Model_epoch116.pth",  # t_p = 0.001
    r"D:\train_result\CDSP-Net\CDSP-Net_ULA_M2_N8_T[200]_Gap10_grid1_temp1e0_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-10_01-32-43\Model_epoch98.pth",  # t_p = 1
    r"D:\train_result\CDSP-Net\CDSP-Net_ULA_M2_N8_T[200]_Gap10_grid1_temp1e2_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-15_21-44-33\Model_epoch101.pth",  # t_p = 100
]

save_dir1 = r"D:\experiment\CDSP-Net\results"
random_seed = 11773
gpu = "0"

M = 2
N = 8
tau = 8
c = 3e8
fc = 3e8
T = 200
snr = 10
lamda = c / fc
d_init = 0.5 * lamda
grid_step = 1
Gap = 10

deta = None
fix = [-20.4, 32.7]  # Degrees; the same observation is used by every model.
coup_level = [-999]
eta_Ratio = 0.0
eta = eta_Ratio * d_init
num_sample = 1

test_type = "Weight"
test_algorithm = "CDSP-Net"
array_type = "ULA"
band_type = "Narrow"
coh_type = "NonCoherent"
angle_type = "fix_angle"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    # Support both `python -m main_test.Weight.config_weight` and direct execution.
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from main_test.Weight.run_weight import run_weight

    run_weight(config=sys.modules[__name__])
