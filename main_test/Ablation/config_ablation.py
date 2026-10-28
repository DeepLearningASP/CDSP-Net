"""Configuration for ablation-study evaluation."""

import sys

import numpy as np
import torch


# =========================== Evaluation Configuration ===========================
batch_size = 256
checkpoint_path = [
    r"D:\train_result\CDSP-Net\CDSP-Net_ULA_M2_N8_T[200]_Gap10_grid1_temp1e0_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-10_01-32-43\Model_epoch95.pth",
    r"D:\train_result\CDSP-Net\CDSP-Net-SD_ULA_M2_N8_T[200]_Gap10_grid1_temp1e0_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-19_01-36-07\Model_epoch101.pth",
    r"D:\train_result\CDSP-Net\CDSP-Net-HS_ULA_M2_N8_T[200]_Gap10_grid1_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-19_02-17-45\Model_epoch104.pth",
    r"D:\train_result\CDSP-Net\CDSP-Net-Top-M_ULA_M2_N8_T[200]_Gap10_grid1_temp1e0_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-19_00-38-43\Model_epoch110.pth",
]
datasets_save_dir = r"D:\experiment\CDSP-Net\datasets"
save_dir1 = r"D:\experiment\CDSP-Net\results"
random_seed = 11773
gpu = "0"


# =========================== Array and Signal Configuration ===========================
M = 2
N = 8
tau = 8
c = 3e8
fc = 3e8
lamda = c / fc
d_init = 0.5 * lamda
grid_step = 1
temp = 1
spectrum_step = 0.01
Gap = 10
deta = None
fix = None
coup_level = [-999]
eta_Ratio = 0.0
eta = eta_Ratio * d_init
num_sample = 20000
is_creat = "False"
test_type = "Ablation_SNR"
test_algorithm = "CDSP-Net-Top-M"
array_type = "ULA"
band_type = "Narrow"
coh_type = "NonCoherent"
angle_type = "rand_angle"
device = torch.device("cuda")


if test_type == "Ablation_SNR":
    T = 200
    snr = np.arange(6, 15, 2)
elif test_type == "Ablation_T":
    T = [10, 20, 50, 100, 200, 500]
    snr = 10
else:
    raise ValueError(f"Unsupported ablation test type: {test_type}")


if __name__ == "__main__":
    from run_ablation import run_ablation

    current_module = sys.modules[__name__]
    run_ablation(config=current_module)
