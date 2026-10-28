"""Configuration for source-distance evaluation."""

import sys

import torch


batch_size = 256
checkpoint_path = [
    r"D:\train_result\CDSP-Net\CDSP-Net_ULA_M2_N8_T[200]_Gap0_grid1_temp1e0_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-15_11-14-38\Model_epoch111.pth",
    r"D:\train_result\CDSP-Net\SubspaceNet_ULA_M2_N8_T[200]_Gap0_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-15_12-12-32\Model_epoch11.pth",
    r"D:\train_result\CDSP-Net\DeepCNN_ULA_M2_N8_T[200]_Gap0_grid1_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-15_13-20-21\Model_epoch83.pth",
]
datasets_save_dir = r"D:\experiment\CDSP-Net\datasets"
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
temp = 1
spectrum_step = 0.01
Gap = 0
deta = [0, 5, 10, 15, 20, 25, 30]
success_tol_deg = 1
fix = None
coup_level = [-999]
eta_Ratio = 0.0
eta = eta_Ratio * d_init
num_sample = 20000
is_creat = "False"
test_type = "Srcs_Dist"
test_algorithm = "G-MUSIC"
array_type = "ULA"
band_type = "Narrow"
coh_type = "NonCoherent"
angle_type = "deta_angle"
device = torch.device("cuda")


if __name__ == "__main__":
    from run_sd import run_sd

    current_module = sys.modules[__name__]
    run_sd(config=current_module)
