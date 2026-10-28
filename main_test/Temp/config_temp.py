"""Configuration for the temperature sensitivity experiment."""

import sys

import torch


batch_size = 256

# Checkpoint order must match the values in temp.
checkpoint_path = [
    r"D:\train_result\CDSP-Net\CDSP-Net_ULA_M2_N8_T[200]_Gap10_grid1_temp1e-3_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-09_16-44-08\Model_epoch114.pth",
    r"D:\train_result\CDSP-Net\CDSP-Net_ULA_M2_N8_T[200]_Gap10_grid1_temp1e-2_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-09_19-31-32\Model_epoch120.pth",
    r"D:\train_result\CDSP-Net\CDSP-Net_ULA_M2_N8_T[200]_Gap10_grid1_temp1e-1_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-09_23-36-09\Model_epoch97.pth",
    r"D:\train_result\CDSP-Net\CDSP-Net_ULA_M2_N8_T[200]_Gap10_grid1_temp1e0_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-10_01-32-43\Model_epoch95.pth",
    r"D:\train_result\CDSP-Net\CDSP-Net_ULA_M2_N8_T[200]_Gap10_grid1_temp1e1_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-10_03-28-14\Model_epoch99.pth",
    r"D:\train_result\CDSP-Net\CDSP-Net_ULA_M2_N8_T[200]_Gap10_grid1_temp1e2_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-15_21-44-33\Model_epoch105.pth",
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
temp = [0.001, 0.01, 0.1, 1, 10, 100]
spectrum_step = 0.01
Gap = 10
deta = None
fix = None
coup_level = [-999]
eta_Ratio = 0.0
eta = eta_Ratio * d_init
num_sample = 20000
is_creat = "True"
test_type = "Temp"
test_algorithm = "CDSP-Net"
array_type = "ULA"
band_type = "Narrow"
coh_type = "NonCoherent"
angle_type = "rand_angle"
device = torch.device("cuda")


if __name__ == "__main__":
    from run_temp import run_temp

    current_module = sys.modules[__name__]
    run_temp(config=current_module)
