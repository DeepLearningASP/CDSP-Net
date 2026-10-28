"""Configuration for grid-resolution evaluation."""

import sys

import torch


batch_size = 64
checkpoint_path = [
    # CDSP-Net
    # r"D:\train_result\CDSP-Net\CDSP-Net_ULA_M2_N8_T[200]_Gap10_grid0.1_temp1e0_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-14_03-42-25\Model_epoch112.pth",
    # r"D:\train_result\CDSP-Net\CDSP-Net_ULA_M2_N8_T[200]_Gap10_grid1_temp1e0_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-14_00-31-38\Model_epoch99.pth",
    # r"D:\train_result\CDSP-Net\CDSP-Net_ULA_M2_N8_T[200]_Gap10_grid2_temp1e0_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-14_10-29-54\Model_epoch102.pth",
    # r"D:\train_result\CDSP-Net\CDSP-Net_ULA_M2_N8_T[200]_Gap10_grid5_temp1e0_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-14_13-01-06\Model_epoch86.pth",
    # r"D:\train_result\CDSP-Net\CDSP-Net_ULA_M2_N8_T[200]_Gap10_grid10_temp1e0_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-14_15-33-31\Model_epoch117.pth",
    # r"D:\train_result\CDSP-Net\CDSP-Net_ULA_M2_N8_T[200]_Gap10_grid20_temp1e0_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-16_20-59-01\Model_epoch96.pth",
    # r"D:\train_result\CDSP-Net\CDSP-Net_ULA_M2_N8_T[200]_Gap10_grid30_temp1e0_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-16_22-12-47\Model_epoch97.pth",
    # CDSP-Net-SD
    r"D:\train_result\CDSP-Net\CDSP-Net-SD_ULA_M2_N8_T[200]_Gap10_grid0.1_temp1e0_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-14_18-05-17\Model_epoch115.pth",
    r"D:\train_result\CDSP-Net\CDSP-Net-SD_ULA_M2_N8_T[200]_Gap10_grid1_temp1e0_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-14_02-17-30\Model_epoch116.pth",
    r"D:\train_result\CDSP-Net\CDSP-Net-SD_ULA_M2_N8_T[200]_Gap10_grid2_temp1e0_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-14_21-16-11\Model_epoch119.pth",
    r"D:\train_result\CDSP-Net\CDSP-Net-SD_ULA_M2_N8_T[200]_Gap10_grid5_temp1e0_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-14_23-00-09\Model_epoch117.pth",
    r"D:\train_result\CDSP-Net\CDSP-Net-SD_ULA_M2_N8_T[200]_Gap10_grid10_temp1e0_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-14_23-42-05\Model_epoch119.pth",
    r"D:\train_result\CDSP-Net\CDSP-Net-SD_ULA_M2_N8_T[200]_Gap10_grid20_temp1e0_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-16_23-18-10\Model_epoch113.pth",
    r"D:\train_result\CDSP-Net\CDSP-Net-SD_ULA_M2_N8_T[200]_Gap10_grid30_temp1e0_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-16_23-52-29\Model_epoch113.pth",
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
grid_step = [0.1, 1, 2, 5, 10, 20, 30]
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
test_type = "Grid"
test_algorithm = "MMV-OMP"
array_type = "ULA"
band_type = "Narrow"
coh_type = "NonCoherent"
angle_type = "rand_angle"
device = torch.device("cuda")


if __name__ == "__main__":
    from run_grid import run_grid

    current_module = sys.modules[__name__]
    run_grid(config=current_module)
