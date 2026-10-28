"""Configuration for the position-error evaluation."""

import sys
import torch

batch_size = 256

# Ideal-condition checkpoints.
# checkpoint_path = [
#     r"D:\train_result\CDSP-Net\CDSP-Net_ULA_M2_N8_T[200]_Gap10_grid1_temp1e0_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0,0.05,0.1,0.15,0.2,0.25]_Narrow_NonCoherent_2026-10-17_21-53-01\Model_epoch104.pth",
#     r"D:\train_result\CDSP-Net\SubspaceNet_ULA_M2_N8_T[200]_Gap10_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0,0.05,0.1,0.15,0.2,0.25]_Narrow_NonCoherent_2026-10-18_00-13-24\Model_epoch40.pth",
#     r"D:\train_result\CDSP-Net\DeepCNN_ULA_M2_N8_T[200]_Gap10_grid1_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0,0.05,0.1,0.15,0.2,0.25]_Narrow_NonCoherent_2026-10-18_01-24-06\Model_epoch26.pth",
# ]

# Challenging-condition checkpoints.
checkpoint_path = [
    r"D:\train_result\CDSP-Net\CDSP-Net_ULA_M2_N8_T[2]_Gap10_grid1_temp1e0_snr10-0-10_CoupLevel[-10]_PosErrRatio[0.0,0.05,0.1,0.15,0.2,0.25]_Narrow_Coherent_2026-10-18_01-57-09\Model_epoch47.pth",
    r"D:\train_result\CDSP-Net\SubspaceNet_ULA_M2_N8_T[2]_Gap10_snr10-0-10_CoupLevel[-10]_PosErrRatio[0.0,0.05,0.1,0.15,0.2,0.25]_Narrow_Coherent_2026-10-18_02-58-29\Model_epoch13.pth",
    r"D:\train_result\CDSP-Net\DeepCNN_ULA_M2_N8_T[2]_Gap10_grid1_snr10-0-10_CoupLevel[-10]_PosErrRatio[0.0,0.05,0.1,0.15,0.2,0.25]_Narrow_Coherent_2026-10-18_04-08-39\Model_epoch11.pth",
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
T = 2
snr = 10
lamda = c / fc
d_init = 0.5 * lamda
grid_step = 1
temp = 1
spectrum_step = 0.01
Gap = 10
deta = None
fix = None
coup_level = [-10]
eta_Ratio = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
eta = [
    ratio * d_init
    for ratio in eta_Ratio
]
num_sample = 20000
is_creat = "False"
test_type = "Pos_Err"
test_algorithm = "DeepCNN"
array_type = "ULA"
band_type = "Narrow"
coh_type = "Coherent"
angle_type = "rand_angle"
device = torch.device("cuda")


if __name__ == "__main__":
    from run_pe import run_pe

    current_module = sys.modules[__name__]
    run_pe(config=current_module)
