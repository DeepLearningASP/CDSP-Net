"""Configuration for mutual-coupling evaluation."""

import sys

import torch


batch_size = 256

# SNR=10, T=200, noncoherent checkpoints.
# checkpoint_path = [
#     r"D:\train_result\CDSP-Net\CDSP-Net_ULA_M2_N8_T[200]_Gap10_grid1_temp1e0_snr10-0-10_CoupLevel[0,-5,-10,-15,-20,-25,-30]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-17_15-06-40\Model_epoch41.pth",
#     r"D:\train_result\CDSP-Net\SubspaceNet_ULA_M2_N8_T[200]_Gap10_snr10-0-10_CoupLevel[0,-5,-10,-15,-20,-25,-30]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-17_16-09-01\Model_epoch26.pth",
#     r"D:\train_result\CDSP-Net\DeepCNN_ULA_M2_N8_T[200]_Gap10_grid1_snr10-0-10_CoupLevel[0,-5,-10,-15,-20,-25,-30]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-17_17-19-18\Model_epoch26.pth",
# ]

# SNR=-10, T=200, coherent checkpoints.
checkpoint_path = [
    r"D:\train_result\CDSP-Net\CDSP-Net_ULA_M2_N8_T[200]_Gap10_grid1_temp1e0_snr-10-0--10_CoupLevel[0,-5,-10,-15,-20,-25,-30]_PosErrRatio[0.0]_Narrow_Coherent_2026-10-17_17-52-27\Model_epoch42.pth",
    r"D:\train_result\CDSP-Net\SubspaceNet_ULA_M2_N8_T[200]_Gap10_snr-10-0--10_CoupLevel[0,-5,-10,-15,-20,-25,-30]_PosErrRatio[0.0]_Narrow_Coherent_2026-10-17_18-54-16\Model_epoch25.pth",
    r"D:\train_result\CDSP-Net\DeepCNN_ULA_M2_N8_T[200]_Gap10_grid1_snr-10-0--10_CoupLevel[0,-5,-10,-15,-20,-25,-30]_PosErrRatio[0.0]_Narrow_Coherent_2026-10-17_20-52-31\Model_epoch14.pth",
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
snr = -10
lamda = c / fc
d_init = 0.5 * lamda
grid_step = 1
temp = 1
spectrum_step = 0.01
Gap = 10
deta = None
fix = None
coup_level = [0, -5, -10, -15, -20, -25, -30]
eta_Ratio = 0.0
eta = eta_Ratio * d_init
num_sample = 20000
is_creat = "False"
test_type = "Coup_Level"
test_algorithm = "DeepCNN"
array_type = "ULA"
band_type = "Narrow"
coh_type = "Coherent"
angle_type = "rand_angle"
device = torch.device("cuda")


if __name__ == "__main__":
    from run_coup import run_coup

    current_module = sys.modules[__name__]
    run_coup(config=current_module)
