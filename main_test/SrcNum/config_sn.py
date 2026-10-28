"""Configuration for source-count evaluation."""

import sys

import torch


batch_size = 256

# Checkpoints are grouped by model and ordered by M_list.
checkpoint_path = [
    r"D:\train_result\CDSP-Net\CDSP-Net_ULA_M2_N8_T[200]_Gap0_grid1_temp1e0_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_Coherent_2026-10-15_23-32-30\Model_epoch112.pth",
    r"D:\train_result\CDSP-Net\CDSP-Net_ULA_M3_N8_T[200]_Gap0_grid1_temp1e0_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_Coherent_2026-10-16_01-49-19\Model_epoch117.pth",
    r"D:\train_result\CDSP-Net\CDSP-Net_ULA_M4_N8_T[200]_Gap0_grid1_temp1e0_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_Coherent_2026-10-16_02-57-05\Model_epoch87.pth",
    r"D:\train_result\CDSP-Net\CDSP-Net_ULA_M5_N8_T[200]_Gap0_grid1_temp1e0_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_Coherent_2026-10-16_04-13-25\Model_epoch42.pth",
    r"D:\train_result\CDSP-Net\SubspaceNet_ULA_M2_N8_T[200]_Gap0_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_Coherent_2026-10-16_07-28-51\Model_epoch32.pth",
    r"D:\train_result\CDSP-Net\SubspaceNet_ULA_M3_N8_T[200]_Gap0_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_Coherent_2026-10-16_10-06-11\Model_epoch5.pth",
    r"D:\train_result\CDSP-Net\SubspaceNet_ULA_M4_N8_T[200]_Gap0_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_Coherent_2026-10-16_12-44-27\Model_epoch17.pth",
    r"D:\train_result\CDSP-Net\SubspaceNet_ULA_M5_N8_T[200]_Gap0_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_Coherent_2026-10-16_15-23-11\Model_epoch71.pth",
    r"D:\train_result\CDSP-Net\DeepCNN_ULA_M2_N8_T[200]_Gap0_grid1_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_Coherent_2026-10-16_18-02-40\Model_epoch46.pth",
    r"D:\train_result\CDSP-Net\DeepCNN_ULA_M3_N8_T[200]_Gap0_grid1_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_Coherent_2026-10-16_18-56-37\Model_epoch47.pth",
    r"D:\train_result\CDSP-Net\DeepCNN_ULA_M4_N8_T[200]_Gap0_grid1_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_Coherent_2026-10-16_19-46-53\Model_epoch41.pth",
    r"D:\train_result\CDSP-Net\DeepCNN_ULA_M5_N8_T[200]_Gap0_grid1_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_Coherent_2026-10-16_20-22-54\Model_epoch17.pth",
]

datasets_save_dir = r"D:\experiment\CDSP-Net\datasets"
save_dir1 = r"D:\experiment\CDSP-Net\results"
random_seed = 11773
gpu = "0"

M_list = [2, 3, 4, 5]
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
deta = None
fix = None
coup_level = [-999]
eta_Ratio = 0.0
eta = eta_Ratio * d_init
num_sample = 20000
is_creat = "False"
test_type = "Src_Num"
test_algorithm = "DeepCNN"
array_type = "ULA"
band_type = "Narrow"
coh_type = "Coherent"
angle_type = "rand_angle"
device = torch.device("cuda")


if __name__ == "__main__":
    from run_sn import run_sn

    current_module = sys.modules[__name__]
    run_sn(config=current_module)
