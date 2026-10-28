"""Configuration for SNR and snapshot evaluation."""

import sys

import numpy as np
import torch


batch_size = 256

# Few-snapshot checkpoints.
# checkpoint_path = [
#     r"D:\train_result\CDSP-Net\CDSP-Net_ULA_M2_N8_T[2,6,10]_Gap10_grid1_temp1e0_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-15_13-52-54\Model_epoch46.pth",
#     r"D:\train_result\CDSP-Net\SubspaceNet_ULA_M2_N8_T[2,6,10]_Gap10_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-15_14-51-08\Model_epoch85.pth",
#     r"D:\train_result\CDSP-Net\DeepCNN_ULA_M2_N8_T[2,6,10]_Gap10_grid1_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-15_15-59-09\Model_epoch13.pth",
# ]

# # Low-SNR checkpoints.
# checkpoint_path = [
#     r"D:\train_result\CDSP-Net\CDSP-Net_ULA_M2_N8_T[200]_Gap10_grid1_temp1e0_snr-10-5-0_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-15_16-31-39\Model_epoch42.pth",
#     r"D:\train_result\CDSP-Net\SubspaceNet_ULA_M2_N8_T[200]_Gap10_snr-10-5-0_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-15_17-29-51\Model_epoch41.pth",
#     r"D:\train_result\CDSP-Net\DeepCNN_ULA_M2_N8_T[200]_Gap10_grid1_snr-10-5-0_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-15_18-38-06\Model_epoch10.pth",
# ]

# # Coherent-signal checkpoints.
# checkpoint_path = [
#     r"D:\train_result\CDSP-Net\CDSP-Net_ULA_M2_N8_T[10,50,200]_Gap10_grid1_temp1e0_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_Coherent_2026-10-15_19-10-41\Model_epoch112.pth",
#     r"D:\train_result\CDSP-Net\SubspaceNet_ULA_M2_N8_T[10,50,200]_Gap10_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_Coherent_2026-10-15_20-16-45\Model_epoch26.pth",
#     r"D:\train_result\CDSP-Net\DeepCNN_ULA_M2_N8_T[10,50,200]_Gap10_grid1_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_Coherent_2026-10-15_22-49-30\Model_epoch42.pth",
# ]

# Low-SNR coherent-signal checkpoints.
checkpoint_path = [
    r"D:\train_result\CDSP-Net\CDSP-Net_ULA_M2_N8_T[200]_Gap10_grid1_temp1e0_snr-10-5-0_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_Coherent_2026-10-17_09-57-36\Model_epoch43.pth",
    r"D:\train_result\CDSP-Net\SubspaceNet_ULA_M2_N8_T[200]_Gap10_snr-10-5-0_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_Coherent_2026-10-17_11-47-54\Model_epoch45.pth",
    r"D:\train_result\CDSP-Net\DeepCNN_ULA_M2_N8_T[200]_Gap10_grid1_snr-10-5-0_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_Coherent_2026-10-17_13-34-25\Model_epoch12.pth",
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
lamda = c / fc
d_init = 0.5 * lamda
grid_step = 1
temp = 1
spectrum_step = 0.01
Gap = 10
success_tol_deg = 1
deta = None
fix = None
coup_level = [-999]
eta_Ratio = 0.0
eta = eta_Ratio * d_init
num_sample = 20000
is_creat = "False"
test_type = "SNR"
test_algorithm = "ESPRIT"
array_type = "ULA"
band_type = "Narrow"
coh_type = "Coherent"
angle_type = "rand_angle"
device = torch.device("cuda")

if test_type == "SNR":
    T = 200
    snr = np.arange(-10, 1, 2)
    # snr = np.arange(-10, -4, 1)
elif test_type == "Snapshot":
    # T = [2, 4, 6, 8, 10]
    T = [10, 20, 50, 100, 150, 200]
    snr = 10
else:
    raise ValueError(
        f"Unsupported test type: {test_type}"
    )


if __name__ == "__main__":
    from run_ss import run_ss

    current_module = sys.modules[__name__]
    run_ss(config=current_module)
