"""Configuration for CDF evaluation."""

import sys

import numpy as np
import torch


batch_size = 256
checkpoint_path = [
    r"D:\train_result\CDSP-Net\CDSP-Net_ULA_M2_N8_T[2]_Gap10_grid1_temp1e0_snr10-0-10_CoupLevel[-10]_PosErrRatio[0.15]_Narrow_Coherent_2026-10-18_04-41-50\Model_epoch44.pth",
    r"D:\train_result\CDSP-Net\SubspaceNet_ULA_M2_N8_T[2]_Gap10_snr10-0-10_CoupLevel[-10]_PosErrRatio[0.15]_Narrow_Coherent_2026-10-18_05-43-13\Model_epoch10.pth",
    r"D:\train_result\CDSP-Net\DeepCNN_ULA_M2_N8_T[2]_Gap10_grid1_snr10-0-10_CoupLevel[-10]_PosErrRatio[0.15]_Narrow_Coherent_2026-10-18_06-53-08\Model_epoch12.pth",
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
fix = [-20.381, 57.5]
coup_level = [-10]
eta_Ratio = 0.15
eta = eta_Ratio * d_init
num_sample = 50000
is_creat = "False"
test_type = "CDF"
test_algorithm = "DeepCNN"
array_type = "ULA"
band_type = "Narrow"
coh_type = "Coherent"
angle_type = "rand_angle"
cdf_x_deg = np.linspace(0, 2, 200)
device = torch.device("cuda")


if __name__ == "__main__":
    from run_cdf import run_cdf

    current_module = sys.modules[__name__]
    run_cdf(config=current_module)
