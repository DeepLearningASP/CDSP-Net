"""Configuration for runtime-complexity evaluation."""

import sys
from pathlib import Path

import torch


batch_size = 256
# Classical methods use float64/complex128; batch size 256 measures throughput.
timing_repeats = 20
warmup_batches = 20  # Untimed inference batches, cycling through the dataset.
# Placeholder timestamps; replace them with the actual training output directories.
# Order: CDSP-Net, SubspaceNet, DeepCNN.
checkpoint_path = [
    r"D:\train_result\CDSP-Net\CDSP-Net_ULA_M2_N8_T[200]_Gap10_grid1_temp1e0_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-19_02-03-17\Model_epoch1.pth",
    r"D:\train_result\CDSP-Net\SubspaceNet_ULA_M2_N8_T[200]_Gap10_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-19_02-04-42\Model_epoch1.pth",
    r"D:\train_result\CDSP-Net\DeepCNN_ULA_M2_N8_T[200]_Gap10_grid1_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-19_02-08-09\Model_epoch1.pth",
]

datasets_save_dir = r"D:\experiment\CDSP-Net\datasets"
save_dir1 = r"D:\experiment\CDSP-Net\results"
random_seed = 11773
gpu = "0"

M = 2
N = 8

# CDSP-Net checkpoints for the source/sensor counts in the paper.
# The additional timestamps below are placeholders as well.
cdsp_checkpoint_paths = {
    (2, 8): checkpoint_path[0],
    (3, 8): r"D:\train_result\CDSP-Net\CDSP-Net_ULA_M3_N8_T[200]_Gap10_grid1_temp1e0_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-21_12-11-36\Model_epoch1.pth",
    (4, 8): r"D:\train_result\CDSP-Net\CDSP-Net_ULA_M4_N8_T[200]_Gap10_grid1_temp1e0_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-21_12-12-23\Model_epoch1.pth",
    (5, 8): r"D:\train_result\CDSP-Net\CDSP-Net_ULA_M5_N8_T[200]_Gap10_grid1_temp1e0_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-21_02-13-08\Model_epoch1.pth",
    (2, 9): r"D:\train_result\CDSP-Net\CDSP-Net_ULA_M2_N9_T[200]_Gap10_grid1_temp1e0_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-21_02-13-42\Model_epoch1.pth",
    (2, 10): r"D:\train_result\CDSP-Net\CDSP-Net_ULA_M2_N10_T[200]_Gap10_grid1_temp1e0_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-21_02-14-16\Model_epoch1.pth",
    (2, 11): r"D:\train_result\CDSP-Net\CDSP-Net_ULA_M2_N11_T[200]_Gap10_grid1_temp1e0_snr10-0-10_CoupLevel[-999]_PosErrRatio[0.0]_Narrow_NonCoherent_2026-10-21_02-14-50\Model_epoch1.pth",
}
checkpoint_path[0] = cdsp_checkpoint_paths[(M, N)]

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
Gap = 10
deta = None
fix = None
coup_level = [-999]
eta_Ratio = 0.0
eta = eta_Ratio * d_init
num_sample = 1000
is_creat = "False"
test_type = "Complexity"
# For size tests, select "CDSP-Net" only; its checkpoint follows M and N.
test_algorithm = [
    "CDSP-Net", "SMV-OMP", "NOMP", "U-ESPRIT", "G-MUSIC", "SubspaceNet", "DeepCNN",
]
array_type = "ULA"
band_type = "Narrow"
coh_type = "NonCoherent"
angle_type = "rand_angle"
device = torch.device("cuda")


if __name__ == "__main__":
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from main_test.Complexity.run_complexity import run_complexity

    current_module = sys.modules[__name__]
    run_complexity(config=current_module)
