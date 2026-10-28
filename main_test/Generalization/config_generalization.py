"""Configure the three generalization experiments reported in the paper."""

import sys
from dataclasses import dataclass
from pathlib import Path

import torch


batch_size = 256
random_seed = 11773
num_sample = 20000
gpu = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_labels = ["UnifiedMixed", "NoCoup", "Unified"]
checkpoint_path = {
    "UnifiedMixed": r"D:\train_result\CDSP-Net\CDSP-Net_ULA_M2_N8_T[10,50,200]_Gap10_grid1_temp1e0_snr-5-5-10_CoupLevel[-999,-30,-20,-10]_PosErrRatio[0.0,0.05,0.1,0.15]_Narrow_Mixed_2026-10-22_03-07-42\Model_epoch43.pth",
    "NoCoup": r"D:\train_result\CDSP-Net\CDSP-Net_ULA_M2_N8_T[10,50,200]_Gap10_grid1_temp1e0_snr-5-5-10_CoupLevel[-999]_PosErrRatio[0.0,0.05,0.1,0.15]_Narrow_NonCoherent_2026-10-21_02-42-30\Model_epoch41.pth",
    "Unified": r"D:\train_result\CDSP-Net\CDSP-Net_ULA_M2_N8_T[10,50,200]_Gap10_grid1_temp1e0_snr-5-5-10_CoupLevel[-999,-30,-20,-10]_PosErrRatio[0.0,0.05,0.1,0.15]_Narrow_NonCoherent_2026-10-21_01-42-11\Model_epoch50.pth",
}
save_dir1 = r"D:\experiment\CDSP-Net\results"
save_raw = True

M = 2
N = 8
tau = 8
lamda = 1.0
d_init = 0.5
grid_step = 1
temp = 1
Gap = 10
T = 200
test_type = "Generalization"
test_algorithm = "CDSP-Net"

snrs = [-5, 0, 5, 10]
coup_levels = [-30, -25, -20, -15, -10]
eta_ratios = [i / 40 for i in range(13)]
coherence_coupling_pairs = [
    ("NonCoherent", -999), ("NonCoherent", -10),
    ("Coherent", -999), ("Coherent", -10),
]


@dataclass(frozen=True)
class Scenario:
    """Physical test conditions; alpha=-999 disables mutual coupling."""

    alpha: float = -999
    eta: float = 0.0
    T: int = 200
    snr: float = 10.0
    coherence: str = "NonCoherent"
    gain_db: float = 0.0
    phase_deg: float = 0.0
    noise_rho: float = 0.0
    source_rho: float = -1.0

    @property
    def key(self):
        return (
            f"a{self.alpha:g}_e{self.eta:g}_T{self.T}_s{self.snr:g}_{self.coherence}"
            f"_g{self.gain_db:g}_p{self.phase_deg:g}"
            f"_n{self.noise_rho:g}_r{self.source_rho:g}"
        )


def panel_scenarios():
    """Return the conditions for each panel, preserving the evaluation order."""
    return {
        "a": [
            Scenario(alpha=alpha, eta=0.1, T=T, snr=snr, coherence=coherence)
            for coherence, alpha in coherence_coupling_pairs for snr in snrs
        ],
        "b": [Scenario(alpha=alpha, eta=0.1, T=T) for alpha in coup_levels],
        "c": [
            Scenario(alpha=alpha, eta=eta, T=T)
            for alpha in [-999, -20] for eta in eta_ratios
        ],
    }


def scenarios():
    """Deduplicate shared conditions across panels: 44 with the paper defaults."""
    return list({
        scene.key: scene
        for block in panel_scenarios().values() for scene in block
    }.values())


if __name__ == "__main__":
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from main_test.Generalization.run_generalization import run_generalization

    run_generalization(config=sys.modules[__name__])

