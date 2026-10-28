"""Position-error evaluation utilities."""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.BaselineAlgorithm import TraditionalMethods
from models.CDSP_Net import CDSP_Net
from models.DeepCNN import DeepCNN
from models.SubspaceNet import SubspaceNet
from tools.load_checkpoint import load_checkpoint


TRADITIONAL_ALGORITHMS = (
    "SMV-OMP",
    "NOMP",
    "MUSIC",
    "G-MUSIC",
    "ESPRIT",
    "U-ESPRIT",
    "SPS-MUSIC",
    "SPS-G-MUSIC",
    "SPS-ESPRIT",
    "SPS-U-ESPRIT",
)


class PosErrTestSolver:
    """Evaluate one estimator across multiple position-error ratios."""

    def __init__(
        self,
        lamda,
        tau,
        M,
        N,
        Ts,
        fc,
        grid_step,
        temp,
        spectrum_step,
        array_pos,
        eta_Ratio,
        test_type,
        test_algorithm,
        test_dataset,
        logger,
        criterion,
        file_dir_path,
        checkpoint_path,
        batch_size,
        device,
    ):
        self.N = N
        self.lamda = lamda
        self.tau = tau
        self.M = M
        self.Ts = Ts
        self.fc = fc
        self.grid_step = grid_step
        self.temp = temp
        self.spectrum_step = spectrum_step
        self.array_pos = array_pos
        self.eta_Ratio = eta_Ratio
        self.test_type = test_type
        self.test_algorithm = test_algorithm
        self.test_dataset = test_dataset
        self.logger = logger
        self.criterion = criterion
        self.file_dir_path = file_dir_path
        self.checkpoint_path = checkpoint_path
        self.batch_size = batch_size
        self.device = device
        self.R2D = 180.0 / np.pi

    def eva(self) -> None:
        """Evaluate all configured position-error ratios and save the results."""
        if self.test_type != "Pos_Err":
            return

        rmspe_array = np.zeros(
            len(self.eta_Ratio),
            dtype=np.float64,
        )
        corr_all, sig_n_all, doa_all, label_all = self.test_dataset

        self.logger.info(
            "=========================== RMSPE ===========================\n"
        )

        for index, eta_ratio in enumerate(self.eta_Ratio):
            dataset = TensorDataset(
                corr_all[index],
                doa_all[index],
                sig_n_all[index],
                label_all[index],
            )
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
            )
            rmspe_array[index] = self.eva_model(
                test_algorithm=self.test_algorithm,
                loader=loader,
            )

            self.logger.info(
                "=========================== Eta_Ratio %s "
                "===========================\n",
                eta_ratio,
            )
            self.logger.info(
                "RMSPE for %s is: %.6f rad (%.6f deg)\n",
                self.test_algorithm,
                rmspe_array[index],
                rmspe_array[index] * self.R2D,
            )

        result_frame = pd.DataFrame(
            {
                "eta_Ratio": self.eta_Ratio,
                f"{self.test_algorithm.upper()}_RMSPE": rmspe_array,
            }
        )
        csv_path = os.path.join(
            self.file_dir_path,
            f"{self.test_algorithm}_{self.test_type}.csv",
        )
        result_frame.to_csv(
            csv_path,
            index=False,
        )

    def _build_model(self, test_algorithm: str):
        """Create the estimator selected by the configuration."""
        if test_algorithm in TRADITIONAL_ALGORITHMS:
            return TraditionalMethods(
                self.N,
                self.M,
                self.Ts,
                self.fc,
                self.grid_step,
                self.spectrum_step,
                self.array_pos,
                self.lamda,
                self.device,
            )

        if test_algorithm == "SubspaceNet":
            model = SubspaceNet(
                self.tau,
                self.M,
                self.fc,
                self.array_pos,
                "ULA",
            )
            model = load_checkpoint(
                model,
                self.checkpoint_path[1],
            ).to(self.device)
            model.eval()
            return model

        if test_algorithm == "CDSP-Net":
            model = CDSP_Net(
                self.M,
                self.N,
                self.lamda,
                self.grid_step,
                self.temp,
            )
            model = load_checkpoint(
                model,
                self.checkpoint_path[0],
            ).to(self.device)
            model.eval()
            return model

        if test_algorithm == "DeepCNN":
            model = DeepCNN(
                self.N,
                int(120 / self.grid_step + 1),
            )
            model = load_checkpoint(
                model,
                self.checkpoint_path[2],
            ).to(self.device)
            model.eval()
            return model

        raise ValueError(
            f"Unsupported test algorithm: {test_algorithm}"
        )

    def _estimate_doa(
        self,
        model,
        test_algorithm: str,
        current_corr: torch.Tensor,
        current_sig_n: torch.Tensor,
    ) -> torch.Tensor:
        """Run one inference batch."""
        if test_algorithm == "SMV-OMP":
            return model.smv_omp(current_corr)
        if test_algorithm == "NOMP":
            return model.nomp(current_sig_n)
        if test_algorithm == "MUSIC":
            doa_est, _ = model.music(current_corr)
            return doa_est
        if test_algorithm == "G-MUSIC":
            doa_est, _ = model.g_music(current_corr)
            return doa_est
        if test_algorithm == "ESPRIT":
            return model.esprit(current_corr)
        if test_algorithm == "U-ESPRIT":
            return model.unitary_esprit(current_corr)
        if test_algorithm == "SPS-MUSIC":
            doa_est, _ = model.sps_music(current_corr)
            return doa_est
        if test_algorithm == "SPS-G-MUSIC":
            doa_est, _ = model.sps_g_music(current_corr)
            return doa_est
        if test_algorithm == "SPS-ESPRIT":
            return model.sps_esprit(current_corr)
        if test_algorithm == "SPS-U-ESPRIT":
            return model.sps_unitary_esprit(current_corr)
        if test_algorithm == "SubspaceNet":
            return model(current_corr)[0]
        if test_algorithm == "CDSP-Net":
            return model(
                current_corr,
                self.array_pos,
            )
        if test_algorithm == "DeepCNN":
            spectrum = model(current_corr)
            _, doa_indices = torch.topk(
                spectrum,
                k=self.M,
                dim=1,
            )
            doa_est_deg = (
                doa_indices.float() * self.grid_step - 60.0
            )
            return doa_est_deg * torch.pi / 180.0

        raise ValueError(
            f"Unsupported test algorithm: {test_algorithm}"
        )

    def eva_model(
        self,
        test_algorithm,
        loader,
    ) -> float:
        """Return the sample-weighted mean RMSPE."""
        model = self._build_model(test_algorithm)
        total_rmspe = 0.0
        total_samples = 0

        with torch.no_grad():
            for (
                current_corr,
                current_doa,
                current_sig_n,
                _,
            ) in loader:
                batch_size = current_corr.shape[0]
                current_corr = current_corr.to(self.device)
                current_doa = current_doa.to(self.device)
                current_sig_n = current_sig_n.to(self.device)

                doa_est = self._estimate_doa(
                    model,
                    test_algorithm,
                    current_corr,
                    current_sig_n,
                )
                batch_rmspe = self.criterion(
                    doa_est,
                    current_doa,
                )
                batch_rmspe_value = (
                    batch_rmspe.item()
                    if isinstance(batch_rmspe, torch.Tensor)
                    else float(batch_rmspe)
                )

                total_rmspe += (
                    batch_rmspe_value * batch_size
                )
                total_samples += batch_size

        if total_samples == 0:
            raise RuntimeError(
                "The evaluation data loader produced no samples."
            )

        return total_rmspe / total_samples
