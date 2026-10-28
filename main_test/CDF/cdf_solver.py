"""CDF evaluation utilities for DOA estimation methods."""

import os
from typing import Any, List, Tuple

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
    "SPS-ESPRIT",
    "SPS-U-ESPRIT",
    "SPS-MUSIC",
    "SPS-G-MUSIC",
)


class CDFTestSolver:
    """Evaluate one algorithm and save its empirical error CDF."""

    def __init__(
        self,
        lamda,
        tau,
        M,
        N,
        T,
        snr,
        fc,
        grid_step,
        temp,
        spectrum_step,
        array_pos,
        cdf_x_deg,
        test_type,
        coh_type,
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
        self.T = T
        self.snr = snr
        self.fc = fc
        self.grid_step = grid_step
        self.temp = temp
        self.spectrum_step = spectrum_step
        self.array_pos = array_pos
        self.cdf_x_deg = cdf_x_deg
        self.test_type = test_type
        self.coh_type = coh_type
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
        """Evaluate the configured algorithm and save the CDF values."""
        if self.test_type != "CDF":
            return

        corr_all, sig_n_all, doa_all, label_all = self.test_dataset
        dataset = TensorDataset(
            corr_all,
            doa_all,
            sig_n_all,
            label_all,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        errors_rad = self.eva_model(
            test_algorithm=self.test_algorithm,
            loader=loader,
        )
        errors_deg = errors_rad * self.R2D
        cdf_x_deg, cdf_y = self.compute_cdf_with_thresholds(
            errors_deg,
            self.cdf_x_deg,
        )

        self.logger.info(
            "=========================== CDF ===========================\n"
        )
        self.logger.info(
            "CDF for %s is: %s\n",
            self.test_algorithm,
            cdf_y,
        )

        result_frame = pd.DataFrame(
            {
                "error_deg": cdf_x_deg,
                f"{self.test_algorithm.upper()}_CDF": cdf_y,
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
        """Create the configured estimator."""
        if test_algorithm in TRADITIONAL_ALGORITHMS:
            return TraditionalMethods(
                self.N,
                self.M,
                self.T,
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
            checkpoint_index = 1
        elif test_algorithm == "CDSP-Net":
            model = CDSP_Net(
                self.M,
                self.N,
                self.lamda,
                self.grid_step,
                self.temp,
            )
            checkpoint_index = 0
        elif test_algorithm == "DeepCNN":
            grid_num = int(120 / self.grid_step + 1)
            model = DeepCNN(
                self.N,
                grid_num,
            )
            checkpoint_index = 2
        else:
            raise ValueError(
                f"Unsupported test algorithm: {test_algorithm}"
            )

        model = load_checkpoint(
            model,
            self.checkpoint_path[checkpoint_index],
        ).to(self.device)
        model.eval()
        return model

    def _estimate_doa(
        self,
        model,
        test_algorithm: str,
        current_corr: torch.Tensor,
        current_sig_n: torch.Tensor,
    ) -> torch.Tensor:
        """Run the estimator for one batch."""
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
    ) -> np.ndarray:
        """Return per-sample estimation errors in radians."""
        model = self._build_model(test_algorithm)
        all_errors: List[torch.Tensor] = []

        with torch.no_grad():
            for (
                current_corr,
                current_doa,
                current_sig_n,
                _,
            ) in loader:

                current_corr = current_corr.to(self.device)
                current_doa = current_doa.to(self.device)
                current_sig_n = current_sig_n.to(self.device)
                doa_est = self._estimate_doa(
                    model,
                    test_algorithm,
                    current_corr,
                    current_sig_n,
                )
                rmspe = self.criterion(
                    doa_est,
                    current_doa,
                )
                all_errors.append(
                    rmspe.detach().cpu()
                )

        if not all_errors:
            return np.empty(
                0,
                dtype=np.float32,
            )

        return torch.cat(
            all_errors,
            dim=0,
        ).numpy()

    def compute_cdf_with_thresholds(
        self,
        errors,
        cdf_x,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the empirical CDF at the specified thresholds."""
        cdf_y = np.array(
            [
                np.mean(errors <= threshold)
                for threshold in cdf_x
            ]
        )
        return cdf_x, cdf_y
