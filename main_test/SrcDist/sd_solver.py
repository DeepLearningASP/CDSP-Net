"""Source-distance evaluation utilities."""

import os
from typing import Tuple

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
)


class SrcDistTestSolver:
    """Evaluate DOA estimation performance across source separations."""

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
        deta,
        test_type,
        test_algorithm,
        test_dataset,
        logger,
        criterion,
        file_dir_path,
        checkpoint_path,
        batch_size,
        device,
        success_tol_deg,
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
        self.deta = deta
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
        self.success_tol_rad = success_tol_deg * np.pi / 180.0

    def _create_loader(
        self,
        corr,
        doa,
        signal,
        label,
    ) -> DataLoader:
        dataset = TensorDataset(
            corr,
            doa,
            signal,
            label,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def _build_model(self, test_algorithm):
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
            grid_num = int(
                120 / self.grid_step + 1
            )
            model = DeepCNN(
                self.N,
                grid_num,
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
        test_algorithm,
        current_corr,
        current_sig_n,
    ):
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

    def _compute_batch_metrics(
        self,
        doa_est,
        current_doa,
    ) -> Tuple[float, int]:
        batch_rmspe = self.criterion(
            doa_est,
            current_doa,
        )
        sorted_doa_est, _ = torch.sort(
            doa_est,
            dim=-1,
        )
        sorted_current_doa, _ = torch.sort(
            current_doa,
            dim=-1,
        )
        absolute_error = torch.abs(
            sorted_doa_est - sorted_current_doa
        )
        successful_samples = torch.all(
            absolute_error < self.success_tol_rad,
            dim=1,
        )
        return (
            batch_rmspe.item(),
            int(successful_samples.sum().item()),
        )

    def eva(self) -> None:
        """Evaluate every configured source separation and save the results."""
        rmspe_array = np.zeros(
            len(self.deta),
        )
        success_rate_array = np.zeros(
            len(self.deta),
        )
        (
            corr_all,
            sig_n_all,
            doa_all,
            label_all,
        ) = self.test_dataset

        self.logger.info(
            "=========================== "
            "RMSPE & Success Rate "
            "===========================\n"
        )

        for index, current_param in enumerate(
            self.deta
        ):
            loader = self._create_loader(
                corr_all[index],
                doa_all[index],
                sig_n_all[index],
                label_all[index],
            )
            (
                rmspe_array[index],
                success_rate_array[index],
            ) = self.eva_model(
                test_algorithm=self.test_algorithm,
                loader=loader,
            )

            self.logger.info(
                "=========================== "
                "SrcDist %srad "
                "===========================\n",
                current_param,
            )
            self.logger.info(
                "[%s] RMSPE: %.6frad (%.6f deg)",
                self.test_algorithm,
                rmspe_array[index],
                rmspe_array[index] * self.R2D,
            )
            self.logger.info(
                "[%s] Resolution Prob(Success Rate): %.2f%%\n",
                self.test_algorithm,
                success_rate_array[index] * 100,
            )

        result_frame = pd.DataFrame(
            {
                "deta": self.deta,
                f"{self.test_algorithm.upper()}_RMSPE": rmspe_array,
                f"{self.test_algorithm.upper()}_Success_Rate": (
                    success_rate_array
                ),
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

    def eva_model(
        self,
        test_algorithm,
        loader,
    ) -> Tuple[float, float]:
        """Evaluate one model over one source-separation dataset."""
        total_rmspe = 0.0
        total_success_samples = 0
        total_samples = 0
        model = self._build_model(
            test_algorithm
        )

        with torch.no_grad():
            for (
                current_corr,
                current_doa,
                current_sig_n,
                _,
            ) in loader:
                batch_sample_num = current_corr.shape[0]
                current_corr = current_corr.to(
                    self.device
                )
                current_doa = current_doa.to(
                    self.device
                )
                current_sig_n = current_sig_n.to(
                    self.device
                )

                doa_est = self._estimate_doa(
                    model,
                    test_algorithm,
                    current_corr,
                    current_sig_n,
                )
                (
                    batch_rmspe,
                    successful_samples,
                ) = self._compute_batch_metrics(
                    doa_est,
                    current_doa,
                )

                total_rmspe += (
                    batch_rmspe * batch_sample_num
                )
                total_success_samples += (
                    successful_samples
                )
                total_samples += batch_sample_num

        average_rmspe = (
            total_rmspe / total_samples
        )
        average_success_rate = (
            total_success_samples / total_samples
        )
        return (
            average_rmspe,
            average_success_rate,
        )
