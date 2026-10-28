"""SNR and snapshot evaluation utilities."""

import os
from typing import List, Sequence, Tuple

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
    "MMV-OMP",
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


class SnrSnapshot_TestSolver:
    """Evaluate DOA estimators across SNR or snapshot settings."""

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
        snrs,
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
        self.snrs = snrs
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
        self.success_tol_rad = (
            success_tol_deg * np.pi / 180.0
        )

    def _get_evaluation_axis(
        self,
    ) -> Tuple[Sequence[float], str, str]:
        if self.test_type == "SNR":
            return self.snrs, "SNR", "SNR"

        if self.test_type == "Snapshot":
            return self.Ts, "Snapshot", "T"

        raise ValueError(
            f"Unsupported test type: {self.test_type}"
        )

    def _create_loader(
        self,
        index: int,
    ) -> DataLoader:
        corr_all, sig_n_all, doa_all, label_all = (
            self.test_dataset
        )
        dataset = TensorDataset(
            corr_all[index],
            doa_all[index],
            sig_n_all[index],
            label_all[index],
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def _get_evaluation_indices(
        self,
        index: int,
    ) -> Tuple[int, int]:
        if self.test_type == "SNR":
            return index, 0
        return 0, index

    def _log_result(
        self,
        parameter_name: str,
        parameter_value,
        rmspe: float,
        success_rate: float,
    ) -> None:
        self.logger.info(
            "=========================== %s %s "
            "===========================",
            parameter_name,
            parameter_value,
        )
        self.logger.info(
            "[%s] RMSPE: %.6f rad (%.6f deg)",
            self.test_algorithm,
            rmspe,
            rmspe * self.R2D,
        )

        if not np.isnan(success_rate):
            self.logger.info(
                "[%s] Resolution Probability: %.2f%%\n",
                self.test_algorithm,
                success_rate * 100.0,
            )
        else:
            self.logger.info(
                "[%s] Resolution Probability: "
                "N/A (Theoretical Bound)\n",
                self.test_algorithm,
            )

    def eva(self) -> None:
        """Evaluate all configured test points and save the results."""
        parameter_values, parameter_name, csv_axis = (
            self._get_evaluation_axis()
        )
        rmspe_array = np.zeros(
            len(parameter_values),
        )
        success_rate_array = np.zeros(
            len(parameter_values),
        )

        self.logger.info(
            "=========================== RMSPE & Success Rate "
            "===========================\n"
        )

        for index, parameter_value in enumerate(
            parameter_values
        ):
            loader = self._create_loader(index)
            snr_index, snapshot_index = (
                self._get_evaluation_indices(index)
            )
            rmspe, success_rate = self.eva_model(
                SNR_idx=snr_index,
                T_idx=snapshot_index,
                test_algorithm=self.test_algorithm,
                loader=loader,
            )
            rmspe_array[index] = rmspe
            success_rate_array[index] = success_rate

            self._log_result(
                parameter_name,
                parameter_value,
                rmspe,
                success_rate,
            )

        result_frame = pd.DataFrame(
            {
                csv_axis: parameter_values,
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

    def _build_model(
        self,
        test_algorithm: str,
        snapshot_index: int,
    ):
        if test_algorithm in TRADITIONAL_ALGORITHMS:
            return TraditionalMethods(
                self.N,
                self.M,
                self.Ts[snapshot_index],
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
        if test_algorithm == "SMV-OMP":
            return model.smv_omp(current_corr)
        if test_algorithm == "MMV-OMP":
            return model.mmv_omp(current_sig_n)
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
        if test_algorithm == "DeepCNN":
            spectrum = model(current_corr)
            _, doa_indices = torch.topk(
                spectrum,
                k=self.M,
                dim=1,
            )
            doa_est_deg = (
                doa_indices.float() * self.grid_step
                - 60.0
            )
            return doa_est_deg * torch.pi / 180.0
        if test_algorithm == "CDSP-Net":
            return model(
                current_corr,
                self.array_pos,
            )

        raise ValueError(
            f"Unsupported test algorithm: {test_algorithm}"
        )

    def eva_model(
        self,
        SNR_idx,
        T_idx,
        test_algorithm,
        loader,
    ) -> Tuple[float, float]:
        """Evaluate one algorithm for one test point."""
        model = self._build_model(
            test_algorithm,
            T_idx,
        )
        total_rmspe = 0.0
        total_success_samples = 0
        total_samples = 0

        with torch.no_grad():
            for (
                current_corr,
                current_doa,
                current_sig_n,
                _,
            ) in loader:
                batch_sample_count = current_corr.shape[0]

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
                total_rmspe += (
                    float(batch_rmspe)
                    * batch_sample_count
                )

                sorted_doa_est = torch.sort(
                    doa_est,
                    dim=-1,
                ).values
                sorted_current_doa = torch.sort(
                    current_doa,
                    dim=-1,
                ).values
                absolute_error = torch.abs(
                    sorted_doa_est
                    - sorted_current_doa
                )
                is_success = torch.all(
                    absolute_error
                    < self.success_tol_rad,
                    dim=1,
                )

                total_success_samples += int(
                    is_success.sum().item()
                )
                total_samples += batch_sample_count

        if total_samples == 0:
            raise RuntimeError(
                "The evaluation data loader produced no samples."
            )

        average_rmspe = total_rmspe / total_samples
        average_success_rate = (
            total_success_samples / total_samples
        )
        return (
            average_rmspe,
            average_success_rate,
        )
