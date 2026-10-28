"""Grid-resolution evaluation utilities."""

import os
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.AblationAlgorithm import CDSP_Net_StaticDict
from models.BaselineAlgorithm import TraditionalMethods
from models.CDSP_Net import CDSP_Net
from tools.load_checkpoint import load_checkpoint


TRADITIONAL_ALGORITHMS = (
    "SMV-OMP",
    "NOMP",
    "MMV-OMP",
)


class GridTestSolver:
    """Evaluate one method across multiple grid resolutions."""

    def __init__(
        self,
        lamda,
        M,
        N,
        Ts,
        fc,
        grid_step,
        temp,
        spectrum_step,
        array_pos,
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
        self.M = M
        self.Ts = Ts
        self.fc = fc
        self.grid_step = grid_step
        self.temp = temp
        self.spectrum_step = spectrum_step
        self.array_pos = array_pos
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
        """Evaluate every configured grid step and save the RMSPE values."""
        if self.test_type != "Grid":
            return

        rmspe_array = np.zeros(
            len(self.grid_step),
            dtype=np.float64,
        )
        corr_all, sig_n_all, doa_all, label_all = self.test_dataset
        self.logger.info(
            "=========================== RMSPE ===========================\n"
        )

        for grid_index, current_grid_step in enumerate(
            self.grid_step
        ):
            dataset = TensorDataset(
                corr_all[grid_index],
                doa_all[grid_index],
                sig_n_all[grid_index],
                label_all[grid_index],
            )
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
            )
            rmspe_array[grid_index] = self.eva_model(
                grid_idx=grid_index,
                test_algorithm=self.test_algorithm,
                loader=loader,
            )

            self.logger.info(
                "=========================== Grid %s "
                "===========================",
                current_grid_step,
            )
            self.logger.info(
                "RMSPE for %s: %.6f rad (%.6f deg)\n",
                self.test_algorithm,
                rmspe_array[grid_index],
                rmspe_array[grid_index] * self.R2D,
            )

        result_frame = pd.DataFrame(
            {
                "grid": self.grid_step,
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

    def _build_model(
        self,
        grid_idx: int,
        test_algorithm: str,
    ) -> Any:
        """Create the estimator for one grid resolution."""
        current_grid_step = self.grid_step[grid_idx]

        if test_algorithm in TRADITIONAL_ALGORITHMS:
            return TraditionalMethods(
                self.N,
                self.M,
                self.Ts,
                self.fc,
                current_grid_step,
                self.spectrum_step,
                self.array_pos,
                self.lamda,
                self.device,
            )

        if test_algorithm == "CDSP-Net":
            model = CDSP_Net(
                self.M,
                self.N,
                self.lamda,
                current_grid_step,
                self.temp,
            )
            model = load_checkpoint(
                model,
                self.checkpoint_path[grid_idx],
            ).to(self.device)
            model.eval()
            return model

        if test_algorithm == "CDSP-Net-SD":
            model = CDSP_Net_StaticDict(
                self.M,
                self.N,
                self.lamda,
                current_grid_step,
                self.temp,
            )
            model = load_checkpoint(
                model,
                self.checkpoint_path[grid_idx],
            ).to(self.device)
            model.eval()
            return model

        raise ValueError(
            f"Unsupported test algorithm: {test_algorithm}"
        )

    def _run_inference(
        self,
        model: Any,
        test_algorithm: str,
        current_corr: torch.Tensor,
        current_sig_n: torch.Tensor,
    ) -> torch.Tensor:
        """Run one estimator branch."""
        if test_algorithm == "SMV-OMP":
            return model.smv_omp(current_corr)
        if test_algorithm == "NOMP":
            return model.nomp(current_sig_n)
        if test_algorithm == "MMV-OMP":
            return model.mmv_omp(current_sig_n)
        if test_algorithm in ("CDSP-Net", "CDSP-Net-SD"):
            return model(
                current_corr,
                self.array_pos,
            )

        raise ValueError(
            f"Unsupported test algorithm: {test_algorithm}"
        )

    def eva_model(
        self,
        grid_idx,
        test_algorithm,
        loader,
    ) -> float:
        """Evaluate one grid resolution and return its mean RMSPE."""
        model = self._build_model(
            grid_idx,
            test_algorithm,
        )
        total_rmspe = 0.0
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

                doa_est = self._run_inference(
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
                    float(batch_rmspe.detach().cpu())
                    * batch_sample_count
                )
                total_samples += batch_sample_count

        if total_samples == 0:
            raise RuntimeError(
                "The evaluation data loader produced no samples."
            )

        return total_rmspe / total_samples
