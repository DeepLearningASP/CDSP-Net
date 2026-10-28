"""Temperature sensitivity evaluation utilities."""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.CDSP_Net import CDSP_Net
from tools.load_checkpoint import load_checkpoint


class TempTestSolver:
    """Evaluate CDSP-Net across multiple temperature values."""

    def __init__(
        self,
        lamda,
        M,
        N,
        grid_step,
        temp,
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
        self.grid_step = grid_step
        self.temp = temp
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

    def _create_loader(self, index: int) -> DataLoader:
        corr_all, sig_n_all, doa_all, label_all = self.test_dataset
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

    def _build_model(self, temp_idx: int) -> torch.nn.Module:
        model = CDSP_Net(
            self.M,
            self.N,
            self.lamda,
            self.grid_step,
            self.temp[temp_idx],
        )
        model = load_checkpoint(
            model,
            self.checkpoint_path[temp_idx],
        ).to(self.device)
        model.eval()
        return model

    def eva(self) -> None:
        """Evaluate every configured temperature and save the results."""
        rmspe_array = np.zeros(
            len(self.temp),
            dtype=np.float64,
        )
        self.logger.info(
            "=========================== RMSPE ===========================\n"
        )

        for index, current_temp in enumerate(self.temp):
            loader = self._create_loader(index)
            rmspe_array[index] = self.eva_model(
                temp_idx=index,
                test_algorithm=self.test_algorithm,
                loader=loader,
            )
            self.logger.info(
                "=========================== Temperature %s "
                "===========================\n",
                current_temp,
            )
            self.logger.info(
                "RMSPE for %s is %.6f rad (%.6f deg)\n",
                self.test_algorithm,
                rmspe_array[index],
                rmspe_array[index] * self.R2D,
            )

        result_frame = pd.DataFrame(
            {
                "temp": self.temp,
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

    def eva_model(
        self,
        temp_idx,
        test_algorithm,
        loader,
    ) -> float:
        """Evaluate one temperature setting."""
        if test_algorithm != "CDSP-Net":
            raise ValueError(
                f"Unsupported test algorithm: {test_algorithm}"
            )

        model = self._build_model(temp_idx)
        total_rmspe = 0.0
        total_samples = 0

        with torch.no_grad():
            for current_corr, current_doa, _, _ in loader:
                batch_size = current_corr.shape[0]
                current_corr = current_corr.to(self.device)
                current_doa = current_doa.to(self.device)

                doa_est = model(
                    current_corr,
                    self.array_pos,
                )
                batch_rmspe = self.criterion(
                    doa_est,
                    current_doa,
                )
                total_rmspe += (
                    float(batch_rmspe.detach().item())
                    * batch_size
                )
                total_samples += batch_size

        if total_samples == 0:
            raise ValueError(
                "The evaluation loader contains no samples."
            )

        return total_rmspe / total_samples
