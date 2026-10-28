"""Ablation-study evaluation utilities."""

import os
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.AblationAlgorithm import (
    CDSP_Net_HardSort,
    CDSP_Net_StaticDict,
    CDSP_Net_TopM,
)
from models.CDSP_Net import CDSP_Net
from tools.load_checkpoint import load_checkpoint


class AblationTestSolver:
    """Evaluate trained model variants on ablation datasets."""

    def __init__(
        self,
        lamda,
        M,
        N,
        Ts,
        grid_step,
        temp,
        array_pos,
        snrs,
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
        self.grid_step = grid_step
        self.temp = temp
        self.array_pos = array_pos
        self.snrs = snrs
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

        self._model_cache: Dict[str, torch.nn.Module] = {}

    def _get_evaluation_plan(self):
        if self.test_type == "Ablation_SNR":
            return self.snrs, "SNR", "SNR"

        if self.test_type == "Ablation_T":
            return self.Ts, "T", "Snapshot"

        raise ValueError(f"Unsupported ablation test type: {self.test_type}")

    def _build_model(self, test_algorithm):
        if test_algorithm == "CDSP-Net":
            model = CDSP_Net(
                self.M,
                self.N,
                self.lamda,
                self.grid_step,
                self.temp,
            )
            checkpoint_index = 0
        elif test_algorithm == "CDSP-Net-SD":
            model = CDSP_Net_StaticDict(
                self.M,
                self.N,
                self.lamda,
                self.grid_step,
                self.temp,
            )
            checkpoint_index = 1
        elif test_algorithm == "CDSP-Net-HS":
            model = CDSP_Net_HardSort(
                self.M,
                self.N,
                self.lamda,
                self.grid_step,
            )
            checkpoint_index = 2
        elif test_algorithm == "CDSP-Net-Top-M":
            model = CDSP_Net_TopM(
                self.M,
                self.N,
                self.lamda,
                self.grid_step,
                self.temp,
            )
            checkpoint_index = 3
        else:
            raise ValueError(f"Unsupported test algorithm: {test_algorithm}")

        try:
            checkpoint = self.checkpoint_path[checkpoint_index]
        except IndexError as error:
            raise IndexError(f"Missing checkpoint path for {test_algorithm}.") from error

        model = load_checkpoint(
            model,
            checkpoint,
        ).to(self.device)
        model.eval()
        return model

    def _get_model(self, test_algorithm):
        if test_algorithm not in self._model_cache:
            self._model_cache[test_algorithm] = self._build_model(test_algorithm)
        return self._model_cache[test_algorithm]

    def eva(self):
        parameter_values, column_name, parameter_label = self._get_evaluation_plan()
        rmspe_array = np.zeros(
            len(parameter_values),
            dtype=np.float64,
        )

        corr_all, sig_n_all, doa_all, label_all = self.test_dataset
        self.logger.info("=========================== RMSPE " "===========================")

        for index, current_parameter in enumerate(parameter_values):
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
                "=========================== %s %s " "===========================",
                parameter_label,
                current_parameter,
            )
            self.logger.info(
                "RMSPE for %s: %.6f rad (%.6f deg)",
                self.test_algorithm,
                rmspe_array[index],
                rmspe_array[index] * self.R2D,
            )

        result_frame = pd.DataFrame(
            {
                column_name: parameter_values,
                f"{self.test_algorithm.upper()}_RMSPE": (rmspe_array),
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

    def eva_model(self, test_algorithm, loader):
        model = self._get_model(test_algorithm)

        total_rmspe = 0.0
        total_samples = 0

        with torch.no_grad():
            for (
                current_corr,
                current_doa,
                _,
                _,
            ) in loader:
                batch_size = current_corr.shape[0]
                current_corr = current_corr.to(self.device)
                current_doa = current_doa.to(self.device)

                doa_estimation = model(
                    current_corr,
                    self.array_pos,
                )
                batch_rmspe = self.criterion(
                    doa_estimation,
                    current_doa,
                )

                total_rmspe += float(batch_rmspe.item()) * batch_size
                total_samples += batch_size

        if total_samples == 0:
            raise RuntimeError("The evaluation data loader produced no samples.")

        return total_rmspe / total_samples
