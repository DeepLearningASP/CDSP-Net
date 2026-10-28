"""Source-count evaluation utilities."""

import os
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.BaselineAlgorithm import TraditionalMethods
from models.CDSP_Net import CDSP_Net
from models.DeepCNN import DeepCNN
from models.SubspaceNet import SubspaceNet
from tools.criterions import RMSPELoss_Vectorized
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

CHECKPOINT_GROUP_OFFSETS = {
    "CDSP-Net": 0,
    "SubspaceNet": 4,
    "DeepCNN": 8,
}


class SrcNum_TestSolver:
    """Evaluate one estimator over different source counts."""

    def __init__(
        self,
        lamda,
        tau,
        M_or_Mlist,
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
        file_dir_path,
        checkpoint_path,
        batch_size,
        device,
    ):
        self.N = N
        self.lamda = lamda
        self.tau = tau
        self.Ms = (
            M_or_Mlist
            if isinstance(M_or_Mlist, list)
            else [M_or_Mlist]
        )
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
        self.file_dir_path = file_dir_path
        self.checkpoint_path = checkpoint_path
        self.batch_size = batch_size
        self.device = device
        self.R2D = 180.0 / np.pi
        self.criterion = None

    def eva(self) -> None:
        """Evaluate all configured source counts and save the RMSPE values."""
        if self.test_type != "Src_Num":
            return

        rmspe_array = np.zeros(
            len(self.Ms),
            dtype=float,
        )
        corr_all, sig_n_all, doa_all, label_all = self.test_dataset

        self.logger.info(
            "=========================== RMSPE ===========================\n"
        )

        for index, current_sources in enumerate(self.Ms):
            self.criterion = RMSPELoss_Vectorized(
                current_sources,
                reduction="mean",
            )
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
                SNR_idx=0,
                T_idx=0,
                M_idx=index,
                test_algorithm=self.test_algorithm,
                loader=loader,
            )

            self.logger.info(
                "=========================== Source Num %s "
                "===========================\n",
                current_sources,
            )
            self.logger.info(
                "[%s] RMSPE: %.6f rad (%.6f deg)",
                self.test_algorithm,
                rmspe_array[index],
                rmspe_array[index] * self.R2D,
            )

        result_frame = pd.DataFrame(
            {
                "M": self.Ms,
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

    def _checkpoint_index(
        self,
        test_algorithm: str,
        source_index: int,
    ) -> int:
        if test_algorithm not in CHECKPOINT_GROUP_OFFSETS:
            raise ValueError(
                f"No checkpoint mapping is defined for {test_algorithm}."
            )
        return (
            CHECKPOINT_GROUP_OFFSETS[test_algorithm]
            + source_index
        )

    def _build_model(
        self,
        test_algorithm: str,
        current_sources: int,
        source_index: int,
        snapshot_index: int,
    ):
        if test_algorithm in TRADITIONAL_ALGORITHMS:
            return TraditionalMethods(
                self.N,
                current_sources,
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
                current_sources,
                self.fc,
                self.array_pos,
                "ULA",
            )
        elif test_algorithm == "CDSP-Net":
            model = CDSP_Net(
                current_sources,
                self.N,
                self.lamda,
                self.grid_step,
                self.temp,
            )
        elif test_algorithm == "DeepCNN":
            grid_num = int(
                120 / self.grid_step + 1
            )
            model = DeepCNN(
                self.N,
                grid_num,
            )
        else:
            raise ValueError(
                f"Unsupported test algorithm: {test_algorithm}"
            )

        checkpoint_index = self._checkpoint_index(
            test_algorithm,
            source_index,
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
        current_sources: int,
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
                k=current_sources,
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
        M_idx,
        test_algorithm,
        loader,
    ) -> float:
        """Evaluate one source count."""
        source_index = (
            M_idx
            if M_idx is not None
            else 0
        )
        current_sources = self.Ms[source_index]
        model = self._build_model(
            test_algorithm,
            current_sources,
            source_index,
            T_idx,
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
                batch_size = current_corr.shape[0]

                current_corr = current_corr.to(self.device)
                current_doa = current_doa.to(self.device)
                current_sig_n = current_sig_n.to(self.device)
                doa_est = self._estimate_doa(
                    model,
                    test_algorithm,
                    current_corr,
                    current_sig_n,
                    current_sources,
                )
                batch_rmspe = self.criterion(
                    doa_est,
                    current_doa,
                )
                total_rmspe += (
                    float(batch_rmspe)
                    * batch_size
                )
                total_samples += batch_size

        if total_samples == 0:
            raise RuntimeError(
                "The evaluation data loader produced no samples."
            )

        return total_rmspe / total_samples
