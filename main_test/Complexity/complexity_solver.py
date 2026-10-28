"""Runtime-complexity evaluation utilities."""

import time
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.BaselineAlgorithm import TraditionalMethods
from models.CDSP_Net import CDSP_Net
from models.DeepCNN import DeepCNN
from models.SubspaceNet import SubspaceNet


TRADITIONAL_ALGORITHMS = (
    "SMV-OMP",
    "MMV-OMP",
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


class ComplexitySolver:
    """Measure inference time for one configured estimation method."""

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
        timing_repeats=20,
        warmup_batches=20,
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
        self.timing_repeats = timing_repeats
        self.warmup_batches = warmup_batches
        self.R2D = 180.0 / np.pi

    def eva(self):
        """Run the configured complexity test."""
        if self.test_type != "Complexity":
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

        test_time = self.eva_model(
            test_algorithm=self.test_algorithm,
            loader=loader,
        )

        self.logger.info(
            "=========================== Complexity ===========================\n"
        )
        self.logger.info(
            "Complexity for %s is: %s\n",
            self.test_algorithm,
            test_time,
        )
        Path(self.file_dir_path, "runtime_metadata.json").write_text(
            json.dumps(self.last_timing, indent=2), encoding="utf-8"
        )

    def _build_model(self, test_algorithm):
        """Create and initialize the configured estimator."""
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
                real_dtype=torch.float64,
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

        checkpoint_path = Path(self.checkpoint_path[checkpoint_index])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError("The checkpoint must contain 'model_state_dict'.")
        try:
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        except RuntimeError as error:
            raise ValueError(
                f"{test_algorithm} checkpoint does not match M={self.M}, N={self.N}, "
                f"tau={self.tau}, grid_step={self.grid_step}: {checkpoint_path}"
            ) from error
        self.loaded_checkpoint = {
            "path": str(checkpoint_path),
            "epoch": int(checkpoint["epoch"]) if "epoch" in checkpoint else None,
        }
        self.logger.info("Loaded trained checkpoint: %s", checkpoint_path)
        return model.to(self.device).eval()

    def _run_inference(
        self,
        model,
        test_algorithm,
        current_corr,
        current_sig_n,
    ):
        """Run one inference batch."""

        if test_algorithm == "SMV-OMP":
            return model.smv_omp(current_corr)
        if test_algorithm == "NOMP":
            return model.nomp(current_sig_n)
        if test_algorithm == "MMV-OMP":
            return model.mmv_omp(current_sig_n)
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
            return model(current_corr, self.array_pos)
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

    def _prepare_batch(self, batch, test_algorithm):
        """Exclude transfers/casts from timing equally for every estimator."""
        corr, _, signal, _ = batch
        if test_algorithm in TRADITIONAL_ALGORITHMS:
            return (corr.to(device=self.device, dtype=torch.float64),
                    signal.to(device=self.device, dtype=torch.complex128))
        return corr.to(self.device), signal.to(self.device)

    def eva_model(self, test_algorithm, loader):
        """Synchronized wall time of inference on prepared device inputs.

        All classical methods use float64/complex128, the same outer batch,
        the same warm-up batches and repeated timing procedure.
        Return mean total seconds per full dataset pass for compatibility.
        """
        self.loaded_checkpoint = None
        model = self._build_model(test_algorithm)
        parameter_counts = {}
        if isinstance(model, torch.nn.Module):
            total_parameters = sum(p.numel() for p in model.parameters())
            trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            parameter_counts = {
                "total_parameters": total_parameters,
                "trainable_parameters": trainable_parameters,
            }
            self.logger.info(
                "%s model parameters: total=%d (%.6f M), trainable=%d (%.6f M)\n",
                test_algorithm, total_parameters, total_parameters / 1e6,
                trainable_parameters, trainable_parameters / 1e6,
            )
        use_cuda = torch.device(self.device).type == "cuda"
        warmup_iterations = getattr(self, "warmup_batches", 20)
        if not isinstance(warmup_iterations, int) or warmup_iterations < 1:
            raise ValueError("warmup_batches must be a positive integer.")
        warmup_batches_run = 0
        repeats = getattr(self, "timing_repeats", 20)
        if not isinstance(repeats, int) or repeats < 1:
            raise ValueError("timing_repeats must be a positive integer.")
        pass_times, sample_counts, memory_peaks = [], [], []

        with torch.no_grad():
            if len(loader) == 0:
                raise RuntimeError("The evaluation data loader produced no samples.")
            warmup_iterator = iter(loader)
            for _ in range(warmup_iterations):
                try:
                    batch = next(warmup_iterator)
                except StopIteration:
                    warmup_iterator = iter(loader)
                    batch = next(warmup_iterator)
                warmup_batches_run += 1
                current_corr, current_sig_n = self._prepare_batch(batch, test_algorithm)

                self._run_inference(
                    model,
                    test_algorithm,
                    current_corr,
                    current_sig_n,
                )

            if use_cuda:
                torch.cuda.synchronize(self.device)
            for _ in range(repeats):
                total_time, total_samples, extra_peak = 0.0, 0, 0
                for batch in loader:
                    current_corr, current_sig_n = self._prepare_batch(batch, test_algorithm)
                    if use_cuda:
                        torch.cuda.synchronize(self.device)
                        baseline_memory = torch.cuda.memory_allocated(self.device)
                        torch.cuda.reset_peak_memory_stats(self.device)
                    start_time = time.perf_counter()
                    output = self._run_inference(
                        model, test_algorithm, current_corr, current_sig_n,
                    )
                    if use_cuda:
                        torch.cuda.synchronize(self.device)
                    total_time += time.perf_counter() - start_time
                    total_samples += current_corr.shape[0]
                    if use_cuda:
                        extra_peak = max(extra_peak, torch.cuda.max_memory_allocated(self.device) - baseline_memory)
                    del output
                if total_samples == 0:
                    raise RuntimeError("The evaluation data loader produced no samples.")
                pass_times.append(total_time)
                sample_counts.append(total_samples)
                memory_peaks.append(extra_peak)
        if len(set(sample_counts)) != 1:
            raise RuntimeError("Timing requires a repeatable loader with the same sample count per pass.")
        total_time = float(np.mean(pass_times))
        total_samples = sample_counts[0]
        average_time = total_time / total_samples
        self.last_timing = {
            **parameter_counts,
            "algorithm": test_algorithm, "device": str(self.device),
            "M": self.M, "N": self.N, "T": self.T,
            "checkpoint": self.loaded_checkpoint,
            "hardware": torch.cuda.get_device_name(self.device) if use_cuda else "CPU",
            "torch_version": torch.__version__, "cpu_threads": torch.get_num_threads(),
            "classical_precision": "float64/complex128" if test_algorithm in TRADITIONAL_ALGORITHMS else "checkpoint precision",
            "batch_size": getattr(self, "batch_size", None), "samples_per_pass": total_samples,
            "warmup_batches": warmup_batches_run, "requested_warmup_batches": warmup_iterations,
            "timing_repeats": repeats,
            "total_seconds_each_pass": pass_times, "mean_total_seconds": total_time,
            "mean_ms_per_sample": average_time * 1000,
            "extra_peak_cuda_bytes_each_pass": memory_peaks,
            "memory_metric": "extra PyTorch allocated bytes during inference; excludes the caching allocator reserve and allocations outside PyTorch",
            "scope": "full model forward on prepared inputs; includes CDSP-Net dictionary construction; excludes checkpoint loading, transfers, precision casts, estimator initialization and covariance preprocessing",
            "metric": "amortized batch throughput; batch_size=1 is needed for single-scene latency",
            "kind": "DOA estimator",
        }
        self.logger.info("Timing policy: %s\n", json.dumps(self.last_timing))
        self.logger.info(
            "%s: total_time=%.4fs, samples=%d, avg=%.4f ms/sample\n",
            test_algorithm,
            total_time,
            total_samples,
            average_time * 1000,
        )
        return total_time
