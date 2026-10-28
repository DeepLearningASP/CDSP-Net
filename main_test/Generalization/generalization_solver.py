"""Evaluate generalization with paired observations and per-trial angular errors."""

from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import logging
import math
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import torch

from data_handler.CreateDataset import CorrMtrGenerator, DOAGenerator
from models.CDSP_Net import CDSP_Net
from tools.criterions import RMSPELoss_Vectorized
from tools.logger import load_test_logger


@contextmanager
def random_stream(seed, device):
    """Isolate source, geometry, coupling, and noise random draws."""
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if devices:
            torch.cuda.manual_seed_all(seed)
        yield


class GeneralizationGenerator(CorrMtrGenerator):
    """Generate paired observations using the project's covariance estimator."""

    def __init__(self, scene, seed, batch_index, device):
        array_pos = torch.arange(8, device=device, dtype=torch.float32) * 0.5
        super().__init__(
            8, 2, 8, array_pos, 0.5, 1.0, [scene.eta * 0.5],
            logging.getLogger(__name__), scene.coherence, "Narrow", device,
        )
        self.scene = scene
        self.seed = seed
        self.batch_index = batch_index

    def rng(self, component):
        return random_stream(
            self.seed + component * 1000003 + self.batch_index * 1009, self.device,
        )

    def generate_gaussian_sig_batch(self, batch_size, T):
        with self.rng(1):
            # Generate only the snapshots used by this condition.
            sources = torch.complex(
                torch.randn(batch_size, 2, self.scene.T, device=self.device),
                torch.randn(batch_size, 2, self.scene.T, device=self.device),
            ) / math.sqrt(2)
            gamma = 0.5 + 0.5 * torch.rand(batch_size, 2, 1, device=self.device)
            phi = 2 * torch.pi * torch.rand(batch_size, 2, 1, device=self.device)
            gamma[:, 0] = 1.0
            phi[:, 0] = 0.0
            coefficients = torch.polar(gamma, phi)
        if self.scene.coherence == "NonCoherent":
            signals = sources
        elif self.scene.coherence == "Coherent":
            signals = sources[:, :1] * coefficients
        else:
            raise ValueError(self.scene.coherence)
        signals = signals[:, :, :int(T.max())]  # [batch, sources=2, snapshots]
        return torch.cat([signals.real, signals.imag], dim=1)

    def generate_array_pos(self, batch_size, eta):
        with self.rng(2):
            return super().generate_array_pos(batch_size, eta)

    def generate_coup_matrix(self, positions, alpha):
        with self.rng(3):
            return super().generate_coup_matrix(positions, alpha)

    def generate_array_res_batch(self, signals, doa, snr, alpha, T, eta):
        batch_size = signals.shape[0]
        positions = self.generate_array_pos(batch_size, eta)
        coupling = self.generate_coup_matrix(positions, alpha)
        steering = torch.exp(-2j * torch.pi * positions[:, :, None] * torch.sin(doa[:, None, :]))
        response = coupling @ (steering @ signals)  # [batch, sensors=8, snapshots]
        with self.rng(4):
            gains = (2 * torch.rand(batch_size, 8, 1, device=self.device) - 1) * self.scene.gain_db
            phases = ((2 * torch.rand(batch_size, 8, 1, device=self.device) - 1)
                      * math.radians(self.scene.phase_deg))
        response = response * torch.polar(torch.pow(10.0, gains / 20), phases)
        power = response.abs().square().mean(dim=(1, 2), keepdim=True)
        with self.rng(5):
            noise = torch.complex(
                torch.randn(batch_size, 8, self.scene.T, device=self.device),
                torch.randn(batch_size, 8, self.scene.T, device=self.device),
            ) / math.sqrt(2)
        noise = noise[:, :, :signals.shape[-1]]
        noise = noise * torch.sqrt(power / torch.pow(10.0, snr[:, None, None] / 10))
        return response + noise


def observations(scene, seed, batch_index, count, device):
    generator = GeneralizationGenerator(scene, seed, batch_index, device)
    with generator.rng(0):
        truth = DOAGenerator(2, 10, None, None, "rand_angle", device).generate_doa_batch(count)
    corr, signal = generator.creat_corr_mtr_batch(
        count, truth,
        torch.full((count,), float(scene.snr), device=device),
        torch.full((count,), float(scene.alpha), device=device),
        torch.full((count,), scene.T, device=device, dtype=torch.long),
        torch.full((count,), scene.eta * 0.5, device=device),
    )
    # truth: [batch, 2] radians; corr: [batch, tau=8, 2*N=16, N=8]
    return truth, corr, signal, generator


def metric_arrays(prediction, truth):
    """Original pi-periodic permutation error and sorted maximum error, in degrees."""
    permutations = torch.stack((prediction, prediction.flip(-1)), dim=1)
    residual = torch.remainder(
        permutations - truth[:, None, :] + torch.pi / 2, torch.pi,
    ) - torch.pi / 2
    error = torch.linalg.vector_norm(residual, dim=-1).div(2 ** 0.5).min(dim=1).values
    max_error = (
        prediction.sort(dim=1).values - truth.sort(dim=1).values
    ).abs().max(dim=1).values
    return torch.rad2deg(error), torch.rad2deg(max_error)


class GeneralizationTestSolver:
    """Evaluate all three checkpoints on the same observations in every condition."""

    def __init__(self, config, scenes, result_path):
        self.config = config
        self.scenes = scenes
        self.result_path = Path(result_path)

    def _build_models(self):
        models, metadata = {}, {}
        for label in self.config.model_labels:
            path = Path(self.config.checkpoint_path[label])
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            model = CDSP_Net(2, 8, 1.0, 1, 1).to(self.config.device)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model.eval().requires_grad_(False)
            models[label] = model
            metadata[label] = {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "epoch": int(checkpoint["epoch"]),
            }
        return models, metadata

    def _evaluate_seed(self, scene, seed, models):
        config = self.config
        arrays = {"truth_deg": []}
        for label in models:
            for suffix in ("error_deg", "max_absolute_deg", "prediction_deg"):
                arrays[f"{label}_{suffix}"] = []
        criterion = RMSPELoss_Vectorized(2, reduction="batch").to(config.device)
        with torch.inference_mode():
            for batch_index, begin in enumerate(range(0, config.num_sample, config.batch_size)):
                count = min(config.num_sample - begin, config.batch_size)
                truth, corr, _, generator = observations(
                    scene, seed, batch_index, count, config.device,
                )
                if not torch.isfinite(corr).all():
                    raise ValueError(f"Nonfinite covariance: {scene.key}, seed={seed}")
                arrays["truth_deg"].append(torch.rad2deg(truth).cpu().numpy())
                for label, model in models.items():
                    prediction = model(corr, generator.array_pos)
                    if not torch.isfinite(prediction).all():
                        raise ValueError(f"Nonfinite prediction: {label}, {scene.key}, seed={seed}")
                    errors, maxima = metric_arrays(prediction, truth)
                    if begin == 0:
                        torch.testing.assert_close(
                            torch.deg2rad(errors), criterion(prediction, truth),
                            rtol=2e-5, atol=2e-7,
                        )
                    arrays[f"{label}_error_deg"].append(errors.cpu().numpy())
                    arrays[f"{label}_max_absolute_deg"].append(maxima.cpu().numpy())
                    arrays[f"{label}_prediction_deg"].append(torch.rad2deg(prediction).cpu().numpy())
        result = {key: np.concatenate(values) for key, values in arrays.items()}
        if config.save_raw:
            directory = self.result_path / "raw" / scene.key
            directory.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                directory / f"seed{seed}.npz", **result, seed=np.array(seed),
                scenario_json=np.array(json.dumps(asdict(scene))),
            )
        return result

    def _summarize(self, scene, data, summary, paired):
        meta = asdict(scene)
        for label in self.config.model_labels:
            all_errors = data[f"{label}_error_deg"].astype(float)
            maxima = data[f"{label}_max_absolute_deg"].astype(float)
            predictions = data[f"{label}_prediction_deg"]
            summary.append(dict(
                scenario=scene.key, **meta, model=label, seed=self.config.random_seed,
                n=len(all_errors), rmspe_deg=all_errors.mean(),
                global_rmse_deg=np.sqrt(np.mean(all_errors**2)), success_1deg=(maxima < 1).mean(),
                median_deg=np.median(all_errors), p90_deg=np.quantile(all_errors, 0.9),
                p99_deg=np.quantile(all_errors, 0.99), errors_gt10=int((all_errors > 10).sum()),
                prediction_angles_outside60=int((abs(predictions) > 60).sum()),
            ))
        for baseline, reference in [
            ("NoCoup", "Unified"), ("NoCoup", "UnifiedMixed"), ("Unified", "UnifiedMixed"),
        ]:
            delta = (
                data[f"{baseline}_error_deg"].astype(float)
                - data[f"{reference}_error_deg"].astype(float)
            ).mean()
            paired.append(dict(
                scenario=scene.key, **meta, baseline=baseline, reference=reference,
                delta_baseline_minus_reference_deg=delta,
            ))

    def eva(self):
        models, checkpoints = self._build_models()
        self.result_path.mkdir(parents=True, exist_ok=False)
        logger = load_test_logger(str(self.result_path))
        metadata = {
            "checkpoints": checkpoints, "models": self.config.model_labels,
            "random_seed": self.config.random_seed, "num_sample": self.config.num_sample,
            "batch_size": self.config.batch_size, "generated_snapshots": self.config.T,
            "conditions": [asdict(scene) for scene in self.scenes],
            "python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": np.__version__, "cuda": torch.version.cuda, "device": str(self.config.device),
        }
        (self.result_path / "configuration.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8",
        )
        started = time.perf_counter()
        summary, paired = [], []
        try:
            for index, scene in enumerate(self.scenes, start=1):
                data = self._evaluate_seed(scene, self.config.random_seed, models)
                self._summarize(scene, data, summary, paired)
                message = f"Condition {index}/{len(self.scenes)}: {scene.key}"
                logger.info(message)
                print(message, flush=True)
            tables = self.result_path / "tables"
            tables.mkdir()
            for name, rows in [
                ("summary", summary), ("paired_model_differences", paired),
            ]:
                pd.DataFrame(rows).to_csv(tables / f"{name}.csv", index=False, float_format="%.10g")
            completion = {
                "completed": True, "conditions": len(self.scenes),
                "samples_per_condition": self.config.num_sample,
                "elapsed_seconds": time.perf_counter() - started,
            }
            (self.result_path / "completion.json").write_text(
                json.dumps(completion, indent=2), encoding="utf-8",
            )
            print(f"Saved generalization results: {tables / 'summary.csv'}")
            return tables / "summary.csv"
        finally:
            for handler in list(logger.handlers):
                handler.close()
                logger.removeHandler(handler)

