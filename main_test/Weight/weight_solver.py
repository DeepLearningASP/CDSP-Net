"""Capture the actual weights produced by the original CDSP-Net forward pass."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.CDSP_Net import CDSP_Net


class WeightRecordingCDSPNet(CDSP_Net):
    """Test-only recorder; model parameters and forward calculations are unchanged."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage_weights = []

    def forward(self, Rx, array_pos):
        self.stage_weights.clear()
        return super().forward(Rx, array_pos)

    def softsort(self, v, temp):
        distribution = super().softsort(v, temp)
        # First SoftSort row is the weight used by the original forward.
        # Keep the original tensor untouched and copy only for recording.
        self.stage_weights.append(distribution[:, 0, :].detach().cpu().clone())
        return distribution


class WeightTestSolver:
    """Run each temperature model on one identical observation and save a CSV."""

    def __init__(self, config, array_pos, observation, result_path):
        self.config = config
        self.array_pos = array_pos
        self.observation = observation
        self.result_path = Path(result_path)

    def _build_model(self, index):
        config = self.config
        model = WeightRecordingCDSPNet(
            config.M, config.N, config.lamda, config.grid_step, config.temp[index],
        )
        # CPU mapping also permits inspecting GPU-trained checkpoints on a CPU.
        checkpoint = torch.load(
            config.checkpoint_path[index], map_location="cpu", weights_only=True,
        )
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError("The checkpoint must contain 'model_state_dict'.")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        # temp is not in state_dict: it is explicitly set by the constructor.
        return model.to(config.device).eval()

    def eva(self):
        rows = []
        for index, temperature in enumerate(self.config.temp):
            model = self._build_model(index)
            with torch.no_grad():
                # The predicted DOAs are computed normally but are not saved.
                model(self.observation, self.array_pos)

            if len(model.stage_weights) != self.config.M:
                raise ValueError(f"Expected {self.config.M} recorded pursuit stages.")
            # [stages=M, sample=1, grid_points=G]
            weights = torch.stack(model.stage_weights).numpy()
            grid = model.theta_grid.detach().cpu().numpy()
            if weights.shape != (self.config.M, 1, len(grid)):
                raise ValueError(f"Unexpected weight shape: {weights.shape}")
            if not np.all(np.isfinite(weights)) or np.any(weights < 0):
                raise ValueError("Recorded weights must be finite and nonnegative.")
            np.testing.assert_allclose(weights.sum(axis=-1), 1, atol=1e-6, rtol=0)

            for stage, distribution in enumerate(weights[:, 0], start=1):
                for angle, weight in zip(grid, distribution):
                    rows.append({
                        "temperature": float(temperature),
                        "stage": stage,
                        "grid_angle_deg": float(angle),
                        "weight": float(weight),
                    })
            print(f"Recorded t_p={temperature:g}: {self.config.M} stages x {len(grid)} weights")
            del model

        # Write once after every model succeeds; only weights and their indices.
        self.result_path.mkdir(parents=True, exist_ok=False)
        csv_path = self.result_path / f"{self.config.test_algorithm}_{self.config.test_type}.csv"
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"Saved weights: {csv_path}")
        return csv_path
