"""Training loop utilities for DOA estimation models."""

import datetime
import os
from pathlib import Path
from typing import Any, Tuple
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from tools.cleanup_checkpoints import cleanup_experiment


DOA_ONLY_MODELS = (
    "CDSP-Net",
    "CDSP-Net-SD",
    "CDSP-Net-HS",
    "CDSP-Net-Top-M",
)
SIGNAL_INPUT_MODELS = (
    "CDSP-Net-MCV",
    "CDSP-Net-VSCM",
)



def split_line(info: Any) -> None:
    """Print a timestamped section separator."""
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "=" * 80 + current_time)
    print(f"{info}\n")


class TrainSolver:
    """Train and validate a DOA estimation model."""

    def __init__(
        self,
        num_epochs,
        checkpoint_path,
        resume,
        parallel,
        array_pos,
        train_loader,
        valid_loader,
        model,
        criterion,
        optimizer,
        path,
        logger,
        scheduler,
        model_type,
        device,
    ):
        self.num_epochs = num_epochs
        self.array_pos = array_pos
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.path = path
        self.logger = logger
        self.model_path = checkpoint_path
        self.scheduler = scheduler
        self.model_type = model_type

        self.start_epoch = 1
        self.train_loss = []
        self.val_loss = []

        if parallel:
            self.model = torch.nn.DataParallel(self.model).to(self.device)

        if resume:
            self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        """Restore model, optimizer, scheduler, and loss histories."""
        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=True,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])

        self.start_epoch = checkpoint["epoch"] + 1
        self.train_loss = checkpoint["train_loss"]
        self.val_loss = checkpoint["val_loss"]

    def _compute_loss(
        self,
        corr_batch: torch.Tensor,
        sig_n_batch: torch.Tensor,
        doa_batch: torch.Tensor,
        label_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Run the model-specific forward pass and compute its loss."""
        if self.model_type in DOA_ONLY_MODELS:
            doa_predictions = self.model(
                corr_batch,
                self.array_pos,
            )
            return self.criterion(
                doa_predictions,
                doa_batch,
            )

        if self.model_type in SIGNAL_INPUT_MODELS:
            doa_predictions = self.model(
                corr_batch,
                sig_n_batch,
                self.array_pos,
            )
            return self.criterion(
                doa_predictions,
                doa_batch,
            )

        if self.model_type == "SubspaceNet":
            model_output = self.model(corr_batch)
            doa_predictions = model_output[0]
            return self.criterion(
                doa_predictions,
                doa_batch,
            )

        if self.model_type == "DeepCNN":
            spectrum = self.model(corr_batch)
            return self.criterion(
                spectrum,
                label_batch,
            )

        raise ValueError(f"Unsupported model type: {self.model_type}")

    def _move_batch_to_device(
        self,
        batch: Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Move all tensors in one data-loader batch to the target device."""
        return tuple(tensor.to( self.device, non_blocking=True,) for tensor in batch)

    def _save_checkpoint(self, epoch: int) -> None:
        """Save the complete training state for one epoch."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
        }

        checkpoint_path = os.path.join(
            self.path,
            f"Model_epoch{epoch}.pth",
        )
        print(f"Saving checkpoint to: {checkpoint_path}")
        torch.save(checkpoint, checkpoint_path)

    def _save_loss_history(self) -> None:
        """Save training and validation losses for every model."""
        result_frame = pd.DataFrame(
            {
                f"{self.model_type}_train_loss": self.train_loss,
                f"{self.model_type}_val_loss": self.val_loss,
            }
        )
        csv_path = os.path.join(self.path, f"{self.model_type}_loss.csv")
        result_frame.to_csv(csv_path, index=False)

    def train(self) -> None:
        """Run the training and validation loops."""
        # The upper bound is inclusive so num_epochs represents the final epoch.
        epoch_range = range(
            self.start_epoch,
            self.num_epochs + 1,
        )

        for epoch in epoch_range:
            self.model.train()
            split_line(f"Epoch {epoch} / {self.num_epochs}")

            total_train_loss = 0.0
            total_train_samples = 0

            progress = tqdm(
                enumerate(self.train_loader),
                total=len(self.train_loader),
                desc="Training",
            )

            for step, batch in progress:
                batch_sample_count = batch[0].shape[0]
                (
                    corr_batch,
                    sig_n_batch,
                    doa_batch,
                    label_batch,
                ) = self._move_batch_to_device(batch)

                self.optimizer.zero_grad()
                loss = self._compute_loss(
                    corr_batch,
                    sig_n_batch,
                    doa_batch,
                    label_batch,
                )
                loss.backward()

                # Gradient clipping is only required by SubspaceNet.
                if self.model_type == "SubspaceNet":
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=1.0,
                    )

                self.optimizer.step()

                total_train_samples += batch_sample_count
                total_train_loss += loss.item() * batch_sample_count
                average_train_loss = total_train_loss / total_train_samples
                learning_rate = self.optimizer.param_groups[0]["lr"]

                progress.set_postfix(
                    avg_train_loss=average_train_loss,
                    batch_loss=loss.item(),
                    lr=learning_rate,
                )

                if (step + 1) % 100 == 0:
                    self.logger.info(
                        "Epoch [%d/%d], Step [%d/%d], "
                        "avg_train_loss: %.10f, "
                        "batch_loss: %.10f, lr: %.10f",
                        epoch,
                        self.num_epochs,
                        step + 1,
                        len(self.train_loader),
                        average_train_loss,
                        loss.item(),
                        learning_rate,
                    )

            if total_train_samples == 0:
                raise RuntimeError("The training data loader produced no samples.")

            final_train_loss = total_train_loss / total_train_samples
            self.train_loss.append(final_train_loss)

            self.scheduler.step()

            average_val_loss = self.valid()
            self.val_loss.append(average_val_loss)

            self._save_checkpoint(epoch)

            self.logger.info("Epoch %d Summary", epoch)
            self.logger.info(
                "Training Loss: %.6f",
                final_train_loss,
            )
            self.logger.info(
                "Validation Loss: %.6f",
                average_val_loss,
            )

            if len(self.val_loss) > 1:
                improvement = self.val_loss[-2] - average_val_loss
                trend = "up" if improvement > 0 else "down" if improvement < 0 else "unchanged"
                self.logger.info(
                    "Validation Improvement: %.6f (%s)",
                    improvement,
                    trend,
                )

        if not self.train_loss or not self.val_loss:
            raise RuntimeError("No epochs were executed. Check start_epoch and num_epochs.")

        self._save_loss_history()

        best_val_index = int(np.argmin(self.val_loss))
        best_val_epoch = best_val_index + 1

        self.logger.info("Training Complete")
        self.logger.info(
            "Final Training Loss: %.6f",
            self.train_loss[-1],
        )
        self.logger.info(
            "Final Validation Loss: %.6f",
            self.val_loss[-1],
        )
        self.logger.info(
            "Best Validation Loss: %.6f at epoch %d",
            self.val_loss[best_val_index],
            best_val_epoch,
        )

        # Read the completed log and retain only this run's best checkpoint.
        # Keep cleanup messages on stdout so the best result remains the last log line.
        try:
            for handler in self.logger.handlers:
                handler.flush()
            cleanup_experiment(Path(self.path) / "log.txt", apply=True)
        except (OSError, ValueError) as error:
            print(f"Checkpoint cleanup skipped or failed: {error}")

    def valid(self) -> float:
        """Evaluate the model on the validation data loader."""
        self.model.eval()

        total_val_loss = 0.0
        total_val_samples = 0
        last_batch_loss = None

        progress = tqdm(
            enumerate(self.valid_loader),
            total=len(self.valid_loader),
            desc="Validation",
        )

        with torch.no_grad():
            for _, batch in progress:
                batch_sample_count = batch[0].shape[0]
                (
                    corr_batch,
                    sig_n_batch,
                    doa_batch,
                    label_batch,
                ) = self._move_batch_to_device(batch)

                loss = self._compute_loss(
                    corr_batch,
                    sig_n_batch,
                    doa_batch,
                    label_batch,
                )

                last_batch_loss = loss.item()
                total_val_loss += last_batch_loss * batch_sample_count
                total_val_samples += batch_sample_count
                average_val_loss = total_val_loss / total_val_samples

                progress.set_postfix(
                    avg_val_loss=average_val_loss,
                    batch_loss=last_batch_loss,
                )

        if total_val_samples == 0 or last_batch_loss is None:
            raise RuntimeError("The validation data loader produced no samples.")

        average_val_loss = total_val_loss / total_val_samples
        self.logger.info(
            "Validation: avg_val_loss: %.10f, batch_loss: %.10f",
            average_val_loss,
            last_batch_loss,
        )

        self.model.train()
        return average_val_loss
