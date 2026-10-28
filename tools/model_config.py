"""Model and loss-function configuration."""

from typing import Tuple

import torch
import torch.nn as nn

from models.AblationAlgorithm import (
    CDSP_Net_HardSort,
    CDSP_Net_TopM,
    CDSP_Net_DiffInput,
    CDSP_Net_StaticDict
)
from models.CDSP_Net import CDSP_Net
from models.DeepCNN import DeepCNN
from models.SubspaceNet import SubspaceNet
from tools.criterions import (
    BCE,
    RMSPELoss_Vectorized,
)


def model_config(
    model_type,
    logger,
    N,
    M,
    fc,
    lamda,
    tau,
    array_pos,
    array_type,
    grid_step,
    temp,
    in_type,
    devices,
) -> Tuple[nn.Module, nn.Module]:
    """Create the requested model and its loss function."""

    if model_type == "CDSP-Net":
        model = CDSP_Net(
            M,
            N,
            lamda,
            grid_step,
            temp,
        )
        criterion = RMSPELoss_Vectorized(
            M,
            reduction="mean",
        )
    elif model_type == "CDSP-Net-SD":
        model = CDSP_Net_StaticDict(
            M,
            N,
            lamda,
            grid_step,
            temp,
        )
        criterion = RMSPELoss_Vectorized(
            M,
            reduction="mean",
        )
    elif model_type == "CDSP-Net-HS":
        model = CDSP_Net_HardSort(
            M,
            N,
            lamda,
            grid_step,
        )
        criterion = RMSPELoss_Vectorized(
            M,
            reduction="mean",
        )
    elif model_type == "CDSP-Net-Top-M":
        model = CDSP_Net_TopM(
            M,
            N,
            lamda,
            grid_step,
            temp,
        )
        criterion = RMSPELoss_Vectorized(
            M,
            reduction="mean",
        )
    elif model_type in {
        "CDSP-Net-MCV",
        "CDSP-Net-VSCM",
        "CDSP-Net-ASV",
    }:
        model = CDSP_Net_DiffInput(
            M,
            N,
            lamda,
            grid_step,
            temp,
            in_type,
        )
        criterion = RMSPELoss_Vectorized(
            M,
            reduction="mean",
        )
    elif model_type == "SubspaceNet":
        model = SubspaceNet(
            tau,
            M,
            fc,
            array_pos,
            array_type,
        )
        criterion = RMSPELoss_Vectorized(
            M,
            reduction="mean",
        )
    elif model_type == "DeepCNN":
        grid_num = int(round(120 / grid_step)) + 1
        model = DeepCNN(N, grid_num)
        criterion = BCE()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    device = torch.device(devices)
    model = model.to(device)
    criterion = criterion.to(device)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    logger.info(
        "Model architecture:\n%s",
        model,
    )
    logger.info(
        "Model parameters: %.6fM",
        parameter_count / 1_000_000,
    )

    return model, criterion
