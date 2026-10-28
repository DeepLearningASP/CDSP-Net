"""DeepCNN model definition."""

import torch
import torch.nn as nn
import math


class DeepCNN(nn.Module):

    def __init__(self, N, grid_size):
        super(DeepCNN, self).__init__()
        self.N = N
        self.grid_size = grid_size

        self.conv1 = nn.Conv2d(3, 256, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(256)

        self.conv2 = nn.Conv2d(256, 256, kernel_size=2, stride=1)
        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(256, 256, kernel_size=2, stride=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 256, kernel_size=2, stride=1)
        self.bn4 = nn.BatchNorm2d(256)

        m1 = (N - 3) // 1 + 1
        m2 = (m1 - 2) // 1 + 1
        m3 = (m2 - 2) // 1 + 1
        m4 = (m3 - 2) // 1 + 1

        self.flatten_dim = 256 * m4 * m4

        self.fc1 = nn.Linear(self.flatten_dim, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc3 = nn.Linear(2048, 1024)
        self.fc4 = nn.Linear(1024, self.grid_size)

        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, Rx):

        R_tau_zero = Rx[:, 0, :, :]

        real_part = R_tau_zero[:, : self.N, :]
        imag_part = R_tau_zero[:, self.N :, :]

        phase_part = torch.atan2(imag_part, real_part)

        x = torch.cat(
            [real_part.unsqueeze(1), imag_part.unsqueeze(1), phase_part.unsqueeze(1)], dim=1
        )

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))

        x = x.view(x.size(0), -1)

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.dropout(self.relu(self.fc3(x)))

        x = self.fc4(x)
        return self.sigmoid(x)
