"""
models.py - 模型定义（MNIST + CIFAR-10 通用）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN_MNIST(nn.Module):
    """用于 MNIST 的简单 CNN（兼容 Opacus：只用 GroupNorm，不用 BatchNorm）"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.gn1   = nn.GroupNorm(4, 16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.gn2   = nn.GroupNorm(8, 32)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(32 * 7 * 7, 128)
        self.fc2   = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.gn1(self.conv1(x))))  # 28→14
        x = self.pool(F.relu(self.gn2(self.conv2(x))))  # 14→7
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class SimpleCNN_CIFAR10(nn.Module):
    """用于 CIFAR-10 的 CNN（Opacus 兼容）"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.gn1   = nn.GroupNorm(8, 32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.gn2   = nn.GroupNorm(16, 64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.gn3   = nn.GroupNorm(32, 128)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(128 * 4 * 4, 256)
        self.fc2   = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.gn1(self.conv1(x))))  # 32→16
        x = self.pool(F.relu(self.gn2(self.conv2(x))))  # 16→8
        x = self.pool(F.relu(self.gn3(self.conv3(x))))  # 8→4
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)
