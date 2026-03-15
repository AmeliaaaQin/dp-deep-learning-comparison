"""
models_dp.py - 三种深度学习模型定义（全部兼容 Opacus）

模型：
  1. MLP       - 多层感知机（最简单）
  2. SimpleCNN - 简单卷积网络
  3. LeNet5    - 经典 LeNet-5 变体

注意：Opacus 不支持 BatchNorm，统一使用 GroupNorm 替代
"""

import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════
# 1. MLP（多层感知机）
# ══════════════════════════════════════════════
class MLP(nn.Module):
    """
    3层全连接网络
    输入：28x28=784 → 512 → 256 → 10
    """
    def __init__(self, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════
# 2. SimpleCNN（简单卷积网络）
# ══════════════════════════════════════════════
class SimpleCNN(nn.Module):
    """
    2层卷积 + 2层全连接
    使用 GroupNorm 替代 BatchNorm（Opacus 兼容）
    """
    def __init__(self, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.gn1   = nn.GroupNorm(8, 32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.gn2   = nn.GroupNorm(16, 64)
        self.pool  = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        self.fc1   = nn.Linear(64 * 7 * 7, 256)
        self.fc2   = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.gn1(self.conv1(x))))  # 28→14
        x = self.pool(F.relu(self.gn2(self.conv2(x))))  # 14→7
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


# ══════════════════════════════════════════════
# 3. LeNet-5（经典网络，Opacus 兼容版）
# ══════════════════════════════════════════════
class LeNet5(nn.Module):
    """
    LeNet-5 变体，GroupNorm 替代 BatchNorm
    结构：Conv→Pool→Conv→Pool→FC→FC→FC
    """
    def __init__(self, dropout=0.3):
        super().__init__()
        # 特征提取
        self.conv1 = nn.Conv2d(1, 6,  kernel_size=5, padding=2)
        self.gn1   = nn.GroupNorm(2, 6)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.gn2   = nn.GroupNorm(4, 16)
        self.pool  = nn.AvgPool2d(2, 2)
        # 分类器
        self.fc1     = nn.Linear(16 * 5 * 5, 120)
        self.fc2     = nn.Linear(120, 84)
        self.fc3     = nn.Linear(84, 10)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pool(F.relu(self.gn1(self.conv1(x))))  # 28→14
        x = self.pool(F.relu(self.gn2(self.conv2(x))))  # 10→5
        x = x.view(x.size(0), -1)                       # 16*5*5=400
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.fc3(x)


# ── 模型注册表（方便统一调用）─────────────────
MODEL_REGISTRY = {
    "MLP":       MLP,
    "SimpleCNN": SimpleCNN,
    "LeNet5":    LeNet5,
}

def get_model(name):
    assert name in MODEL_REGISTRY, f"未知模型：{name}，可选：{list(MODEL_REGISTRY.keys())}"
    return MODEL_REGISTRY[name]()
