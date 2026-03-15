"""
dp_mechanisms.py - 差分隐私脱敏机制

包含：
  1. 数据脱敏（Data Sanitization）
     - Laplace 机制：适合 L1 敏感度场景
     - Gaussian 机制：适合 L2 敏感度场景
  2. 梯度脱敏（Gradient Sanitization / DP-SGD）
     - 基于 Opacus 实现，封装为统一接口
  3. 隐私预算计算工具
"""

import torch
import torch.nn as nn
import numpy as np
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier


# ══════════════════════════════════════════════
# 数据脱敏机制
# ══════════════════════════════════════════════

class LaplaceSanitizer:
    """
    Laplace 机制数据脱敏

    原理：对输入数据 x 加入 Laplace 噪声
          x_priv = x + Lap(sensitivity / epsilon)

    适用：数据值域有界，使用 L1 敏感度

    参数：
      epsilon    : 隐私预算（越小隐私越强）
      sensitivity: 数据的 L1 敏感度（MNIST 像素归一化后约为 1.0）
      clip_range : 加噪后裁剪到合法范围，None 则不裁剪
    """
    def __init__(self, epsilon: float, sensitivity: float = 1.0,
                 clip_range: tuple = (-1.0, 2.0)):
        self.epsilon    = epsilon
        self.sensitivity = sensitivity
        self.clip_range = clip_range
        self.scale      = sensitivity / epsilon   # Laplace 分布的 b 参数

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.tensor(
            np.random.laplace(0, self.scale, x.shape),
            dtype=x.dtype, device=x.device
        )
        x_noisy = x + noise
        if self.clip_range:
            x_noisy = torch.clamp(x_noisy, *self.clip_range)
        return x_noisy

    def __repr__(self):
        return f"LaplaceSanitizer(ε={self.epsilon}, sensitivity={self.sensitivity})"


class GaussianSanitizer:
    """
    Gaussian 机制数据脱敏

    原理：对输入数据加入高斯噪声
          x_priv = x + N(0, (sensitivity * noise_multiplier)^2)

    满足 (ε, δ)-差分隐私，δ 通常设为 1e-5

    参数：
      epsilon    : 隐私预算
      delta      : 隐私松弛参数（通常 1e-5）
      sensitivity: 数据的 L2 敏感度
      clip_range : 加噪后裁剪
    """
    def __init__(self, epsilon: float, delta: float = 1e-5,
                 sensitivity: float = 1.0, clip_range: tuple = (-1.0, 2.0)):
        self.epsilon     = epsilon
        self.delta       = delta
        self.sensitivity = sensitivity
        self.clip_range  = clip_range
        # 根据 (ε, δ) 计算 Gaussian 噪声标准差
        # 使用近似公式：σ = sensitivity * sqrt(2 * ln(1.25/δ)) / ε
        self.sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        noise   = torch.randn_like(x) * self.sigma
        x_noisy = x + noise
        if self.clip_range:
            x_noisy = torch.clamp(x_noisy, *self.clip_range)
        return x_noisy

    def __repr__(self):
        return (f"GaussianSanitizer(ε={self.epsilon}, δ={self.delta}, "
                f"σ={self.sigma:.4f})")


class NoSanitizer:
    """占位符：不做任何脱敏（baseline 用）"""
    def __call__(self, x):
        return x
    def __repr__(self):
        return "NoSanitizer(baseline)"


# ══════════════════════════════════════════════
# 梯度脱敏（DP-SGD）封装
# ══════════════════════════════════════════════

def make_dp_optimizer(model, optimizer, train_loader,
                      noise_multiplier, max_grad_norm=1.0):
    """
    使用 Opacus 将普通优化器升级为 DP 优化器（梯度脱敏）

    返回：(privacy_engine, dp_model, dp_optimizer, dp_loader)
    """
    privacy_engine = PrivacyEngine()
    dp_model, dp_optimizer, dp_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )
    return privacy_engine, dp_model, dp_optimizer, dp_loader


# ══════════════════════════════════════════════
# 实验配置注册表
# ══════════════════════════════════════════════

# 数据脱敏实验配置
DATA_SANITIZATION_CONFIGS = {
    "no_dp":           NoSanitizer(),
    "laplace_eps5":    LaplaceSanitizer(epsilon=5.0),
    "laplace_eps2":    LaplaceSanitizer(epsilon=2.0),
    "laplace_eps1":    LaplaceSanitizer(epsilon=1.0),
    "gaussian_eps5":   GaussianSanitizer(epsilon=5.0),
    "gaussian_eps2":   GaussianSanitizer(epsilon=2.0),
    "gaussian_eps1":   GaussianSanitizer(epsilon=1.0),
}

# 梯度脱敏实验配置（noise_multiplier）
GRADIENT_SANITIZATION_CONFIGS = {
    "no_dp":        None,    # 不使用 DP-SGD
    "grad_noise0.5": 0.5,
    "grad_noise1.0": 1.0,
    "grad_noise1.5": 1.5,
}

# 标签映射（用于绘图）
DISPLAY_NAMES = {
    "no_dp":           "无保护 (Baseline)",
    "laplace_eps5":    "Laplace ε=5",
    "laplace_eps2":    "Laplace ε=2",
    "laplace_eps1":    "Laplace ε=1",
    "gaussian_eps5":   "Gaussian ε=5",
    "gaussian_eps2":   "Gaussian ε=2",
    "gaussian_eps1":   "Gaussian ε=1",
    "grad_noise0.5":   "梯度DP σ=0.5",
    "grad_noise1.0":   "梯度DP σ=1.0",
    "grad_noise1.5":   "梯度DP σ=1.5",
}
