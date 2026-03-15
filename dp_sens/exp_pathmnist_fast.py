"""
exp_pathmnist.py - PathMNIST 差分隐私脱敏实验

数据集说明：
  PathMNIST：结直肠癌病理切片图像
  - 图像尺寸：3 × 28 × 28（RGB 彩色）
  - 类别数量：9 类组织
  - 训练集：~89,996 张，测试集：7,180 张
  - 来源：NCT-CRC-HE-100K

实验设计（与 MNIST 最优配置对比）：
  LeNet5 模型（在 MNIST 上表现最鲁棒的模型）

  保护条件（对比在 MNIST 上的最优配置）：
  1. 无保护 baseline
  2. Laplace 数据脱敏 ε = 2
  3. Gaussian 数据脱敏 ε = 2
  4. 梯度脱敏 DP-SGD σ = 1.0
  5. 额外强度：Laplace/Gaussian ε=5,1 和梯度DP σ=0.5,1.5（共10组）

运行方式：
  python exp_pathmnist.py

输出：
  pathmnist_results/ 目录
  ├── results_lenet5.json         原始数据
  ├── pathmnist_lenet5_results.png 训练曲线 + 准确率对比
  └── mnist_vs_pathmnist_lenet5.png MNIST vs PathMNIST 跨数据集对比

运行时间估计（CPU）：
  - 每轮实验约 1-1.5 小时
  - 10 组实验总计约 12-15 小时
  - 建议分批次运行或 overnight 执行
"""

"""
exp_pathmnist_fast.py - PathMNIST快速验证实验
只跑最鲁棒的模型 LeNet5，看趋势是否和MNIST一致
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from opacus import PrivacyEngine

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# MedMNIST
import medmnist
from medmnist import PathMNIST

# ─── 配置 ─────────────────────────────────────────────
EPOCHS = 15
BATCH_SIZE = 128  # 稍微减小加快速度
LR = 0.01
MAX_GRAD_NORM = 1.0
DELTA = 1e-5
RESULT_DIR = "pathmnist_fast_results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 只跑一个模型：LeNet5
MODEL_NAME = "LeNet5"

# 实验条件（10组全跑，但只跑一个模型）
CONDITIONS = [
    ("baseline", "none", {}),
    ("laplace_eps5", "laplace", {"epsilon": 5.0}),
    ("laplace_eps2", "laplace", {"epsilon": 2.0}),
    ("laplace_eps1", "laplace", {"epsilon": 1.0}),
    ("gaussian_eps5", "gaussian", {"epsilon": 5.0}),
    ("gaussian_eps2", "gaussian", {"epsilon": 2.0}),
    ("gaussian_eps1", "gaussian", {"epsilon": 1.0}),
    ("grad_noise0.5", "gradient", {"noise_multiplier": 0.5}),
    ("grad_noise1.0", "gradient", {"noise_multiplier": 1.0}),
    ("grad_noise1.5", "gradient", {"noise_multiplier": 1.5}),
]

DISPLAY = {
    "baseline": "无保护 (Baseline)",
    "laplace_eps5": "Laplace ε=5",
    "laplace_eps2": "Laplace ε=2",
    "laplace_eps1": "Laplace ε=1",
    "gaussian_eps5": "Gaussian ε=5",
    "gaussian_eps2": "Gaussian ε=2",
    "gaussian_eps1": "Gaussian ε=1",
    "grad_noise0.5": "梯度DP σ=0.5",
    "grad_noise1.0": "梯度DP σ=1.0",
    "grad_noise1.5": "梯度DP σ=1.5",
}


# ─────────────────────────────────────────────────────


# ══════════════════════════════════════════════
# LeNet5 模型（适配 PathMNIST 3通道输入）
# ══════════════════════════════════════════════
class LeNet5_PathMNIST(nn.Module):
    """
    经典 LeNet-5 结构，适配 3×28×28 输入，9类输出
    """

    def __init__(self):
        super().__init__()
        # 特征提取
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=2)
        self.gn1 = nn.GroupNorm(2, 6)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.gn2 = nn.GroupNorm(4, 16)
        self.pool = nn.AvgPool2d(2, 2)

        # 分类器
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 9)  # PathMNIST有9类
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.gn1(self.conv1(x))))  # 28→14
        x = self.pool(F.relu(self.gn2(self.conv2(x))))  # 14→10? 要重新算
        # conv2无padding: 14-5+1=10，pool后10→5
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.fc3(x)


# ══════════════════════════════════════════════
# 差分隐私脱敏机制
# ══════════════════════════════════════════════
class LaplaceSanitizer:
    def __init__(self, epsilon, sensitivity=1.0, clip_range=(0.0, 1.0)):
        self.scale = sensitivity / epsilon
        self.clip_range = clip_range

    def __call__(self, x):
        noise = torch.tensor(
            np.random.laplace(0, self.scale, x.shape),
            dtype=x.dtype, device=x.device
        )
        return torch.clamp(x + noise, *self.clip_range)


class GaussianSanitizer:
    def __init__(self, epsilon, delta=1e-5, sensitivity=1.0, clip_range=(0.0, 1.0)):
        self.sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        self.clip_range = clip_range

    def __call__(self, x):
        return torch.clamp(x + torch.randn_like(x) * self.sigma, *self.clip_range)


class NoSanitizer:
    def __call__(self, x):
        return x


# ══════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════
def get_dataloaders(batch_size):
    # PathMNIST 归一化参数
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.7405, 0.5330, 0.7058],
            std=[0.1237, 0.1768, 0.1244]
        )
    ])

    train_set = PathMNIST(split="train", transform=transform,
                          download=True, root="./data/medmnist")
    test_set = PathMNIST(split="test", transform=transform,
                         download=True, root="./data/medmnist")

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, drop_last=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=512,
                             shuffle=False, num_workers=0)

    print(f"  训练集大小：{len(train_set)}")
    print(f"  测试集大小：{len(test_set)}")
    return train_loader, test_loader


# ══════════════════════════════════════════════
# 评估
# ══════════════════════════════════════════════
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
            y = y.squeeze().long()
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


# ══════════════════════════════════════════════
# 单组实验
# ══════════════════════════════════════════════
def run_experiment(cond_name, protect_type, params):
    print(f"\n{'=' * 55}")
    print(f"实验：{DISPLAY[cond_name]}")
    print(f"{'=' * 55}")

    train_loader, test_loader = get_dataloaders(BATCH_SIZE)

    model = LeNet5_PathMNIST().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # 初始化脱敏器
    if protect_type == "laplace":
        sanitizer = LaplaceSanitizer(**params)
    elif protect_type == "gaussian":
        sanitizer = GaussianSanitizer(**params)
    else:
        sanitizer = NoSanitizer()

    # DP-SGD
    privacy_engine = None
    if protect_type == "gradient":
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=params["noise_multiplier"],
            max_grad_norm=MAX_GRAD_NORM,
        )

    acc_curve = []
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch in train_loader:
            x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
            y = y.squeeze().long()

            if protect_type in ("laplace", "gaussian"):
                x = sanitizer(x)

            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()

        raw = model._module if hasattr(model, "_module") else model
        acc = evaluate(raw, test_loader)
        acc_curve.append(round(acc, 4))

        if privacy_engine:
            eps = privacy_engine.get_epsilon(DELTA)
            print(f"  Epoch {epoch:2d}/{EPOCHS} | Test Acc: {acc:.4f} | ε={eps:.4f}")
        else:
            print(f"  Epoch {epoch:2d}/{EPOCHS} | Test Acc: {acc:.4f}")

    final_eps = None
    if privacy_engine:
        final_eps = round(privacy_engine.get_epsilon(DELTA), 4)

    result = {
        "model": "LeNet5",
        "condition": cond_name,
        "protect": protect_type,
        "params": params,
        "final_acc": acc_curve[-1],
        "best_acc": max(acc_curve),
        "acc_curve": acc_curve,
        "epsilon": final_eps,
        "train_time": round(time.time() - start, 1),
    }

    print(f"\n  ✓ 完成 | 最终 Acc={result['final_acc']:.4f} "
          f"最佳 Acc={result['best_acc']:.4f} "
          f"耗时={result['train_time']}s"
          + (f" | ε={final_eps}" if final_eps else ""))
    return result


# ══════════════════════════════════════════════
# 可视化
# ══════════════════════════════════════════════
def plot_results(results, save_dir):
    colors = {
        "baseline": "#2C3E50",
        "laplace_eps5": "#F39C12",
        "laplace_eps2": "#E67E22",
        "laplace_eps1": "#E74C3C",
        "gaussian_eps5": "#BB8FCE",
        "gaussian_eps2": "#8E44AD",
        "gaussian_eps1": "#6C3483",
        "grad_noise0.5": "#7FB3D3",
        "grad_noise1.0": "#2980B9",
        "grad_noise1.5": "#1A5276",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"PathMNIST LeNet5 差分隐私脱敏实验\n（结直肠癌病理图像，9类分类）",
                 fontsize=13, fontweight="bold")

    # 左图：训练曲线
    ax1 = axes[0]
    for r in results:
        name = r["condition"]
        label = DISPLAY[name]
        if r.get("epsilon"):
            label += f" (ε={r['epsilon']:.2f})"
        ax1.plot(range(1, len(r["acc_curve"]) + 1), r["acc_curve"],
                 color=colors[name], linewidth=2, marker="o",
                 markersize=3, label=label)

    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Test Accuracy", fontsize=11)
    ax1.set_title("训练过程：各保护方式测试准确率", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(alpha=0.3)

    # 右图：最终准确率对比柱状图
    ax2 = axes[1]
    names = [r["condition"] for r in results]
    final_accs = [r["final_acc"] for r in results]
    bar_colors = [colors[n] for n in names]
    x = np.arange(len(names))

    bars = ax2.bar(x, final_accs, 0.6, color=bar_colors,
                   alpha=0.85, edgecolor="black", linewidth=0.5)

    # 标注数值
    for bar, acc in zip(bars, final_accs):
        ax2.text(bar.get_x() + bar.get_width() / 2.,
                 bar.get_height() + 0.01,
                 f"{acc:.3f}",
                 ha="center", va="bottom", fontsize=9, rotation=45)

    ax2.set_xticks(x)
    ax2.set_xticklabels([DISPLAY[n] for n in names], fontsize=8, rotation=45, ha="right")
    ax2.set_ylabel("Test Accuracy", fontsize=11)
    ax2.set_title("各保护方式最终准确率对比", fontsize=11, fontweight="bold")
    ax2.set_ylim([0, 1.0])
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "pathmnist_lenet5_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✓ 结果图已保存：{path}")


def plot_comparison_with_mnist(results, save_dir):
    """与MNIST实验结果对比图"""
    # MNIST上LeNet5的结果（从你的实验来）
    mnist_accs = {
        "baseline": 0.9923,
        "laplace_eps5": 0.9925,
        "laplace_eps2": 0.9915,
        "laplace_eps1": 0.9929,
        "gaussian_eps5": 0.9916,
        "gaussian_eps2": 0.9820,
        "gaussian_eps1": 0.9297,
        "grad_noise0.5": 0.9759,
        "grad_noise1.0": 0.9733,
        "grad_noise1.5": 0.9636,
    }
    path_accs = {r["condition"]: r["final_acc"] for r in results}

    conditions = [r["condition"] for r in results]
    labels = [DISPLAY[c] for c in conditions]
    mnist_vals = [mnist_accs[c] for c in conditions]
    path_vals = [path_accs.get(c, 0) for c in conditions]

    x = np.arange(len(conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width / 2, mnist_vals, width, label="MNIST (手写数字)",
                   color="#3498DB", alpha=0.85, edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, path_vals, width, label="PathMNIST (病理图像)",
                   color="#E74C3C", alpha=0.85, edgecolor="black", linewidth=0.5)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.,
                    height + 0.01,
                    f"{height:.3f}",
                    ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("LeNet5: MNIST vs PathMNIST 差分隐私保护效果对比\n"
                 "（验证结论在不同复杂度数据集上的普适性）",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.1])
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "mnist_vs_pathmnist_lenet5.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ 对比图已保存：{path}")


# ══════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════
def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    print(f"设备：{DEVICE}")
    print(f"数据集：PathMNIST（结直肠癌病理图像，9类，RGB 28×28）")
    print(f"模型：LeNet5（最鲁棒的模型）")
    print(f"实验条件：{len(CONDITIONS)} 组\n")

    all_results = []
    for cond_name, protect_type, params in CONDITIONS:
        result = run_experiment(cond_name, protect_type, params)
        all_results.append(result)

        # 每跑完一组立刻保存
        with open(os.path.join(RESULT_DIR, "results_lenet5.json"), "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 汇总打印
    print("\n\n" + "=" * 60)
    print("✅ PathMNIST LeNet5 实验汇总")
    print("=" * 60)
    print(f"  {'保护方式':<22} {'最终Acc':>9} {'最佳Acc':>9} {'ε':>10} {'耗时':>8}")
    print("─" * 60)
    for r in all_results:
        eps_str = f"{r['epsilon']:.4f}" if r["epsilon"] else "N/A"
        print(f"  {DISPLAY[r['condition']]:<22} "
              f"{r['final_acc']:>9.4f} {r['best_acc']:>9.4f} "
              f"{eps_str:>10} {r['train_time']:>7.1f}s")
    print("─" * 60)

    # 生成图表
    print("\n生成图表...")
    plot_results(all_results, RESULT_DIR)
    plot_comparison_with_mnist(all_results, RESULT_DIR)

    print(f"\n✅ 所有结果已保存至：{RESULT_DIR}/")
    print("  - results_lenet5.json     原始数据")
    print("  - pathmnist_lenet5_results.png  训练曲线 + 准确率对比")
    print("  - mnist_vs_pathmnist_lenet5.png MNIST vs PathMNIST 跨数据集对比")


if __name__ == "__main__":
    main()
