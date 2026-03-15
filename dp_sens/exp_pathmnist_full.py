"""
exp_pathmnist_full.py - PathMNIST 差分隐私脱敏实验（完整30组）

数据集说明：
  PathMNIST：结直肠癌病理切片图像
  - 图像尺寸：3 × 28 × 28（RGB 彩色）
  - 类别数量：9 类组织
  - 训练集：~89,996 张，测试集：7,180 张
  - 来源：NCT-CRC-HE-100K

实验设计：
  3个模型 × 10种条件 = 30组实验

  模型（按计算量从小到大排序）：
    1. MLP（最快，先跑）
    2. SimpleCNN（中等）
    3. LeNet5（最慢，最后跑）

  保护条件：
    1. 无保护 baseline
    2. Laplace 数据脱敏 ε=5
    3. Laplace 数据脱敏 ε=2
    4. Laplace 数据脱敏 ε=1
    5. Gaussian 数据脱敏 ε=5
    6. Gaussian 数据脱敏 ε=2
    7. Gaussian 数据脱敏 ε=1
    8. 梯度DP σ=0.5
    9. 梯度DP σ=1.0
    10. 梯度DP σ=1.5

运行方式：
  python exp_pathmnist_full.py

输出：
  pathmnist_full_results/ 目录
  ├── results_mlp.json
  ├── results_simplecnn.json
  ├── results_lenet5.json
  ├── pathmnist_mlp_results.png
  ├── pathmnist_simplecnn_results.png
  ├── pathmnist_lenet5_results.png
  └── pathmnist_all_models_comparison.png

运行时间估计（CPU）：
  - MLP：10组 × 约0.5小时 = 5小时
  - SimpleCNN：10组 × 约1小时 = 10小时
  - LeNet5：10组 × 约1.5小时 = 15小时
  - 总计：约30小时（建议分3天运行）
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
BATCH_SIZE = 128
LR = 0.01
MAX_GRAD_NORM = 1.0
DELTA = 1e-5
RESULT_DIR = "pathmnist_full_results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型列表（按计算量从小到大排序，LeNet5放最后）
MODELS = ["MLP", "SimpleCNN", "LeNet5"]

# 实验条件（10组）
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

# 颜色方案
COLORS = {
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


# ─────────────────────────────────────────────────────


# ══════════════════════════════════════════════
# 模型定义（适配 PathMNIST 3通道输入）
# ══════════════════════════════════════════════

class MLP_PathMNIST(nn.Module):
    """MLP适配PathMNIST（3通道28x28输入，9类输出）"""

    def __init__(self, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 9)
        )

    def forward(self, x):
        return self.net(x)


class SimpleCNN_PathMNIST(nn.Module):
    """SimpleCNN适配PathMNIST（3通道输入）"""

    def __init__(self, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, 32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(16, 64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 9)

    def forward(self, x):
        x = self.pool(F.relu(self.gn1(self.conv1(x))))  # 28→14
        x = self.pool(F.relu(self.gn2(self.conv2(x))))  # 14→7
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


class LeNet5_PathMNIST(nn.Module):
    """LeNet-5适配PathMNIST（3通道输入）"""

    def __init__(self, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=2)
        self.gn1 = nn.GroupNorm(2, 6)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.gn2 = nn.GroupNorm(4, 16)
        self.pool = nn.AvgPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 9)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pool(F.relu(self.gn1(self.conv1(x))))  # 28→14
        x = self.pool(F.relu(self.gn2(self.conv2(x))))  # 14→10→5
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.fc3(x)


# 模型注册表
MODEL_CLASSES = {
    "MLP": MLP_PathMNIST,
    "SimpleCNN": SimpleCNN_PathMNIST,
    "LeNet5": LeNet5_PathMNIST,
}


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
    """加载PathMNIST数据集"""
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
# 评估函数
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

def run_experiment(model_name, cond_name, protect_type, params):
    """运行单组实验"""
    print(f"\n{'=' * 60}")
    print(f"模型：{model_name} | 实验：{DISPLAY[cond_name]}")
    print(f"{'=' * 60}")

    train_loader, test_loader = get_dataloaders(BATCH_SIZE)

    # 创建模型
    model_class = MODEL_CLASSES[model_name]
    model = model_class().to(DEVICE)
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
            loss = criterion(model(x), y)
            loss.backward()
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
        "model": model_name,
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

def plot_model_results(results, model_name, save_dir):
    """绘制单个模型的训练曲线和准确率对比"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"PathMNIST {model_name} 差分隐私脱敏实验\n（结直肠癌病理图像，9类分类）",
                 fontsize=13, fontweight="bold")

    # 左图：训练曲线
    ax1 = axes[0]
    for r in results:
        name = r["condition"]
        label = DISPLAY[name]
        if r.get("epsilon"):
            label += f" (ε={r['epsilon']:.2f})"
        ax1.plot(range(1, len(r["acc_curve"]) + 1), r["acc_curve"],
                 color=COLORS[name], linewidth=2, marker="o",
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
    bar_colors = [COLORS[n] for n in names]
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
    path = os.path.join(save_dir, f"pathmnist_{model_name.lower()}_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ {model_name}结果图已保存：{path}")


def plot_all_models_comparison(all_results, save_dir):
    """绘制三个模型的最终准确率对比柱状图"""
    models = ["MLP", "SimpleCNN", "LeNet5"]
    conditions = [c[0] for c in CONDITIONS]

    fig, ax = plt.subplots(figsize=(15, 7))
    x = np.arange(len(conditions))
    width = 0.25

    for i, model in enumerate(models):
        model_results = [r for r in all_results if r["model"] == model]
        if not model_results:
            continue
        model_dict = {r["condition"]: r["final_acc"] for r in model_results}
        accs = [model_dict.get(cond, 0) for cond in conditions]

        offset = (i - 1) * width
        bars = ax.bar(x + offset, accs, width, label=model,
                      color=["#E74C3C", "#2980B9", "#27AE60"][i],
                      alpha=0.8, edgecolor="black", linewidth=0.5)

        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2.,
                    bar.get_height() + 0.005,
                    f"{acc:.3f}",
                    ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[cond] for cond in conditions], fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("PathMNIST: 三个模型在不同保护方式下的准确率对比", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.0])
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "pathmnist_all_models_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ 三模型对比图已保存：{path}")


# ══════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    print(f"设备：{DEVICE}")
    print(f"数据集：PathMNIST（结直肠癌病理图像，9类，RGB 28×28）")
    print(f"模型列表（按计算量从小到大）：{MODELS}")
    print(f"实验条件：{len(CONDITIONS)} 组/模型")
    print(f"总实验组数：{len(MODELS) * len(CONDITIONS)} 组\n")

    all_results = []

    for model_idx, model_name in enumerate(MODELS):
        print(f"\n\n{'#' * 70}")
        print(f"# 开始运行模型 [{model_idx + 1}/{len(MODELS)}]: {model_name}")
        print(f"{'#' * 70}\n")

        model_results = []
        for cond_name, protect_type, params in CONDITIONS:
            result = run_experiment(model_name, cond_name, protect_type, params)
            model_results.append(result)
            all_results.append(result)

            # 每跑完一组立刻保存该模型的结果
            with open(os.path.join(RESULT_DIR, f"results_{model_name.lower()}.json"), "w", encoding="utf-8") as f:
                json.dump(model_results, f, indent=2, ensure_ascii=False)

        # 绘制该模型的结果图
        plot_model_results(model_results, model_name, RESULT_DIR)

        # 打印该模型的汇总
        print(f"\n\n{'=' * 60}")
        print(f"✅ {model_name} 实验汇总")
        print('=' * 60)
        print(f"  {'保护方式':<22} {'最终Acc':>9} {'最佳Acc':>9} {'ε':>10} {'耗时':>8}")
        print("─" * 60)
        for r in model_results:
            eps_str = f"{r['epsilon']:.4f}" if r["epsilon"] else "N/A"
            print(f"  {DISPLAY[r['condition']]:<22} "
                  f"{r['final_acc']:>9.4f} {r['best_acc']:>9.4f} "
                  f"{eps_str:>10} {r['train_time']:>7.1f}s")
        print("─" * 60)

    # 绘制三模型对比图
    plot_all_models_comparison(all_results, RESULT_DIR)

    # 保存全部结果
    with open(os.path.join(RESULT_DIR, "all_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n\n{'=' * 60}")
    print("✅ 所有实验完成！")
    print('=' * 60)
    print(f"结果已保存至：{RESULT_DIR}/")
    print("  - results_mlp.json")
    print("  - results_simplecnn.json")
    print("  - results_lenet5.json")
    print("  - all_results.json")
    print("  - pathmnist_mlp_results.png")
    print("  - pathmnist_simplecnn_results.png")
    print("  - pathmnist_lenet5_results.png")
    print("  - pathmnist_all_models_comparison.png")


if __name__ == "__main__":
    main()
