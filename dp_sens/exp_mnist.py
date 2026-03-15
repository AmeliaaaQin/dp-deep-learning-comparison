"""
exp_mnist.py - 主实验：数据脱敏 vs 梯度脱敏 × 三种模型

实验矩阵：
  模型  × 保护方式  × 隐私强度
  ─────────────────────────────
  MLP       × 无保护 / Laplace / Gaussian / 梯度DP
  SimpleCNN × 无保护 / Laplace / Gaussian / 梯度DP
  LeNet5    × 无保护 / Laplace / Gaussian / 梯度DP

运行方式：python exp_mnist.py
输出：results/all_results.json + results/figures/
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models_dp import get_model
from dp_mechanisms import (
    LaplaceSanitizer, GaussianSanitizer, NoSanitizer,
    make_dp_optimizer, DISPLAY_NAMES
)

# ─── 全局配置 ─────────────────────────────────────────────
EPOCHS      = 20
BATCH_SIZE  = 256
LR          = 0.05
MAX_GRAD_NORM = 1.0
DELTA       = 1e-5
DATA_DIR    = "./data"
RESULT_DIR  = "results"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 三种模型
MODELS = ["MLP", "SimpleCNN", "LeNet5"]

# 实验条件：(实验名, 保护类型, 参数)
# 保护类型：'none' / 'laplace' / 'gaussian' / 'gradient'
CONDITIONS = [
    # ── 无保护 baseline ──
    ("no_dp",          "none",     {}),
    # ── 数据脱敏：Laplace ──
    ("laplace_eps5",   "laplace",  {"epsilon": 5.0}),
    ("laplace_eps2",   "laplace",  {"epsilon": 2.0}),
    ("laplace_eps1",   "laplace",  {"epsilon": 1.0}),
    # ── 数据脱敏：Gaussian ──
    ("gaussian_eps5",  "gaussian", {"epsilon": 5.0}),
    ("gaussian_eps2",  "gaussian", {"epsilon": 2.0}),
    ("gaussian_eps1",  "gaussian", {"epsilon": 1.0}),
    # ── 梯度脱敏：DP-SGD ──
    ("grad_noise0.5",  "gradient", {"noise_multiplier": 0.5}),
    ("grad_noise1.0",  "gradient", {"noise_multiplier": 1.0}),
    ("grad_noise1.5",  "gradient", {"noise_multiplier": 1.5}),
]
# ─────────────────────────────────────────────────────────


def get_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=transform)
    test_set  = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    return train_set, test_set


def evaluate(model, loader, sanitizer=None):
    """评估模型准确率（数据脱敏实验中测试集不加噪）"""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            # 注意：测试时不对数据加噪（评估真实泛化能力）
            pred     = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total   += y.size(0)
    return correct / total


def run_single(model_name, cond_name, protect_type, params,
               train_set, test_set):
    """
    运行单个实验（一个模型 × 一种保护方式）

    返回：{
        "model": ..., "condition": ...,
        "final_acc": ..., "best_acc": ...,
        "acc_curve": [...],
        "epsilon": ...,  # 仅 gradient 模式有
        "train_time": ...
    }
    """
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    test_loader  = DataLoader(test_set, batch_size=512, shuffle=False)

    model     = get_model(model_name).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # 根据保护类型初始化脱敏器
    if protect_type == "laplace":
        sanitizer = LaplaceSanitizer(**params)
    elif protect_type == "gaussian":
        sanitizer = GaussianSanitizer(**params)
    else:
        sanitizer = NoSanitizer()

    # 梯度脱敏：用 Opacus 包装
    privacy_engine = None
    if protect_type == "gradient":
        privacy_engine, model, optimizer, train_loader = make_dp_optimizer(
            model, optimizer, train_loader,
            noise_multiplier=params["noise_multiplier"],
            max_grad_norm=MAX_GRAD_NORM,
        )

    acc_curve = []
    start     = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            # 数据脱敏：在 batch 上加噪
            if protect_type in ("laplace", "gaussian"):
                x = sanitizer(x)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()

        raw_model = model._module if hasattr(model, "_module") else model
        acc = evaluate(raw_model, test_loader)
        acc_curve.append(round(acc, 4))

    train_time = round(time.time() - start, 1)
    final_acc  = acc_curve[-1]
    best_acc   = max(acc_curve)
    epsilon    = None
    if privacy_engine:
        epsilon = round(privacy_engine.get_epsilon(DELTA), 4)

    return {
        "model":      model_name,
        "condition":  cond_name,
        "protect":    protect_type,
        "params":     params,
        "final_acc":  final_acc,
        "best_acc":   best_acc,
        "acc_curve":  acc_curve,
        "epsilon":    epsilon,
        "train_time": train_time,
    }


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    print(f"设备：{DEVICE}")
    print(f"实验规模：{len(MODELS)} 模型 × {len(CONDITIONS)} 条件 = "
          f"{len(MODELS) * len(CONDITIONS)} 组实验\n")

    train_set, test_set = get_datasets()
    all_results = []
    total       = len(MODELS) * len(CONDITIONS)
    done        = 0

    for model_name in MODELS:
        print(f"\n{'━'*55}")
        print(f"模型：{model_name}")
        print(f"{'━'*55}")

        for cond_name, protect_type, params in CONDITIONS:
            done += 1
            tag = DISPLAY_NAMES.get(cond_name, cond_name)
            print(f"\n  [{done}/{total}] {model_name} × {tag}")

            result = run_single(
                model_name, cond_name, protect_type, params,
                train_set, test_set
            )
            all_results.append(result)

            eps_str = f"  ε={result['epsilon']}" if result['epsilon'] else ""
            print(f"    最终 Acc={result['final_acc']:.4f}  "
                  f"最佳 Acc={result['best_acc']:.4f}  "
                  f"耗时={result['train_time']}s{eps_str}")

    # 保存结果
    out_path = os.path.join(RESULT_DIR, "all_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n\n✓ 所有结果已保存：{out_path}")

    # 打印汇总表
    print_summary(all_results)


def print_summary(results):
    print("\n\n" + "="*70)
    print("✅ 实验汇总（最终 Test Accuracy）")
    print("="*70)

    # 按模型分组打印
    for model_name in MODELS:
        print(f"\n  ── {model_name} ──")
        print(f"  {'保护方式':<22} {'最终Acc':>9} {'最佳Acc':>9} {'ε':>10}")
        print(f"  {'─'*52}")
        model_results = [r for r in results if r["model"] == model_name]
        for r in model_results:
            tag     = DISPLAY_NAMES.get(r["condition"], r["condition"])
            eps_str = f"{r['epsilon']:.4f}" if r["epsilon"] else "N/A"
            print(f"  {tag:<22} {r['final_acc']:>9.4f} {r['best_acc']:>9.4f} {eps_str:>10}")


if __name__ == "__main__":
    main()
