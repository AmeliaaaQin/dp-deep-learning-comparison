"""
step1_train_model_final.py - 最终版训练脚本

关键设计：所有模型（baseline + DP）都用同样的小数据集（5000条）训练。
这样过拟合压力对所有模型均等，DP 噪声的抑制效果才能体现出差异。

攻击数据集构建（统一）：
  - 成员   = 训练用的前 5000 条
  - 非成员 = 训练集中从未用过的 5000 条（索引 5000~10000）

运行方式：python step1_final.py
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from opacus import PrivacyEngine
from models import SimpleCNN_MNIST

# ─── 超参数 ───────────────────────────────────────────────
TRAIN_SIZE    = 5000   # 所有模型统一用这么多训练数据
EPOCHS        = 40     # 足够多的轮次制造过拟合
BATCH_SIZE    = 64     # 小 batch，DP 效果更明显，ε 也更合理
LR            = 0.05
MAX_GRAD_NORM = 1.0
DELTA         = 1e-5
SAVE_DIR      = "saved_models"
DATA_DIR      = "./data"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXPERIMENTS = [
    ("baseline",    None),
    ("dp_noise0.5", 0.5),
    ("dp_noise1.0", 1.0),
    ("dp_noise1.5", 1.5),
]
# ─────────────────────────────────────────────────────────


def load_raw_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=transform)
    test_set  = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    return train_set, test_set


def get_loaders(train_set, test_set, batch_size):
    """所有模型统一用训练集前 TRAIN_SIZE 条"""
    subset    = Subset(train_set, list(range(TRAIN_SIZE)))
    train_loader = DataLoader(subset,    batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader  = DataLoader(test_set,  batch_size=512, shuffle=False)
    return train_loader, test_loader


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            correct += (model(x).argmax(1) == y).sum().item()
            total   += y.size(0)
    return correct / total


def run_experiment(exp_name, noise_multiplier, train_set, test_set):
    print(f"\n{'='*55}")
    tag = f"noise={noise_multiplier}" if noise_multiplier else "无DP"
    print(f"训练：{exp_name}  [{tag}，数据量={TRAIN_SIZE}，epochs={EPOCHS}]")
    print(f"{'='*55}")

    train_loader, test_loader = get_loaders(train_set, test_set, BATCH_SIZE)

    # 构建训练集子集 loader（用于评估训练集 acc）
    member_loader = DataLoader(
        Subset(train_set, list(range(TRAIN_SIZE))),
        batch_size=512, shuffle=False
    )

    model     = SimpleCNN_MNIST().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    history   = {"train_acc": [], "test_acc": [], "epsilon": []}

    privacy_engine = None
    if noise_multiplier is not None:
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=MAX_GRAD_NORM,
        )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            optimizer.step()

        raw = model._module if hasattr(model, "_module") else model
        train_acc = evaluate(raw, member_loader)
        test_acc  = evaluate(raw, test_loader)
        epsilon   = privacy_engine.get_epsilon(DELTA) if privacy_engine else None

        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["epsilon"].append(epsilon)

        if epoch % 10 == 0 or epoch == 1:
            eps_str = f"ε={epsilon:.4f}" if epsilon else "ε=N/A"
            gap = train_acc - test_acc
            print(f"  Epoch {epoch:2d} | Train:{train_acc:.4f} Test:{test_acc:.4f} "
                  f"过拟合:{gap:+.4f} | {eps_str}")

    # 保存
    os.makedirs(SAVE_DIR, exist_ok=True)
    raw = model._module if hasattr(model, "_module") else model
    torch.save(raw.state_dict(), os.path.join(SAVE_DIR, f"{exp_name}.pt"))

    final_gap = history["train_acc"][-1] - history["test_acc"][-1]
    final_eps = history["epsilon"][-1]
    print(f"\n  ✓ 保存完成")
    print(f"  最终 Train:{history['train_acc'][-1]:.4f} "
          f"Test:{history['test_acc'][-1]:.4f} 过拟合差距:{final_gap:+.4f}")
    if final_eps:
        print(f"  最终 ε = {final_eps:.4f}")
    return history


def main():
    print(f"设备：{DEVICE}")
    print(f"\n实验设计：所有模型统一用 {TRAIN_SIZE} 条数据训练 {EPOCHS} 轮")
    print(f"攻击时：成员=前{TRAIN_SIZE}条，非成员=索引{TRAIN_SIZE}~{TRAIN_SIZE*2}条\n")

    train_set, test_set = load_raw_datasets()
    all_history = {}

    for exp_name, noise in EXPERIMENTS:
        hist = run_experiment(exp_name, noise, train_set, test_set)
        all_history[exp_name] = {
            "train_acc": hist["train_acc"],
            "test_acc":  hist["test_acc"],
            "epsilon":   [str(e) if e else "null" for e in hist["epsilon"]]
        }

    # 保存历史
    with open(os.path.join(SAVE_DIR, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(all_history, f, indent=2, ensure_ascii=False)

    # 汇总
    print("\n\n" + "="*65)
    print("✅ 训练完成汇总")
    print("="*65)
    print(f"  {'实验':<20} {'Train':>8} {'Test':>8} {'过拟合差距':>12} {'最终ε':>10}")
    print("─"*65)
    for name, hist in all_history.items():
        tr  = hist["train_acc"][-1]
        te  = hist["test_acc"][-1]
        gap = tr - te
        eps = hist["epsilon"][-1]
        eps_str = f"{float(eps):.4f}" if eps != "null" else "N/A"
        print(f"  {name:<20} {tr:>8.4f} {te:>8.4f} {gap:>+12.4f} {eps_str:>10}")
    print("─"*65)
    print("\n  ⭐ 关键：baseline 过拟合差距应最大，dp_noise1.5 应最小")
    print("     step2 的攻击 AUC 应呈现：baseline > dp0.5 > dp1.0 > dp1.5")


if __name__ == "__main__":
    main()
