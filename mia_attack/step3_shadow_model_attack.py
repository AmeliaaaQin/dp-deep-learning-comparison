"""
step3_shadow_model_attack.py - 进阶：影子模型成员推断攻击

原理：比简单阈值攻击更强的攻击方法。
      训练多个"影子模型"来模拟目标模型的行为，
      再训练一个"攻击分类器"来识别成员/非成员。

这是 Shokri et al. (2017) 的经典方法，也是毕设展示"深度"的好选择。

运行方式：python step3_shadow_model_attack.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import matplotlib

from models import SimpleCNN_MNIST

# ─── 配置 ────────────────────────────────────────────────
SAVE_DIR      = "saved_models"
RESULT_DIR    = "mia_results"
DATA_DIR      = "./data"
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SHADOW      = 3       # 影子模型数量（越多越好，但更慢）
SHADOW_EPOCHS = 5       # 影子模型训练轮次（少一点节省时间）
SHADOW_SIZE   = 5000    # 每个影子模型的数据量
# ─────────────────────────────────────────────────────────


def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_set = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=transform)
    test_set = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    return full_set, test_set


def train_shadow_model(train_subset, epochs=SHADOW_EPOCHS):
    """快速训练一个影子模型"""
    loader = DataLoader(train_subset, batch_size=256, shuffle=True, drop_last=True)
    model  = SimpleCNN_MNIST().to(DEVICE)
    opt    = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    crit   = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
    return model


def extract_features(model, loader):
    """提取模型对样本的 softmax 输出向量作为攻击特征"""
    model.eval()
    features = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            probs = F.softmax(model(x), dim=1).cpu().numpy()
            features.append(probs)
    return np.concatenate(features)


def build_attack_classifier(full_set, n_shadow=N_SHADOW):
    """
    训练影子模型并收集攻击训练数据，
    最终训练一个 RandomForest 攻击分类器
    """
    print(f"  训练 {n_shadow} 个影子模型...")
    attack_X, attack_y = [], []

    total_indices = np.arange(len(full_set))
    rng = np.random.default_rng(2024)

    for i in range(n_shadow):
        # 为每个影子模型随机划分"成员"和"非成员"
        shadow_indices = rng.choice(total_indices, SHADOW_SIZE * 2, replace=False)
        member_idx     = shadow_indices[:SHADOW_SIZE]
        nonmember_idx  = shadow_indices[SHADOW_SIZE:]

        # 训练影子模型（只用 member 数据）
        shadow_train = Subset(full_set, member_idx)
        shadow_model = train_shadow_model(shadow_train)
        print(f"    影子模型 {i+1}/{n_shadow} 训练完成")

        # 提取特征
        member_loader     = DataLoader(Subset(full_set, member_idx),
                                       batch_size=256, shuffle=False)
        nonmember_loader  = DataLoader(Subset(full_set, nonmember_idx),
                                       batch_size=256, shuffle=False)

        member_feats     = extract_features(shadow_model, member_loader)
        nonmember_feats  = extract_features(shadow_model, nonmember_loader)

        # 截断到相同数量
        n_min = min(len(member_feats), len(nonmember_feats))
        attack_X.append(member_feats[:n_min])
        attack_X.append(nonmember_feats[:n_min])
        attack_y.extend([1] * n_min)   # 成员 = 1
        attack_y.extend([0] * n_min)   # 非成员 = 0

    attack_X = np.concatenate(attack_X)
    attack_y = np.array(attack_y)

    print(f"  攻击训练数据规模：{attack_X.shape}")
    print(f"  训练攻击分类器（RandomForest）...")

    attack_clf = RandomForestClassifier(
        n_estimators=100, max_depth=10,
        random_state=42, n_jobs=-1
    )
    attack_clf.fit(attack_X, attack_y)
    return attack_clf


def run_shadow_mia(target_model_name, attack_clf, full_set, test_set):
    """用影子模型攻击分类器攻击目标模型"""
    model_path = os.path.join(SAVE_DIR, f"{target_model_name}.pt")
    if not os.path.exists(model_path):
        print(f"  ⚠ 未找到：{model_path}")
        return None

    model = SimpleCNN_MNIST().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    # 构建评估数据
    rng = np.random.default_rng(99)
    n_eval = 1000

    train_idx = rng.choice(len(full_set), n_eval, replace=False)
    test_idx  = rng.choice(len(test_set), n_eval, replace=False)

    member_loader     = DataLoader(Subset(full_set, train_idx), batch_size=256, shuffle=False)
    nonmember_loader  = DataLoader(Subset(test_set, test_idx),  batch_size=256, shuffle=False)

    member_feats     = extract_features(model, member_loader)
    nonmember_feats  = extract_features(model, nonmember_loader)

    X_eval = np.concatenate([member_feats, nonmember_feats])
    y_eval = np.concatenate([np.ones(len(member_feats)), np.zeros(len(nonmember_feats))])

    y_pred  = attack_clf.predict(X_eval)
    y_score = attack_clf.predict_proba(X_eval)[:, 1]

    acc = accuracy_score(y_eval, y_pred)
    auc = roc_auc_score(y_eval, y_score)

    print(f"  {target_model_name:<20} → Attack Acc: {acc:.4f}  AUC: {auc:.4f}")
    return {"exp_name": target_model_name, "attack_acc": acc, "attack_auc": auc}


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    print("="*55)
    print("影子模型成员推断攻击（Shadow Model MIA）")
    print("="*55)
    print(f"设备：{DEVICE}")

    full_set, test_set = load_data()

    # Step 1：用影子模型训练攻击分类器（与目标模型无关）
    print("\nStep 1：构建攻击分类器")
    attack_clf = build_attack_classifier(full_set)

    # Step 2：攻击所有目标模型
    print("\nStep 2：攻击目标模型")
    targets = ["baseline", "dp_noise0.5", "dp_noise1.0", "dp_noise1.5"]
    epsilons = {
        "baseline": None,
        "dp_noise0.5": 3.52,
        "dp_noise1.0": 0.24,
        "dp_noise1.5": 0.13,
    }

    results = []
    for name in targets:
        res = run_shadow_mia(name, attack_clf, full_set, test_set)
        if res:
            res["epsilon"] = epsilons[name]
            results.append(res)

    # Step 3：与简单阈值攻击对比（如果有结果文件的话）
    print("\n\n" + "="*55)
    print("✅ 影子模型攻击结果汇总")
    print("="*55)
    print(f"  {'模型':<22} {'ε':>8} {'攻击准确率':>12} {'AUC':>8}")
    print("─"*55)
    for r in results:
        eps_str = f"{r['epsilon']:.2f}" if r["epsilon"] else "N/A"
        print(f"  {r['exp_name']:<22} {eps_str:>8} {r['attack_acc']:>12.4f} {r['attack_auc']:>8.4f}")
    print("─"*55)

    # 保存结果
    import json
    shadow_file = os.path.join(RESULT_DIR, "shadow_mia_results.json")
    with open(shadow_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ 影子模型攻击结果已保存：{shadow_file}")

    # 简单对比图
    if len(results) >= 2:
        names  = [r["exp_name"].replace("_", "\n") for r in results]
        aucs   = [r["attack_auc"] for r in results]
        colors = ["#E74C3C" if r["epsilon"] is None else "#3498DB" for r in results]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(names, aucs, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1.5, label="随机基线=0.5")
        ax.set_ylabel("Attack AUC", fontsize=12)
        ax.set_title("影子模型攻击 AUC 对比\n（红=无DP，蓝=有DP）", fontsize=12, fontweight="bold")
        ax.set_ylim([0.4, 1.0])
        for bar, val in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fig_path = os.path.join(RESULT_DIR, "fig5_shadow_attack.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"✓ 图5已保存：{fig_path}")


if __name__ == "__main__":
    main()
