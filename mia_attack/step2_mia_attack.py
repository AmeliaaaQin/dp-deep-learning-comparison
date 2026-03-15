"""
step2_mia_attack.py - 成员推断攻击（Membership Inference Attack）

原理：训练过的模型对"训练集样本"的预测置信度高于"测试集样本"。
      攻击者利用这一差异，判断某条数据是否参与了训练。

方法：Yeom et al. (2018) 基于置信度阈值的成员推断攻击

运行方式：python step2_mia_attack.py
输出：mia_results/ 目录下的结果文件 + 可视化图表
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)
import matplotlib
matplotlib.use("Agg")  # 无显示器环境
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import matplotlib.pyplot as plt
import matplotlib

from models import SimpleCNN_MNIST

# ─── 配置 ────────────────────────────────────────────────
SAVE_DIR    = "saved_models"
RESULT_DIR  = "mia_results"
DATA_DIR    = "./data"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_ATTACK    = 2000   # 攻击样本数量（成员+非成员各一半）

EXPERIMENTS = [
    ("baseline",    None),
    ("dp_noise0.5", 9.48),   # 实际训练结果
    ("dp_noise1.0", 1.11),
    ("dp_noise1.5", 0.58),
]
# ─────────────────────────────────────────────────────────


def load_datasets():
    """加载 MNIST 训练集和测试集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=transform)
    test_set  = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
    return train_set, test_set


def build_attack_dataset(train_set, test_set, is_baseline=False, n=N_ATTACK):
    """
    统一攻击数据集构建（与 step1_final.py 对应）：
      - 成员   = 训练集前 5000 条（所有模型都用这些数据训练的）
      - 非成员 = 训练集索引 5000~10000（从未参与训练，但分布完全相同）
    is_baseline 参数保留但不再影响逻辑，所有模型统一处理。
    """
    TRAIN_SIZE = 5000
    n_half = n // 2
    rng    = np.random.default_rng(42)

    member_indices     = rng.choice(range(0, TRAIN_SIZE),
                                    n_half, replace=False).tolist()
    non_member_indices = rng.choice(range(TRAIN_SIZE, TRAIN_SIZE * 2),
                                    n_half, replace=False).tolist()

    member_loader     = DataLoader(Subset(train_set, member_indices),
                                   batch_size=512, shuffle=False)
    non_member_loader = DataLoader(Subset(train_set, non_member_indices),
                                   batch_size=512, shuffle=False)
    return member_loader, non_member_loader


def get_confidence_scores(model, loader):
    """
    获取模型对 loader 中每个样本的置信度（正确类别的 softmax 概率）
    """
    model.eval()
    confidences = []
    true_labels = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            probs = F.softmax(model(x), dim=1)          # [B, 10]
            # 取预测为各自真实类别的置信度
            conf = probs[torch.arange(len(y)), y]       # [B]
            confidences.append(conf.cpu().numpy())
            true_labels.append(y.cpu().numpy())

    return np.concatenate(confidences), np.concatenate(true_labels)


def find_optimal_threshold(member_scores, non_member_scores):
    """用 ROC 曲线找最优阈值（Youden's J statistic）"""
    scores = np.concatenate([member_scores, non_member_scores])
    labels = np.concatenate([
        np.ones(len(member_scores)),
        np.zeros(len(non_member_scores))
    ])
    fpr, tpr, thresholds = roc_curve(labels, scores)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx], fpr, tpr


def run_mia(exp_name, epsilon, train_set, test_set):
    """对单个模型运行成员推断攻击"""
    print(f"\n{'─'*50}")
    print(f"攻击模型：{exp_name}  (ε = {epsilon if epsilon else 'N/A'})")
    print(f"{'─'*50}")

    # 1. 加载模型
    model_path = os.path.join(SAVE_DIR, f"{exp_name}.pt")
    if not os.path.exists(model_path):
        print(f"  ⚠ 未找到模型文件：{model_path}，跳过")
        return None

    model = SimpleCNN_MNIST().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # 2. 构建攻击数据集
    is_baseline = (epsilon is None)
    member_loader, non_member_loader = build_attack_dataset(
        train_set, test_set, is_baseline=is_baseline
    )

    # 3. 获取置信度分数
    member_scores,     _ = get_confidence_scores(model, member_loader)
    non_member_scores, _ = get_confidence_scores(model, non_member_loader)

    print(f"  成员平均置信度：    {member_scores.mean():.4f} ± {member_scores.std():.4f}")
    print(f"  非成员平均置信度：  {non_member_scores.mean():.4f} ± {non_member_scores.std():.4f}")

    # 4. 构建攻击标签和分数
    attack_scores = np.concatenate([member_scores, non_member_scores])
    attack_labels = np.concatenate([
        np.ones(len(member_scores), dtype=int),
        np.zeros(len(non_member_scores), dtype=int)
    ])

    # 5. 找最优阈值并预测
    threshold, fpr_curve, tpr_curve = find_optimal_threshold(member_scores, non_member_scores)
    attack_preds = (attack_scores >= threshold).astype(int)

    # 6. 计算指标
    attack_acc = accuracy_score(attack_labels, attack_preds)
    attack_auc = roc_auc_score(attack_labels, attack_scores)
    # 计算 TPR@FPR=0.1（评估低假阳率下的攻击能力）
    target_fpr = 0.1
    tpr_at_low_fpr = np.interp(target_fpr, fpr_curve, tpr_curve)

    print(f"\n  ── 攻击结果 ──")
    print(f"  最优阈值：          {threshold:.4f}")
    print(f"  Attack Accuracy：   {attack_acc:.4f}  (随机基线=0.5000)")
    print(f"  Attack AUC：        {attack_auc:.4f}  (随机基线=0.5000)")
    print(f"  TPR@FPR=0.1：       {tpr_at_low_fpr:.4f}")

    result = {
        "exp_name":       exp_name,
        "epsilon":        epsilon,
        "threshold":      float(threshold),
        "attack_acc":     float(attack_acc),
        "attack_auc":     float(attack_auc),
        "tpr_at_fpr01":   float(tpr_at_low_fpr),
        "member_mean":    float(member_scores.mean()),
        "nonmember_mean": float(non_member_scores.mean()),
        "fpr_curve":      fpr_curve.tolist(),
        "tpr_curve":      tpr_curve.tolist(),
        "member_scores":  member_scores.tolist(),
        "nonmember_scores": non_member_scores.tolist(),
    }
    return result


def plot_confidence_distribution(results, save_dir):
    """图1：置信度分布对比（成员 vs 非成员）"""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    colors_m  = "#E74C3C"   # 成员：红色
    colors_nm = "#3498DB"   # 非成员：蓝色

    for ax, res in zip(axes, results):
        m  = np.array(res["member_scores"])
        nm = np.array(res["nonmember_scores"])
        ax.hist(m,  bins=50, alpha=0.6, color=colors_m,  label="Member (训练集)",    density=True)
        ax.hist(nm, bins=50, alpha=0.6, color=colors_nm, label="Non-Member (测试集)", density=True)
        ax.axvline(res["threshold"], color="black", linestyle="--", linewidth=1.2, label=f"阈值={res['threshold']:.3f}")

        eps_str = f"ε={res['epsilon']}" if res["epsilon"] else "Baseline"
        ax.set_title(f"{res['exp_name']}\n({eps_str})", fontsize=11)
        ax.set_xlabel("预测置信度", fontsize=10)
        ax.set_ylabel("密度", fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle("成员推断攻击：置信度分布对比", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "fig1_confidence_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 图1已保存：{path}")


def plot_roc_curves(results, save_dir):
    """图2：各模型的 ROC 曲线对比"""
    fig, ax = plt.subplots(figsize=(7, 6))

    colors = ["#E74C3C", "#E67E22", "#27AE60", "#2980B9"]
    for res, color in zip(results, colors):
        eps_str = f"ε={res['epsilon']}" if res["epsilon"] else "Baseline(无DP)"
        label = f"{eps_str}  AUC={res['attack_auc']:.3f}"
        ax.plot(res["fpr_curve"], res["tpr_curve"], color=color, linewidth=2, label=label)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="随机猜测 (AUC=0.5)")
    ax.set_xlabel("假阳性率 (FPR)", fontsize=12)
    ax.set_ylabel("真阳性率 (TPR)", fontsize=12)
    ax.set_title("成员推断攻击 ROC 曲线\n（越靠近随机基线，隐私保护越好）", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3)

    path = os.path.join(save_dir, "fig2_roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 图2已保存：{path}")


def plot_privacy_utility_tradeoff(results, save_dir):
    """图3：隐私-效用权衡曲线（核心结果图）"""
    dp_results = [r for r in results if r["epsilon"] is not None]

    epsilons   = [r["epsilon"]    for r in dp_results]
    attack_auc = [r["attack_auc"] for r in dp_results]

    fig, ax1 = plt.subplots(figsize=(7, 5))

    # 左轴：Attack AUC
    color1 = "#E74C3C"
    ax1.set_xlabel("隐私预算 ε（越小越隐私）", fontsize=12)
    ax1.set_ylabel("攻击 AUC（越低越安全）", fontsize=12, color=color1)
    line1, = ax1.plot(epsilons, attack_auc, "o-", color=color1,
                       linewidth=2.5, markersize=9, label="Attack AUC")
    ax1.axhline(0.5, color=color1, linestyle=":", alpha=0.5, label="随机基线 (AUC=0.5)")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim([0.45, 1.0])

    # 标注 baseline
    baseline = next(r for r in results if r["epsilon"] is None)
    ax1.axhline(baseline["attack_auc"], color="gray", linestyle="--",
                alpha=0.7, label=f"Baseline AUC={baseline['attack_auc']:.3f}")

    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)

    fig.suptitle("隐私预算 ε 与攻击成功率的权衡\n（验证 DP 对成员推断攻击的防御效果）",
                 fontsize=12, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(save_dir, "fig3_privacy_utility_tradeoff.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 图3已保存：{path}")


def plot_summary_bar(results, save_dir):
    """图4：攻击准确率对比柱状图"""
    labels   = [r["exp_name"].replace("_", "\n") for r in results]
    acc_vals = [r["attack_acc"] for r in results]
    auc_vals = [r["attack_auc"] for r in results]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, acc_vals, width, label="Attack Accuracy", color="#3498DB", alpha=0.8)
    bars2 = ax.bar(x + width/2, auc_vals, width, label="Attack AUC",      color="#E74C3C", alpha=0.8)

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.2, label="随机基线 = 0.5")
    ax.set_ylabel("攻击指标", fontsize=12)
    ax.set_title("各模型成员推断攻击效果对比\n（越接近0.5说明DP防御越有效）", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim([0.4, 1.0])
    ax.legend(fontsize=10)

    # 在柱子上标注数值
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "fig4_attack_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 图4已保存：{path}")


def print_final_table(results):
    """打印最终汇总表格"""
    print("\n\n" + "="*70)
    print("✅ 成员推断攻击实验汇总")
    print("="*70)
    print(f"  {'模型':<20} {'ε':>8} {'攻击准确率':>12} {'AUC':>8} {'成员置信度':>12} {'非成员置信度':>14}")
    print("─"*70)
    for r in results:
        eps_str = f"{r['epsilon']:.2f}" if r["epsilon"] else "N/A"
        print(f"  {r['exp_name']:<20} {eps_str:>8} {r['attack_acc']:>12.4f} "
              f"{r['attack_auc']:>8.4f} {r['member_mean']:>12.4f} {r['nonmember_mean']:>14.4f}")
    print("─"*70)
    print("  参考：Attack Acc/AUC = 0.5 代表攻击完全失败（隐私保护最强）")
    print("        Attack Acc/AUC = 1.0 代表攻击完全成功（隐私完全泄露）")


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    print(f"使用设备：{DEVICE}")
    print(f"每类攻击样本数：{N_ATTACK // 2}（成员 + 非成员各半）\n")

    # 加载数据集
    train_set, test_set = load_datasets()

    # 对每个模型运行攻击
    all_results = []
    for exp_name, epsilon in EXPERIMENTS:
        result = run_mia(exp_name, epsilon, train_set, test_set)
        if result:
            all_results.append(result)

    if not all_results:
        print("\n❌ 没有找到任何模型文件，请先运行 step1_train_models.py")
        return

    # 保存原始结果
    save_results = []
    for r in all_results:
        save_r = {k: v for k, v in r.items() if k not in ("fpr_curve", "tpr_curve",
                                                            "member_scores", "nonmember_scores")}
        save_r["fpr_curve"] = r["fpr_curve"][:50]   # 只保存部分点
        save_r["tpr_curve"] = r["tpr_curve"][:50]
        save_results.append(save_r)

    with open(os.path.join(RESULT_DIR, "mia_results.json"), "w", encoding="utf-8") as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)

    # 生成所有图表
    print(f"\n生成图表...")
    plot_confidence_distribution(all_results, RESULT_DIR)
    plot_roc_curves(all_results, RESULT_DIR)
    plot_privacy_utility_tradeoff(all_results, RESULT_DIR)
    plot_summary_bar(all_results, RESULT_DIR)

    # 打印汇总表
    print_final_table(all_results)

    print(f"\n✓ 所有结果已保存至：{RESULT_DIR}/")


if __name__ == "__main__":
    main()
