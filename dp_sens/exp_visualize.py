"""
exp_visualize.py - 实验结果可视化与对比分析

生成 6 张图表：
  图1：三模型 × 各保护方式 最终准确率热力图
  图2：数据脱敏（Laplace vs Gaussian）精度-隐私权衡曲线
  图3：梯度脱敏 精度-隐私权衡曲线（DP-SGD）
  图4：数据脱敏 vs 梯度脱敏 对比（相同 ε 下）
  图5：三模型训练曲线对比（无保护 vs 最强保护）
  图6：综合结论雷达图

运行方式：python exp_visualize.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

RESULT_DIR  = "results"
FIGURE_DIR  = os.path.join(RESULT_DIR, "figures")

# 中文字体
plt.rcParams["font.family"]       = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 颜色方案
MODEL_COLORS = {"MLP": "#E74C3C", "SimpleCNN": "#2980B9", "LeNet5": "#27AE60"}
PROTECT_COLORS = {
    "none":     "#2C3E50",
    "laplace":  "#E67E22",
    "gaussian": "#8E44AD",
    "gradient": "#2980B9",
}
PROTECT_LABELS = {
    "none":     "无保护",
    "laplace":  "Laplace数据脱敏",
    "gaussian": "Gaussian数据脱敏",
    "gradient": "DP-SGD梯度脱敏",
}

from dp_mechanisms import DISPLAY_NAMES


def load_results():
    path = os.path.join(RESULT_DIR, "all_results.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def filter_results(results, model=None, protect=None, condition=None):
    out = results
    if model:     out = [r for r in out if r["model"]     == model]
    if protect:   out = [r for r in out if r["protect"]   == protect]
    if condition: out = [r for r in out if r["condition"] == condition]
    return out


# ──────────────────────────────────────────────
# 图1：热力图（模型 × 保护方式 → 准确率）
# ──────────────────────────────────────────────
def plot_heatmap(results, save_dir):
    models     = ["MLP", "SimpleCNN", "LeNet5"]
    conditions = [r["condition"] for r in results if r["model"] == "MLP"]

    data = np.zeros((len(conditions), len(models)))
    for j, model in enumerate(models):
        for i, cond in enumerate(conditions):
            r = next((x for x in results
                      if x["model"] == model and x["condition"] == cond), None)
            if r:
                data[i, j] = r["final_acc"]

    fig, ax = plt.subplots(figsize=(9, 10))
    cmap = LinearSegmentedColormap.from_list("rg", ["#E74C3C", "#F39C12", "#27AE60"])
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=data.min() - 0.02, vmax=data.max())

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=12, fontweight="bold")
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels([DISPLAY_NAMES.get(c, c) for c in conditions], fontsize=10)

    # 在每个格子里标注数值
    for i in range(len(conditions)):
        for j in range(len(models)):
            val = data[i, j]
            color = "white" if val < (data.min() + data.max()) / 2 else "black"
            ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Test Accuracy")
    ax.set_title("各保护方式 × 各模型 最终测试准确率\n（绿色=高准确率，红色=低准确率）",
                 fontsize=13, fontweight="bold", pad=15)

    # 添加分隔线
    ax.axhline(0.5,  color="white", linewidth=2)  # 无保护 vs 数据脱敏
    ax.axhline(4.5,  color="white", linewidth=2)  # Laplace vs Gaussian
    ax.axhline(7.5,  color="white", linewidth=2)  # 数据脱敏 vs 梯度脱敏

    plt.tight_layout()
    path = os.path.join(save_dir, "fig1_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 图1（热力图）已保存：{path}")


# ──────────────────────────────────────────────
# 图2：数据脱敏精度-隐私权衡曲线
# ──────────────────────────────────────────────
def plot_data_sanitization_tradeoff(results, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, mech, title_label in zip(
        axes,
        ["laplace", "gaussian"],
        ["Laplace 机制", "Gaussian 机制"]
    ):
        for model_name, color in MODEL_COLORS.items():
            # 按 ε 从大到小排序（ε 越小隐私越强）
            dp_results = sorted(
                filter_results(results, model=model_name, protect=mech),
                key=lambda r: r["params"]["epsilon"], reverse=True
            )
            if not dp_results:
                continue

            epsilons = [r["params"]["epsilon"] for r in dp_results]
            accs     = [r["final_acc"]          for r in dp_results]

            # 加入 baseline 点（ε=∞ 对应无保护）
            baseline = next(
                (r["final_acc"] for r in results
                 if r["model"] == model_name and r["protect"] == "none"), None
            )
            if baseline:
                epsilons = [10.0] + epsilons  # 用 10 代表"无限大 ε"
                accs     = [baseline] + accs

            ax.plot(epsilons, accs, "o-", color=color, linewidth=2,
                    markersize=8, label=model_name)
            # 标注每个点的准确率
            for e, a in zip(epsilons, accs):
                ax.annotate(f"{a:.3f}", (e, a), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=8, color=color)

        ax.set_xlabel("隐私预算 ε（越小隐私保护越强）", fontsize=11)
        ax.set_ylabel("测试准确率", fontsize=11)
        ax.set_title(f"{title_label}\n精度-隐私权衡曲线", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 11])
        # 在 ε=10 处加标注
        ax.axvline(10, color="gray", linestyle=":", alpha=0.5)
        ax.text(10.1, ax.get_ylim()[0], "无保护", fontsize=8, color="gray",
                rotation=90, va="bottom")

    fig.suptitle("数据脱敏：精度-隐私权衡分析", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "fig2_data_sanitization_tradeoff.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 图2（数据脱敏权衡）已保存：{path}")


# ──────────────────────────────────────────────
# 图3：梯度脱敏精度-隐私权衡曲线
# ──────────────────────────────────────────────
def plot_gradient_sanitization_tradeoff(results, save_dir):
    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name, color in MODEL_COLORS.items():
        dp_results = sorted(
            filter_results(results, model=model_name, protect="gradient"),
            key=lambda r: r["params"]["noise_multiplier"]
        )
        if not dp_results:
            continue

        noises   = [r["params"]["noise_multiplier"] for r in dp_results]
        epsilons = [r["epsilon"]   for r in dp_results]
        accs     = [r["final_acc"] for r in dp_results]

        # 同时画两条轴的版本：主轴用噪声系数，标注 ε
        ax.plot(noises, accs, "o-", color=color, linewidth=2,
                markersize=9, label=model_name)
        for noise, eps, acc in zip(noises, epsilons, accs):
            eps_str = f"ε={eps:.2f}" if eps else ""
            ax.annotate(f"{acc:.3f}\n({eps_str})",
                        (noise, acc), textcoords="offset points",
                        xytext=(5, 5), ha="left", fontsize=8, color=color)

    # 画 baseline 横线
    for model_name, color in MODEL_COLORS.items():
        baseline = next(
            (r["final_acc"] for r in results
             if r["model"] == model_name and r["protect"] == "none"), None
        )
        if baseline:
            ax.axhline(baseline, color=color, linestyle="--",
                       alpha=0.4, linewidth=1)

    ax.set_xlabel("噪声系数 σ（越大隐私保护越强）", fontsize=11)
    ax.set_ylabel("测试准确率", fontsize=11)
    ax.set_title("梯度脱敏（DP-SGD）：精度-隐私权衡曲线\n（虚线为各模型无保护基线）",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "fig3_gradient_sanitization_tradeoff.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 图3（梯度脱敏权衡）已保存：{path}")


# ──────────────────────────────────────────────
# 图4：数据脱敏 vs 梯度脱敏 直接对比
# ──────────────────────────────────────────────
def plot_method_comparison(results, save_dir):
    """选取 ε≈2 附近的结果做横向对比"""
    models = ["MLP", "SimpleCNN", "LeNet5"]

    # 对比的4种条件
    compare_conditions = [
        ("no_dp",          "无保护",       "#95A5A6"),
        ("laplace_eps2",   "Laplace ε=2",  "#E67E22"),
        ("gaussian_eps2",  "Gaussian ε=2", "#8E44AD"),
        ("grad_noise1.0",  "梯度DP σ=1.0", "#2980B9"),
    ]

    x      = np.arange(len(models))
    width  = 0.18
    fig, ax = plt.subplots(figsize=(11, 6))

    for i, (cond, label, color) in enumerate(compare_conditions):
        accs = []
        for model in models:
            r = next((res for res in results
                      if res["model"] == model and res["condition"] == cond), None)
            accs.append(r["final_acc"] if r else 0)

        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, accs, width, label=label,
                      color=color, alpha=0.85, edgecolor="black", linewidth=0.5)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2.,
                    bar.get_height() + 0.002,
                    f"{acc:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylabel("测试准确率", fontsize=12)
    ax.set_title("数据脱敏 vs 梯度脱敏：相近隐私强度下的准确率对比\n"
                 "（Laplace/Gaussian ε=2，梯度DP σ=1.0）",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    ax.set_ylim([0.8, 1.02])
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "fig4_method_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 图4（方法对比）已保存：{path}")


# ──────────────────────────────────────────────
# 图5：训练曲线对比（无保护 vs 最强保护）
# ──────────────────────────────────────────────
def plot_training_curves(results, save_dir):
    models = ["MLP", "SimpleCNN", "LeNet5"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    strong_conditions = [
        ("no_dp",         "无保护",          "#2C3E50", "-"),
        ("laplace_eps1",  "Laplace ε=1",     "#E67E22", "--"),
        ("gaussian_eps1", "Gaussian ε=1",    "#8E44AD", "-."),
        ("grad_noise1.5", "梯度DP σ=1.5",    "#2980B9", ":"),
    ]

    for ax, model_name in zip(axes, models):
        for cond, label, color, ls in strong_conditions:
            r = next((res for res in results
                      if res["model"] == model_name and res["condition"] == cond), None)
            if r and r.get("acc_curve"):
                epochs = range(1, len(r["acc_curve"]) + 1)
                ax.plot(epochs, r["acc_curve"], color=color, linestyle=ls,
                        linewidth=2, label=label)

        ax.set_title(f"{model_name}\n训练曲线对比", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Test Accuracy", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("各模型：无保护 vs 强隐私保护 训练过程对比",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(save_dir, "fig5_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 图5（训练曲线）已保存：{path}")


# ──────────────────────────────────────────────
# 图6：精度损失汇总柱状图（相对 baseline 的下降）
# ──────────────────────────────────────────────
def plot_accuracy_drop(results, save_dir):
    """展示各保护方式相对于无保护 baseline 的精度下降"""
    models = ["MLP", "SimpleCNN", "LeNet5"]

    # 选取代表性条件
    compare = [
        ("laplace_eps5",  "Laplace\nε=5",  "#F39C12"),
        ("laplace_eps2",  "Laplace\nε=2",  "#E67E22"),
        ("laplace_eps1",  "Laplace\nε=1",  "#E74C3C"),
        ("gaussian_eps5", "Gaussian\nε=5", "#BB8FCE"),
        ("gaussian_eps2", "Gaussian\nε=2", "#8E44AD"),
        ("gaussian_eps1", "Gaussian\nε=1", "#6C3483"),
        ("grad_noise0.5", "梯度DP\nσ=0.5", "#7FB3D3"),
        ("grad_noise1.0", "梯度DP\nσ=1.0", "#2980B9"),
        ("grad_noise1.5", "梯度DP\nσ=1.5", "#1A5276"),
    ]

    # 计算每个模型的 baseline 准确率
    baselines = {
        m: next((r["final_acc"] for r in results
                 if r["model"] == m and r["protect"] == "none"), None)
        for m in models
    }

    x      = np.arange(len(compare))
    width  = 0.25
    fig, ax = plt.subplots(figsize=(14, 6))

    for j, model in enumerate(models):
        drops = []
        for cond, _, _ in compare:
            r = next((res for res in results
                      if res["model"] == model and res["condition"] == cond), None)
            drop = (baselines[model] - r["final_acc"]) * 100 if r else 0
            drops.append(max(drop, 0))   # 负值（精度提升）按 0 处理

        offset = (j - 1) * width
        color  = list(MODEL_COLORS.values())[j]
        bars   = ax.bar(x + offset, drops, width, label=model,
                        color=color, alpha=0.8, edgecolor="black", linewidth=0.4)

    labels = [label for _, label, _ in compare]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("精度下降（%）", fontsize=12)
    ax.set_title("各保护方式导致的精度下降（相对无保护 baseline）\n"
                 "（越低越好——说明隐私保护代价小）",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # 添加分组标注
    ax.axvline(2.5,  color="gray", linestyle="--", alpha=0.4)
    ax.axvline(5.5,  color="gray", linestyle="--", alpha=0.4)
    ax.text(1,   ax.get_ylim()[1] * 0.95, "Laplace 机制",  ha="center", fontsize=9, color="gray")
    ax.text(4,   ax.get_ylim()[1] * 0.95, "Gaussian 机制", ha="center", fontsize=9, color="gray")
    ax.text(7,   ax.get_ylim()[1] * 0.95, "梯度DP",        ha="center", fontsize=9, color="gray")

    plt.tight_layout()
    path = os.path.join(save_dir, "fig6_accuracy_drop.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ 图6（精度损失）已保存：{path}")


# ──────────────────────────────────────────────
# 文字分析报告
# ──────────────────────────────────────────────
def generate_text_report(results):
    models = ["MLP", "SimpleCNN", "LeNet5"]

    print("\n\n" + "="*65)
    print("📊 实验分析报告")
    print("="*65)

    # 1. 各模型最优结果
    print("\n【1】各模型最优准确率（无保护 baseline）")
    for m in models:
        r = next((res for res in results
                  if res["model"] == m and res["protect"] == "none"), None)
        if r:
            print(f"  {m:<12}: {r['final_acc']:.4f}")

    # 2. 精度损失分析
    print("\n【2】各保护方式平均精度损失（三模型平均）")
    protect_groups = {
        "Laplace 机制":  ["laplace_eps5",  "laplace_eps2",  "laplace_eps1"],
        "Gaussian 机制": ["gaussian_eps5", "gaussian_eps2", "gaussian_eps1"],
        "梯度DP (DP-SGD)": ["grad_noise0.5", "grad_noise1.0", "grad_noise1.5"],
    }
    for group_name, conds in protect_groups.items():
        drops = []
        for cond in conds:
            for m in models:
                baseline = next((r["final_acc"] for r in results
                                 if r["model"] == m and r["protect"] == "none"), None)
                r = next((res for res in results
                          if res["model"] == m and res["condition"] == cond), None)
                if r and baseline:
                    drops.append(baseline - r["final_acc"])
        if drops:
            avg_drop = np.mean(drops) * 100
            max_drop = np.max(drops) * 100
            print(f"  {group_name:<20}: 平均下降 {avg_drop:.2f}%，最大下降 {max_drop:.2f}%")

    # 3. 最优隐私-效用平衡点
    print("\n【3】推荐配置（最优隐私-效用平衡）")
    for m in models:
        best_r    = None
        best_score = -1
        for r in results:
            if r["model"] != m or r["protect"] == "none":
                continue
            baseline = next((res["final_acc"] for res in results
                             if res["model"] == m and res["protect"] == "none"), 1.0)
            # 评分：准确率保留率 / 0.8（归一化）
            acc_retain = r["final_acc"] / baseline
            if acc_retain > best_score:
                best_score = acc_retain
                best_r = r
        if best_r:
            tag = DISPLAY_NAMES.get(best_r["condition"], best_r["condition"])
            print(f"  {m:<12}: 推荐 [{tag}]，保留准确率 {best_score:.2%}")

    print("\n" + "="*65)


def main():
    os.makedirs(FIGURE_DIR, exist_ok=True)
    print("加载实验结果...")
    results = load_results()
    print(f"共 {len(results)} 组实验结果\n")
    print("生成图表：")

    plot_heatmap(results, FIGURE_DIR)
    plot_data_sanitization_tradeoff(results, FIGURE_DIR)
    plot_gradient_sanitization_tradeoff(results, FIGURE_DIR)
    plot_method_comparison(results, FIGURE_DIR)
    plot_training_curves(results, FIGURE_DIR)
    plot_accuracy_drop(results, FIGURE_DIR)
    generate_text_report(results)

    print(f"\n✅ 所有图表已保存至：{FIGURE_DIR}/")


if __name__ == "__main__":
    main()
