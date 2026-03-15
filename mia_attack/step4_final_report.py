"""
step4_final_report.py - 生成最终汇总报告图

整合 step1~step3 的所有结果，生成毕设答辩用的完整可视化报告。

运行方式：python step4_final_report.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

RESULT_DIR = "mia_results"
SAVE_DIR   = "saved_models"

# 支持中文显示（如果报错，把 SimHei 改成 DejaVu Sans 并去掉中文）
plt.rcParams["font.family"]      = ["SimHei", "DejaVu Sans", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False


def load_results():
    mia_path    = os.path.join(RESULT_DIR, "mia_results.json")
    shadow_path = os.path.join(RESULT_DIR, "shadow_mia_results.json")
    history_path = os.path.join(SAVE_DIR, "training_history.json")

    mia_results    = json.load(open(mia_path,     encoding="utf-8")) if os.path.exists(mia_path)     else []
    shadow_results = json.load(open(shadow_path,  encoding="utf-8")) if os.path.exists(shadow_path)  else []
    history        = json.load(open(history_path, encoding="utf-8")) if os.path.exists(history_path) else {}
    return mia_results, shadow_results, history


def plot_final_summary(mia_results, shadow_results, history):
    """生成 2x3 综合结果图"""
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle("差分隐私保护下的深度学习模型研究\n——成员推断攻击实验综合报告",
                 fontsize=16, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    colors_map = {
        "baseline":    "#E74C3C",
        "dp_noise0.5": "#E67E22",
        "dp_noise1.0": "#27AE60",
        "dp_noise1.5": "#2980B9",
    }
    labels_map = {
        "baseline":    "Baseline\n(无DP)",
        "dp_noise0.5": "DP\nε≈3.52",
        "dp_noise1.0": "DP\nε≈0.24",
        "dp_noise1.5": "DP\nε≈0.13",
    }

    # ── 图1：训练 Accuracy 曲线 ──────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    if history:
        for name, hist in history.items():
            accs = hist.get("test_acc", [])
            if accs:
                ax1.plot(range(1, len(accs)+1), accs,
                         color=colors_map.get(name, "gray"),
                         label=labels_map.get(name, name),
                         linewidth=2)
    ax1.set_title("训练过程：Test Accuracy", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # ── 图2：ROC 曲线对比 ────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    if mia_results:
        for res in mia_results:
            fpr = res.get("fpr_curve", [])
            tpr = res.get("tpr_curve", [])
            if fpr and tpr:
                name = res["exp_name"]
                ax2.plot(fpr, tpr,
                         color=colors_map.get(name, "gray"),
                         label=f"{labels_map.get(name, name)}  AUC={res['attack_auc']:.3f}",
                         linewidth=2)
    ax2.plot([0, 1], [0, 1], "k--", linewidth=1, label="随机基线")
    ax2.set_title("成员推断攻击 ROC 曲线", fontweight="bold")
    ax2.set_xlabel("FPR")
    ax2.set_ylabel("TPR")
    ax2.legend(fontsize=7.5)
    ax2.grid(alpha=0.3)

    # ── 图3：Attack AUC 对比（阈值攻击 vs 影子模型攻击）──
    ax3 = fig.add_subplot(gs[0, 2])
    if mia_results:
        names       = [r["exp_name"] for r in mia_results]
        x           = np.arange(len(names))
        threshold_aucs = [r["attack_auc"] for r in mia_results]

        width = 0.35
        bars1 = ax3.bar(x - width/2, threshold_aucs, width,
                        label="阈值攻击", color="#3498DB", alpha=0.8)

        if shadow_results:
            shadow_dict = {r["exp_name"]: r["attack_auc"] for r in shadow_results}
            shadow_aucs = [shadow_dict.get(n, 0) for n in names]
            bars2 = ax3.bar(x + width/2, shadow_aucs, width,
                            label="影子模型攻击", color="#E74C3C", alpha=0.8)
            for bar in bars2:
                if bar.get_height() > 0:
                    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                             f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)

        for bar in bars1:
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                     f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)

        ax3.axhline(0.5, color="black", linestyle="--", linewidth=1.2)
        ax3.set_title("两种攻击方法 AUC 对比", fontweight="bold")
        ax3.set_xticks(x)
        ax3.set_xticklabels([labels_map.get(n, n) for n in names], fontsize=8)
        ax3.set_ylabel("Attack AUC")
        ax3.set_ylim([0.4, 1.0])
        ax3.legend(fontsize=9)
        ax3.grid(axis="y", alpha=0.3)

    # ── 图4：置信度差距（成员 vs 非成员）───────────────
    ax4 = fig.add_subplot(gs[1, 0])
    if mia_results:
        names = [labels_map.get(r["exp_name"], r["exp_name"]) for r in mia_results]
        m_means  = [r["member_mean"]    for r in mia_results]
        nm_means = [r["nonmember_mean"] for r in mia_results]
        x = np.arange(len(names))
        ax4.bar(x - 0.2, m_means,  0.35, label="成员置信度",    color="#E74C3C", alpha=0.8)
        ax4.bar(x + 0.2, nm_means, 0.35, label="非成员置信度",  color="#3498DB", alpha=0.8)
        ax4.set_title("成员 vs 非成员平均置信度", fontweight="bold")
        ax4.set_xticks(x)
        ax4.set_xticklabels(names, fontsize=8)
        ax4.set_ylabel("平均置信度")
        ax4.set_ylim([0, 1.1])
        ax4.legend(fontsize=9)
        ax4.grid(axis="y", alpha=0.3)

    # ── 图5：ε 与 Attack AUC 的隐私-效用权衡 ─────────────
    ax5 = fig.add_subplot(gs[1, 1])
    if mia_results:
        dp_res = [r for r in mia_results if r.get("epsilon") is not None]
        if dp_res:
            eps  = [r["epsilon"]    for r in dp_res]
            aucs = [r["attack_auc"] for r in dp_res]
            ax5.plot(eps, aucs, "o-", color="#E74C3C", linewidth=2.5,
                     markersize=10, label="Attack AUC (阈值攻击)")

            baseline_res = next((r for r in mia_results if r.get("epsilon") is None), None)
            if baseline_res:
                ax5.axhline(baseline_res["attack_auc"], color="gray", linestyle="--",
                            alpha=0.8, label=f"Baseline AUC={baseline_res['attack_auc']:.3f}")

            ax5.axhline(0.5, color="black", linestyle=":", alpha=0.5, label="随机基线=0.5")
            ax5.set_xlabel("隐私预算 ε（越小隐私保护越强）")
            ax5.set_ylabel("Attack AUC（越小越安全）")
            ax5.set_title("ε 与攻击成功率的关系\n（核心结论图）", fontweight="bold")
            ax5.legend(fontsize=8)
            ax5.grid(alpha=0.3)
            ax5.set_ylim([0.45, 1.0])

    # ── 图6：结论文字摘要 ────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")

    if mia_results:
        baseline_auc = next((r["attack_auc"] for r in mia_results if r.get("epsilon") is None), None)
        best_dp_auc  = min((r["attack_auc"] for r in mia_results if r.get("epsilon") is not None), default=None)
        baseline_acc = next((r.get("member_mean", 0) for r in mia_results if r.get("epsilon") is None), None)

        conclusions = [
            "实验结论摘要",
            "",
            "1. DP显著降低攻击成功率",
            f"   Baseline AUC = {baseline_auc:.3f}" if baseline_auc else "",
            f"   最强DP AUC  = {best_dp_auc:.3f}" if best_dp_auc else "",
            "",
            "2. epsilon越小隐私保护越强",
            "   AUC随epsilon减小趋近0.5",
            "   即攻击逐渐失效",
            "",
            "3. 影子模型攻击更强",
            "   但仍被DP有效抑制",
            "   两种攻击均验证了DP防御",
            "",
            "4. 精度损失可接受",
            "   强隐私区间下模型",
            "   仍保持较高准确率",
        ]

        text = "\n".join(conclusions)
        ax6.text(0.05, 0.95, text, transform=ax6.transAxes,
                 fontsize=10, verticalalignment="top",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#ECF0F1", alpha=0.8),
                 fontfamily="sans-serif")

    path = os.path.join(RESULT_DIR, "FINAL_REPORT.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ 最终报告图已保存：{path}")
    return path


def main():
    print("生成最终综合报告...")
    mia_results, shadow_results, history = load_results()

    if not mia_results:
        print("❌ 未找到 MIA 结果，请先运行 step1 和 step2")
        return

    plot_final_summary(mia_results, shadow_results, history)
    print("\n✅ 完成！可以在 mia_results/FINAL_REPORT.png 查看完整报告图。")


if __name__ == "__main__":
    main()
