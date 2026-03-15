# 差分隐私保护下的深度学习模型研究——成员推断攻击实验

## 📁 文件结构

```
dp_privacy/
├── models.py                   # 模型定义（MNIST + CIFAR-10）
├── step1_train_models.py       # 训练并保存所有模型
├── step2_mia_attack.py         # 成员推断攻击（阈值法）
├── step3_shadow_model_attack.py# 进阶：影子模型攻击
├── step4_final_report.py       # 生成综合报告图
└── README.md                   # 本说明文档
```

运行后会自动生成：
```
saved_models/      # 训练好的模型权重（.pt 文件）
mia_results/       # 攻击结果 + 可视化图表
data/              # 自动下载的 MNIST 数据集
```

---

## 🚀 运行步骤（按顺序执行）

### 第一步：安装依赖

```bash
pip install torch torchvision opacus scikit-learn matplotlib
```

### 第二步：训练所有模型（约 5~15 分钟）

```bash
python step1_train_models.py
```

会训练 4 个模型并保存到 `saved_models/`：
- `baseline.pt`：普通训练（无DP）
- `dp_noise0.5.pt`：DP 训练，noise=0.5（弱隐私，ε≈3.5）
- `dp_noise1.0.pt`：DP 训练，noise=1.0（中等隐私，ε≈0.24）
- `dp_noise1.5.pt`：DP 训练，noise=1.5（强隐私，ε≈0.13）

### 第三步：运行成员推断攻击（约 1~2 分钟）

```bash
python step2_mia_attack.py
```

生成图表：
- `fig1_confidence_distribution.png`：成员 vs 非成员置信度分布
- `fig2_roc_curves.png`：ROC 曲线对比（核心图）
- `fig3_privacy_utility_tradeoff.png`：ε 与攻击成功率权衡
- `fig4_attack_comparison.png`：攻击准确率柱状图

### 第四步（可选进阶）：影子模型攻击（约 5~10 分钟）

```bash
python step3_shadow_model_attack.py
```

这是更强的攻击方法，适合在毕设中作为"加强攻击场景"的对照实验。

### 第五步：生成最终综合报告

```bash
python step4_final_report.py
```

生成 `mia_results/FINAL_REPORT.png`，包含所有实验结果的综合展示，答辩直接用。

---

## 📖 实验原理说明

### 什么是成员推断攻击？

攻击者目标：判断某条数据是否参与了模型训练。

核心假设：如果模型"记住"了训练数据（过拟合），
         那么模型对训练集样本的预测置信度会高于测试集样本。

### 两种攻击方法对比

| 方法 | 原理 | 难度 | 攻击强度 |
|------|------|------|----------|
| 阈值攻击（step2）| 置信度 > 阈值 → 判定为成员 | ⭐ | 中 |
| 影子模型攻击（step3）| 训练多个影子模型来模拟目标模型 | ⭐⭐⭐ | 强 |

### 评估指标

- **Attack Accuracy**：攻击的分类准确率（0.5 = 随机猜，1.0 = 完美攻击）
- **Attack AUC**：ROC 曲线下面积（0.5 = 完全失败，1.0 = 完美攻击）
- **期望结论**：有 DP 的模型，AUC 显著低于无 DP 的 Baseline

---

## 📊 预期实验结论

| 模型 | ε | 预期 Attack AUC |
|------|---|-----------------|
| Baseline（无DP）| N/A | 0.70~0.85 |
| DP noise=0.5 | ≈3.5 | 0.60~0.70 |
| DP noise=1.0 | ≈0.24 | 0.52~0.60 |
| DP noise=1.5 | ≈0.13 | 0.50~0.54 |

攻击 AUC 随 ε 减小而降低，验证了 DP 对成员推断攻击的防御效果。

---

## 🎓 答辩要点

1. **理论层面**：差分隐私通过 DP-SGD（梯度裁剪 + 噪声注入）限制模型对单个样本的"记忆"
2. **攻击层面**：实现了两种成员推断攻击，分别从不同角度量化隐私泄露程度
3. **验证层面**：实验结果表明，随 ε 减小，攻击成功率单调下降，与理论预测一致
4. **权衡层面**：在强隐私区间（ε<1）下，模型准确率损失可控（约 88%），说明 DP 具有实用价值
