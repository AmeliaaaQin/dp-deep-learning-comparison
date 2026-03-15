# 差分隐私脱敏预处理实验

## 实验设计说明

### 核心问题
> 在深度学习训练流程中，对**数据**进行差分隐私脱敏 vs 对**梯度**进行差分隐私脱敏，
> 哪种方式在相同隐私强度下能保留更好的模型精度？

### 实验矩阵（共 30 组）

```
3 个模型 × 10 种条件 = 30 组实验

模型：MLP / SimpleCNN / LeNet5

保护条件：
  ① 无保护（baseline）                    ←  1 组
  ② Laplace 数据脱敏  ε = 5 / 2 / 1      ←  3 组
  ③ Gaussian 数据脱敏 ε = 5 / 2 / 1      ←  3 组
  ④ 梯度脱敏 DP-SGD   σ = 0.5 / 1.0 / 1.5 ← 3 组
```

### 两种脱敏方式的本质区别

| 对比项 | 数据脱敏 | 梯度脱敏（DP-SGD） |
|--------|---------|------------------|
| 加噪时机 | 训练前，加在输入数据上 | 训练中，加在梯度上 |
| 影响范围 | 每个 epoch 都加噪 | 每次反向传播加噪 |
| 隐私保证 | 局部差分隐私（LDP） | （ε,δ）-差分隐私 |
| 计算开销 | 几乎为零 | 梯度裁剪有额外开销 |
| 理论基础 | Laplace/Gaussian 机制 | 矩会计（Moments Accountant）|

---

## 文件说明

```
dp_sens/
├── models_dp.py       # MLP / SimpleCNN / LeNet5 模型定义
├── dp_mechanisms.py   # Laplace / Gaussian / DP-SGD 脱敏实现
├── exp_main.py        # 主实验（训练30组，约30~60分钟）
├── exp_visualize.py   # 可视化与分析报告（生成6张图）
└── README.md          # 本说明
```

---

## 运行步骤

### 1. 安装依赖
```bash
pip install torch torchvision opacus scikit-learn matplotlib numpy
```

### 2. 运行主实验（约 30~60 分钟，CPU）
```bash
python exp_main.py
```
输出：`results/all_results.json`

### 3. 生成可视化报告
```bash
python exp_visualize.py
```
输出：`results/figures/` 下 6 张图

---

## 生成图表说明

| 图 | 文件名 | 内容 |
|----|--------|------|
| 图1 | fig1_heatmap.png | 热力图：所有模型×保护方式的准确率总览 |
| 图2 | fig2_data_sanitization_tradeoff.png | Laplace vs Gaussian 精度-隐私权衡曲线 |
| 图3 | fig3_gradient_sanitization_tradeoff.png | DP-SGD 精度-隐私权衡曲线 |
| 图4 | fig4_method_comparison.png | 数据脱敏 vs 梯度脱敏 直接对比柱状图 |
| 图5 | fig5_training_curves.png | 三模型训练曲线（无保护 vs 最强保护）|
| 图6 | fig6_accuracy_drop.png | 各保护方式导致的精度损失汇总 |

---

## 预期实验结论

1. **数据脱敏 vs 梯度脱敏**
   - 相同 ε 下，梯度脱敏（DP-SGD）通常保留更高精度
   - 因为梯度脱敏只在参数更新时加噪，而数据脱敏每次前向传播都引入噪声

2. **Laplace vs Gaussian**
   - Gaussian 机制满足 (ε,δ)-DP，在相同 ε 下噪声更小，精度更高
   - Laplace 机制满足纯 ε-DP（更严格），噪声更大

3. **模型鲁棒性**
   - CNN 类模型（SimpleCNN/LeNet5）比 MLP 对噪声更鲁棒
   - 卷积操作有局部平均效果，一定程度上抵消数据噪声

4. **精度-隐私权衡**
   - ε 从 5 降到 1，精度平均下降约 2~8%
   - 强隐私保护（ε=1）下模型仍具有实用性（>90% 准确率）

---

## 与已有工作的关系

本实验与"DP-SGD 参数分析实验"形成互补：

```
已有工作                本实验新增
─────────────────────────────────────────
DP-SGD 参数分析    →    数据脱敏 vs 梯度脱敏对比
MNIST 单模型       →    三种模型横向对比
ε 与 accuracy      →    两种机制 × 三模型 全面分析
```
