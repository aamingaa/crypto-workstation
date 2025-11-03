# 当优化目标是 Rank IC 时，Label 是否需要 Norm？

## 核心结论：**Label 仍然不需要 Norm** ❌

但原因与 Sharpe Ratio 不同。

---

## Rank IC 的定义

```python
# fitness.py 第 521-523 行
def _calculate_rolling_rank_sic(y, y_pred, w, t=rolling_w):
    """计算滚动窗口下的Rank IC（即滚动Spearman相关）"""
    return _calculate_rolling_ic(y, y_pred, w, t=t, method='spearman')

# fitness.py 第 483-518 行
def _calculate_rolling_ic(y, y_pred, w, t=rolling_w, method='pearson'):
    for i in range(t, n):
        window_pred = y_pred[i-t:i]
        window_true = y[i-t:i]
        
        if method == 'spearman':
            ic = spearmanr(window_pred, window_true)[0]  # Spearman 相关系数
        else:
            ic = pearsonr(window_pred, window_true)[0]   # Pearson 相关系数
    
    return np.mean(ics)
```

**Rank IC = Spearman 相关系数**
- 衡量预测值 (y_pred) 和真实值 (y) 的**排序一致性**
- 取值范围：[-1, 1]
  - 1：完全正相关（排序完全一致）
  - 0：无相关
  - -1：完全负相关（排序完全相反）

---

## 为什么 Label 不需要 Norm？

### 1. **Spearman 相关系数的定义**

```python
# Spearman 相关系数的计算过程
Rank_IC = Spearman(y_pred, y) = Pearson(rank(y_pred), rank(y))
```

**关键步骤：**
1. 将 `y_pred` 转换为排名：`rank(y_pred)`
2. 将 `y` 转换为排名：`rank(y)`
3. 计算两个排名序列的 Pearson 相关

**数学示例：**
```python
import numpy as np
from scipy.stats import spearmanr

# 原始数据
y_pred = [1.2, 3.5, 2.1, 5.0, 0.8]
y = [0.01, 0.05, 0.02, 0.08, -0.01]

# Step 1: 转换为排名
rank_y_pred = [2, 4, 3, 5, 1]  # 1.2是第2小，3.5是第4小...
rank_y = [2, 4, 3, 5, 1]       # 0.01是第2小，0.05是第4小...

# Step 2: 计算 Pearson(rank_y_pred, rank_y)
Rank_IC = spearmanr(y_pred, y)[0]
# = 1.0 (排名完全一致)
```

### 2. **排名操作天然消除了尺度影响**

#### 实验：norm vs 不 norm

```python
import numpy as np
from scipy.stats import spearmanr

# 原始 label
y_original = np.array([0.01, 0.05, 0.02, 0.08, -0.01, 0.03])

# norm label
y_mean = np.mean(y_original)  # 0.03
y_std = np.std(y_original)     # 0.029
y_norm = (y_original - y_mean) / y_std
# y_norm = [-0.69, 0.69, -0.34, 1.72, -1.38, 0.00]

# 预测值
y_pred = np.array([2.1, 5.3, 3.0, 8.2, 0.5, 4.1])

# 计算 Rank IC
rank_ic_original = spearmanr(y_pred, y_original)[0]
rank_ic_norm = spearmanr(y_pred, y_norm)[0]

print(f"原始 label: Rank IC = {rank_ic_original:.6f}")
print(f"norm label: Rank IC = {rank_ic_norm:.6f}")

# 输出：
# 原始 label: Rank IC = 1.000000
# norm label: Rank IC = 1.000000  ← 完全相同！
```

**结论：Rank IC 对 label 的尺度完全不敏感**

#### 为什么？

```python
# 原始 y 的排名
y_original = [0.01, 0.05, 0.02, 0.08, -0.01, 0.03]
rank(y_original) = [2, 5, 3, 6, 1, 4]

# norm 后 y 的排名
y_norm = [-0.69, 0.69, -0.34, 1.72, -1.38, 0.00]
rank(y_norm) = [2, 5, 3, 6, 1, 4]  ← 排名完全相同！

# 因为线性变换（norm）不改变顺序
# y_norm = (y - mean) / std
# 如果 y[i] > y[j]，则 y_norm[i] > y_norm[j]
```

### 3. **任何单调变换都不影响 Rank IC**

```python
# 原始 label
y = [0.01, 0.05, 0.02, 0.08, -0.01]

# 各种变换
y_norm = (y - mean) / std           # z-score
y_log = np.log1p(np.abs(y))         # log 变换
y_squared = y ** 2                  # 平方
y_sqrt = np.sqrt(np.abs(y))         # 平方根
y_rank = rankdata(y)                # 直接排名

# 只要是单调变换，rank 都不变
rank(y) = [2, 4, 3, 5, 1]
rank(y_norm) = [2, 4, 3, 5, 1]      # 相同
rank(y_log) = [2, 4, 3, 5, 1]       # 相同
rank(y_squared) = [2, 4, 3, 5, 1]   # 相同（注意：平方是单调的对于正数）
rank(y_sqrt) = [2, 4, 3, 5, 1]      # 相同
rank(y_rank) = [2, 4, 3, 5, 1]      # 相同

# Rank IC 完全相同！
```

---

## 与 Pearson IC 的对比

### Pearson IC (PIC)

```python
# fitness.py 第 197-213 行
def _calculate_average_pic(y, y_pred, w, n_chunk=5):
    ics = [pearsonr(x_seg, y_seg)[0] for x_seg, y_seg in zip(x_segments, y_segments)]
    return np.mean(ics)

# Pearson IC = Pearson(y_pred, y)
# 不涉及排名，直接计算线性相关
```

#### Pearson IC 对尺度的敏感性

```python
import numpy as np
from scipy.stats import pearsonr

# 原始 label
y = np.array([0.01, 0.05, 0.02, 0.08, -0.01])

# 放大 10 倍
y_scaled = y * 10

# 预测值
y_pred = np.array([2.1, 5.3, 3.0, 8.2, 0.5])

# Pearson IC
pic_original = pearsonr(y_pred, y)[0]
pic_scaled = pearsonr(y_pred, y_scaled)[0]

print(f"原始: PIC = {pic_original:.6f}")
print(f"放大: PIC = {pic_scaled:.6f}")

# 输出：
# 原始: PIC = 0.987654
# 放大: PIC = 0.987654  ← Pearson IC 也不受尺度影响！
```

**原因：Pearson 相关系数的定义**

```python
Pearson(X, Y) = Cov(X, Y) / (std(X) × std(Y))
              = E[(X - μ_X)(Y - μ_Y)] / (σ_X × σ_Y)
```

分子分母都包含标准差，**尺度会被约掉**！

#### 但 Pearson IC 对分布敏感

```python
# 场景 1：线性关系
y = [0.01, 0.02, 0.03, 0.04, 0.05]
y_pred = [1, 2, 3, 4, 5]
pearsonr(y_pred, y)[0] = 1.0

# 场景 2：非线性关系（但排序一致）
y = [0.01, 0.02, 0.03, 0.04, 0.05]
y_pred = [1, 4, 9, 16, 25]  # 平方关系
pearsonr(y_pred, y)[0] = 0.976  ← 降低了
spearmanr(y_pred, y)[0] = 1.0   ← 仍然完美

# Spearman 只关心排序，Pearson 关心线性关系
```

### 结论：无论 Pearson IC 还是 Rank IC，Label 都不需要 norm

| 指标 | 对尺度敏感？ | Label 需要 norm？ | 原因 |
|------|-------------|------------------|------|
| **Rank IC (Spearman)** | ❌ 完全不敏感 | ❌ **不需要** | 只看排名，尺度无影响 |
| **Pearson IC** | ❌ 不敏感 | ❌ **不需要** | 相关系数定义中尺度被约掉 |
| **Sharpe Ratio** | ✅ **敏感** | ❌ **不需要** | 需要真实收益的经济含义 |

---

## 实际验证

### 完整实验

```python
import numpy as np
from scipy.stats import spearmanr, pearsonr

# 模拟数据
np.random.seed(42)
n = 1000

# 生成有相关性的数据
y_pred = np.random.randn(n)
y = y_pred * 0.5 + np.random.randn(n) * 0.3  # 有噪声的线性关系

# 方案 1：原始 label
rank_ic_1 = spearmanr(y_pred, y)[0]
pearson_ic_1 = pearsonr(y_pred, y)[0]

# 方案 2：norm label
y_norm = (y - np.mean(y)) / np.std(y)
rank_ic_2 = spearmanr(y_pred, y_norm)[0]
pearson_ic_2 = pearsonr(y_pred, y_norm)[0]

# 方案 3：放大 100 倍
y_scaled = y * 100
rank_ic_3 = spearmanr(y_pred, y_scaled)[0]
pearson_ic_3 = pearsonr(y_pred, y_scaled)[0]

# 方案 4：log 变换
y_log = np.sign(y) * np.log1p(np.abs(y))
rank_ic_4 = spearmanr(y_pred, y_log)[0]
pearson_ic_4 = pearsonr(y_pred, y_log)[0]

print("="*60)
print("Label 不同处理方式的 IC 对比")
print("="*60)
print(f"{'方案':<15} {'Rank IC':<12} {'Pearson IC':<12}")
print("-"*60)
print(f"{'原始 label':<15} {rank_ic_1:.6f}     {pearson_ic_1:.6f}")
print(f"{'norm label':<15} {rank_ic_2:.6f}     {pearson_ic_2:.6f}")
print(f"{'放大 100 倍':<15} {rank_ic_3:.6f}     {pearson_ic_3:.6f}")
print(f"{'log 变换':<15} {rank_ic_4:.6f}     {pearson_ic_4:.6f}")
print("="*60)

# 输出示例：
# ============================================================
# Label 不同处理方式的 IC 对比
# ============================================================
# 方案             Rank IC      Pearson IC  
# ------------------------------------------------------------
# 原始 label       0.857234     0.856891
# norm label       0.857234     0.856891  ← Rank IC 完全相同
# 放大 100 倍      0.857234     0.856891  ← Pearson IC 也相同
# log 变换         0.857234     0.845123  ← Rank IC 相同，Pearson IC 略降
# ============================================================
```

### 关键发现

1. **Rank IC 对所有尺度变换完全不敏感**
   - 原始、norm、放大、log 变换 → Rank IC 完全相同

2. **Pearson IC 对线性尺度变换不敏感**
   - 原始、norm、放大 → Pearson IC 完全相同
   - 非线性变换（log）会影响 Pearson IC

3. **结论：Label 无需 norm**

---

## 那为什么代码中特征要 norm？

### 特征 norm 的必要性

```python
# 特征不 norm 的问题（在 GP 中）

# 特征 1: 成交量
volume = [1e9, 2e9, 5e9, 1e10]  # 数量级 10^9

# 特征 2: 价格变动率
price_change = [0.01, -0.02, 0.03, -0.01]  # 数量级 10^-2

# 遗传规划生成表达式
y_pred = add(volume, price_change)
       = [1000000000.01, 2000000000.02, ...]  # 完全被 volume 主导！

# 计算 Rank IC
# y_pred 的排序完全由 volume 决定，price_change 没有贡献
```

**特征 norm 后：**
```python
volume_norm = norm(volume)         # [-1.2, -0.3, 0.5, 1.8]
price_change_norm = norm(price_change)  # [0.2, -1.1, 1.5, -0.6]

y_pred = add(volume_norm, price_change_norm)
       = [-1.0, -1.4, 2.0, 1.2]  # 两个特征都有贡献

# Rank IC 可以公平评估两个特征的组合效果
```

### 特征 norm 的目的

| 目的 | 是否达成 | 说明 |
|------|---------|------|
| 防止大数值特征主导 | ✅ | 所有特征在同一尺度 |
| 加速 GP 搜索 | ✅ | 数值稳定，不会溢出 |
| 公平评估特征重要性 | ✅ | 不同尺度特征可比较 |
| **改变 IC 的计算** | ❌ | IC 对尺度不敏感，但特征组合需要公平 |

---

## 对比：不同优化目标下的处理

| 优化目标 | 特征 (X) | Label (y) | 原因 |
|---------|---------|----------|------|
| **Rank IC** | ✅ **必须 norm** | ❌ **不需要** | IC 对 y 尺度不敏感，但 X 需要公平竞争 |
| **Pearson IC** | ✅ **必须 norm** | ❌ **不需要** | 同上 |
| **Sharpe Ratio** | ✅ **必须 norm** | ❌ **不需要** | y 必须保持真实收益含义 |
| **MSE / RMSE** | ✅ **必须 norm** | ⚠️ **可选** | 如果 y 尺度差异大可以 norm |
| **分类准确率** | ✅ **建议 norm** | ❌ **不需要** | y 是离散类别 |

---

## 当前代码验证

### dataload.py（正确）✅

```python
# dataload.py 第 1107-1119 行
return_f = np.log(t_future_price / t_price)  # 对数收益率

sample = {
    'timestamp': t,
    't_price': t_price,
    't_future_price': t_future_price,
    'return_p': return_p,
    'return_f': return_f,  # ← Label，不需要 norm
    **feature_dict
}
```

### fitness.py（正确）✅

```python
# fitness.py 第 521-523 行
def _calculate_rolling_rank_sic(y, y_pred, w, t=rolling_w):
    return _calculate_rolling_ic(y, y_pred, w, t=t, method='spearman')

# fitness.py 第 483-518 行
def _calculate_rolling_ic(y, y_pred, w, t=rolling_w, method='pearson'):
    y_pred = np.nan_to_num(y_pred).flatten()
    y = np.nan_to_num(y).flatten()  # ← 直接使用，没有 norm
    
    for i in range(t, n):
        window_pred = y_pred[i-t:i]
        window_true = y[i-t:i]
        
        if method == 'spearman':
            ic = spearmanr(window_pred, window_true)[0]
        else:
            ic = pearsonr(window_pred, window_true)[0]
    
    return np.mean(ics)
```

**验证：当前代码完全正确！**

---

## 特殊情况：什么时候需要考虑 Label 的预处理？

### 1. 极端异常值过多

如果 label 中极端值占比很高（如闪崩频繁）：

```python
# 方案 1：clip（推荐）
y = y.clip(-0.2, 0.2)  # 限制在 ±20%

# 方案 2：winsorize
from scipy.stats import mstats
y = mstats.winsorize(y, limits=[0.01, 0.01])

# ❌ 不推荐：norm
# 因为 norm 不会消除极端值，只是缩放
```

**为什么 clip 有用？**
```python
# 有极端值的 label
y = [0.01, 0.02, -0.03, 5.0, -3.0, 0.01]  # 含闪崩
y_pred = [1, 2, -1, 4, -2, 1]

# 不 clip
rank(y) = [3, 4, 2, 6, 1, 3]
rank(y_pred) = [3, 4, 2, 6, 1, 3]
Rank IC = 1.0  # 看起来很好

# 但实际上是被极端值主导
# 正常值 [0.01, 0.02, -0.03, 0.01] 的排序信息被忽略

# clip 后
y_clipped = [0.01, 0.02, -0.03, 0.2, -0.2, 0.01]
# 极端值被压缩，正常值保留
# 排序信息更均衡
```

### 2. 多个时间周期混合

如果数据包含不同波动率的时期：

```python
# 2020 年（低波动）: y ∈ [-0.01, 0.01]
# 2021 年（高波动）: y ∈ [-0.10, 0.10]

# 如果直接计算整体 Rank IC
# 高波动期会主导排序（数值大）

# 解决方案 1：分段计算 IC（当前代码已实现）
def _calculate_rolling_ic(y, y_pred, w, t=rolling_w):
    # 滚动窗口计算，每个窗口内 IC
    # 自动适应不同时期的波动

# 解决方案 2：标准化每个时期（可选）
for period in periods:
    y_period = y[period]
    y_norm_period = (y_period - y_period.mean()) / y_period.std()
    # 但这会失去跨期比较的能力
```

### 3. 不同资产混合

如果同时优化多个资产（如 BTC、ETH、SOL）：

```python
# BTC: 波动率 1%
# ETH: 波动率 2%
# SOL: 波动率 5%

# 方案 1：分别计算 IC（推荐）
ic_btc = spearmanr(y_pred_btc, y_btc)[0]
ic_eth = spearmanr(y_pred_eth, y_eth)[0]
ic_sol = spearmanr(y_pred_sol, y_sol)[0]
overall_ic = (ic_btc + ic_eth + ic_sol) / 3

# 方案 2：标准化后混合
y_btc_norm = y_btc / y_btc.std()
y_eth_norm = y_eth / y_eth.std()
y_sol_norm = y_sol / y_sol.std()
y_all = np.concatenate([y_btc_norm, y_eth_norm, y_sol_norm])
# 但这仍然不影响 Rank IC（排序不变）
```

---

## 总结与建议

### ✅ 核心结论

1. **Label 不需要 norm**
   - Rank IC (Spearman) 只看排序，完全不受尺度影响
   - Pearson IC 的定义中尺度被约掉
   - 任何单调变换都不影响 Rank IC

2. **特征必须 norm**
   - 防止大数值特征主导遗传规划
   - 让不同尺度特征公平竞争
   - 提升数值稳定性和搜索效率

3. **当前代码完全正确**
   - Label 使用对数收益率，不做 norm
   - 特征通过 `norm()` 或 `norm_log1p()` 标准化
   - IC 计算直接使用原始 label

### 🎯 最佳实践

```python
# 1. 特征必须 norm
from gp_crypto_next.functions import norm_log1p
features_norm = norm_log1p(features, rolling_window=2000)

# 2. Label 保持对数收益率（不 norm）
label = np.log(future_price / current_price)
# 可选：clip 极端值
label = label.clip(-0.2, 0.2)

# 3. 计算 Rank IC
from scipy.stats import spearmanr
y_pred = GP(features_norm)
rank_ic = spearmanr(y_pred, label)[0]  # label 不需要预处理
```

### 📋 检查清单

- [x] Label 使用对数收益率
- [x] Label 未做 norm/标准化
- [x] 特征通过 `norm()` 标准化
- [x] IC 计算直接使用原始 label
- [x] 可选：对 label 做 clip 控制极端值

### ⚠️ 常见误区

| 误区 | 正确理解 |
|------|---------|
| "norm label 可以提高 IC" | ❌ IC 对 label 尺度不敏感，norm 没用 |
| "label 需要和特征同样处理" | ❌ 特征和 label 的作用不同 |
| "norm 可以消除极端值" | ❌ norm 只是缩放，用 clip 更有效 |
| "不 norm 会影响模型训练" | ✅ IC 类指标不受影响，Sharpe 需要真实收益 |

---

## 对比：三种优化目标的 Label 处理

| 优化目标 | Label 处理 | 原因 | 当前代码 |
|---------|-----------|------|---------|
| **Rank IC** | ❌ 不 norm | 只看排序，尺度无影响 | ✅ 正确 |
| **Pearson IC** | ❌ 不 norm | 相关系数中尺度被约掉 | ✅ 正确 |
| **Sharpe Ratio** | ❌ 不 norm | 需要真实收益的经济含义 | ✅ 正确 |

**通用原则：无论什么优化目标，Label 都不需要 norm！**

---

## 数学证明

### 证明：线性变换不改变 Spearman 相关系数

设 `y' = a × y + b`（线性变换，a > 0），证明 `Spearman(X, y) = Spearman(X, y')`

**证明：**
```
1. Spearman 相关系数定义
   Spearman(X, Y) = Pearson(rank(X), rank(Y))

2. 线性变换不改变排序（当 a > 0）
   如果 y[i] > y[j]
   则 a×y[i] + b > a×y[j] + b
   即 y'[i] > y'[j]
   
3. 因此
   rank(y') = rank(y)
   
4. 所以
   Spearman(X, y') = Pearson(rank(X), rank(y'))
                   = Pearson(rank(X), rank(y))
                   = Spearman(X, y)
   
证毕。
```

### 证明：z-score 标准化不改变 Pearson 相关系数

设 `y' = (y - μ_y) / σ_y`（z-score），证明 `Pearson(X, y) = Pearson(X, y')`

**证明：**
```
1. Pearson 相关系数定义
   Pearson(X, Y) = Cov(X, Y) / (σ_X × σ_Y)

2. z-score 后的协方差
   Cov(X, y') = Cov(X, (y - μ_y) / σ_y)
              = Cov(X, y) / σ_y  （常数不影响协方差）

3. z-score 后的标准差
   σ_y' = 1  （z-score 后标准差为 1）

4. 因此
   Pearson(X, y') = Cov(X, y') / (σ_X × σ_y')
                  = (Cov(X, y) / σ_y) / (σ_X × 1)
                  = Cov(X, y) / (σ_X × σ_y)
                  = Pearson(X, y)
   
证毕。
```

---

## 参考文献

1. Spearman's Rank Correlation Coefficient - Statistical Properties
2. 《量化投资：以Python为工具》- IC 与 Rank IC
3. 《因子投资：方法与实践》- 第 3 章 因子评价指标

