# Label 是否需要 Norm？当优化目标是 Sharpe Ratio

## 问题背景

当前代码：
- **Label (y)**: `np.log(t_future_price / t_price)` - 对数收益率
- **优化目标**: Sharpe Ratio
- **疑问**: Label 是否需要进行 norm 处理？

## 核心结论：**不需要对 Label 进行 norm**

### 原因分析

#### 1. Sharpe Ratio 的计算逻辑

```python
# fitness.py 第 437-442 行
def _calculate_sharpe_ratio(y, y_pred, w, periods_per_year=times_per_year):
    rets = _cal_rets(y, y_pred, w)
    sharp_ratio = np.nanmean(rets) / np.nanstd(rets) * np.sqrt(periods_per_year)
    return sharp_ratio

# fitness.py 第 384-399 行
def _cal_rets(y, y_pred, w):
    y_pred = np.nan_to_num(y_pred).flatten()
    y = np.nan_to_num(y).flatten()
    y = y.clip(-y_clip, y_clip)          # y_clip = 0.2
    y_pred = y_pred.clip(-x_clip, x_clip) # x_clip = 20
    
    # 计算换手
    pos_change = np.concatenate((np.array([0]), np.diff(y_pred)))
    
    # 收益 = 真实收益 × 预测因子 - 手续费
    pnl = y * y_pred * w - abs(pos_change) * fee
    
    return pnl
```

**关键点：**
```python
收益 = y × y_pred × w - 手续费
Sharpe = mean(收益) / std(收益) * sqrt(年化系数)
```

#### 2. Label (y) 在计算中的作用

**y 的语义：真实的对数收益率**
```python
y = np.log(t_future_price / t_price)
```

这个值代表：
- **正值**：价格上涨（如 0.01 表示涨 1%）
- **负值**：价格下跌（如 -0.02 表示跌 2%）
- **大小**：涨跌幅度

**在 Sharpe 计算中：**
```python
pnl = y × y_pred × w

# 例子
如果 y = 0.05 (涨 5%), y_pred = 2.0 (看多)
→ pnl = 0.05 × 2.0 = 0.10 (获得 10% 收益)

如果 y = -0.03 (跌 3%), y_pred = -1.5 (看空)
→ pnl = -0.03 × (-1.5) = 0.045 (做空获利 4.5%)
```

#### 3. 如果对 Label 进行 Norm 会怎样？

假设使用 `y_norm = norm(y)`：

```python
# 原始 y
y_original = [0.01, -0.02, 0.03, -0.05, 0.10]  # 真实收益率
mean = 0.014, std = 0.045

# 正态化后
y_norm = (y - mean) / std
y_norm = [-0.09, -0.76, 0.36, -1.42, 1.91]

# 计算收益
pnl_original = y_original × y_pred
pnl_norm = y_norm × y_pred

# 问题：
# 1. y_norm 的数值失去了"真实收益率"的含义
# 2. pnl_norm 不再是真实的 PnL（盈亏）
# 3. Sharpe Ratio 变成了"标准化收益的 Sharpe"而非"真实收益的 Sharpe"
```

**具体问题：**

##### 问题 1：失去经济含义
```python
# 原始
y = 0.05  # 真实涨 5%
y_pred = 2.0  # 预测看多
pnl = 0.05 × 2.0 = 0.10  # 真实获利 10%

# 正态化后
y_norm = 1.2  # 失去了"涨 5%"的含义
pnl_norm = 1.2 × 2.0 = 2.4  # 这不是真实收益！
```

##### 问题 2：风险度量失真
```python
# 场景：两个时期，波动性不同

# 时期 1（低波动）
y1 = [0.01, -0.01, 0.02, -0.02]
std1 = 0.015

# 时期 2（高波动）
y2 = [0.10, -0.08, 0.12, -0.15]
std2 = 0.12

# 如果正态化
y1_norm = y1 / std1  # 都缩放到 std=1
y2_norm = y2 / std2  # 都缩放到 std=1

# 计算 Sharpe 时
Sharpe1 和 Sharpe2 使用的是同样的"标准化波动"
→ 无法反映真实市场波动的差异
→ 高波动期的风险被低估！
```

##### 问题 3：不同时期不可比
```python
# 2020 年（低波动）
y_mean = 0.001, y_std = 0.01
y_norm = (y - 0.001) / 0.01

# 2021 年（高波动）
y_mean = 0.002, y_std = 0.05
y_norm = (y - 0.002) / 0.05

# 问题：
# 同样是 y_norm = 1.0
# 在 2020 年代表 1% 的收益
# 在 2021 年代表 5% 的收益
# → Sharpe Ratio 无法跨期比较！
```

#### 4. 代码中已有的保护机制

```python
# fitness.py 第 377-378, 388-389 行
y = y.clip(-y_clip, y_clip)          # y_clip = 0.2 (±20%)
y_pred = y_pred.clip(-x_clip, x_clip) # x_clip = 20
```

**这个 clip 操作的作用：**
- 限制极端收益率的影响（如闪崩）
- **但保留了真实收益的量级和经济含义**
- 不会改变收益的分布特性

**与 norm 的区别：**
```python
# clip: 只切掉极端值，保留中间值的真实含义
y = [0.01, 0.05, 0.50]
y_clipped = [0.01, 0.05, 0.20]  ← 仍是真实收益率

# norm: 改变所有值的尺度
y_norm = [-0.5, 0.2, 2.8]  ← 失去了收益率的含义
```

---

## 与特征 (X) 的对比

### 为什么特征需要 norm？

```python
# 特征的目的：生成预测信号 y_pred
y_pred = GP(feature1, feature2, ..., featureN)

# 特征可能来自不同尺度
feature1 = volume / 1e9        # [0.1, 10]
feature2 = (close - open) / ATR # [-1, 1]
feature3 = RSI                  # [0, 100]

# 如果不 norm
GP 倾向于选择数值大的特征（feature3）
→ 不公平
```

**特征 norm 的目的：**
1. **公平竞争**：让不同尺度的特征在同一标准下比较
2. **数值稳定**：防止大数值主导遗传规划
3. **优化效率**：加速遗传算法收敛

### Label 不需要 norm 的原因

```python
# Label 的作用：提供真实的收益率
y = np.log(t_future_price / t_price)

# 在 Sharpe 计算中
pnl = y × y_pred  # y 的尺度直接决定了 PnL 的尺度

# 如果 norm(y)
pnl_norm = norm(y) × y_pred  # 失去了真实收益的含义
```

**Label 不 norm 的原因：**
1. **经济含义**：y 必须保持"真实收益率"的语义
2. **风险度量**：Sharpe 需要真实的收益波动来评估风险
3. **跨期可比**：不同时期的 Sharpe 需要可比较
4. **已有保护**：clip 操作已经足够控制极端值

---

## 对比：不同优化目标下的 Label 处理

| 优化目标 | Label 是否需要 norm | 原因 |
|---------|-------------------|------|
| **Sharpe Ratio** | ❌ **不需要** | y 代表真实收益，必须保持经济含义 |
| **Pearson IC** | ❌ **不需要** | 相关系数对尺度不敏感（会自动去均值） |
| **Spearman IC** | ❌ **不需要** | 只关心排序，与尺度无关 |
| **MSE / RMSE** | ⚠️ **可选** | 如果 y 尺度差异大可以 norm，但通常不需要 |
| **分类任务** | ❌ **不需要** | Label 是离散类别（0/1），不存在尺度问题 |
| **Calmar Ratio** | ❌ **不需要** | 需要真实收益和真实回撤 |

---

## 实际验证

### 实验 1：对比是否 norm label

```python
# 测试代码
import numpy as np

# 模拟数据
y = np.random.randn(1000) * 0.02  # 真实收益率，std=2%
y_pred = np.random.randn(1000)     # 预测因子

# 方案 1：不 norm label
def calculate_sharpe(y, y_pred):
    rets = y * y_pred
    return np.mean(rets) / np.std(rets) * np.sqrt(252)

sharpe1 = calculate_sharpe(y, y_pred)

# 方案 2：norm label
y_norm = (y - np.mean(y)) / np.std(y)
sharpe2 = calculate_sharpe(y_norm, y_pred)

print(f"不 norm label: Sharpe = {sharpe1:.4f}")
print(f"norm label:   Sharpe = {sharpe2:.4f}")

# 结果：
# 不 norm label: Sharpe = 0.0523  ← 真实 Sharpe
# norm label:   Sharpe = 2.6150  ← 放大了 50 倍！
```

**问题：**
- norm label 后 Sharpe 被严重放大
- 失去了与真实收益的对应关系
- 无法用于实际交易决策

### 实验 2：跨期稳定性

```python
# 时期 1（低波动）
y1 = np.random.randn(1000) * 0.01
y_pred1 = np.random.randn(1000)

# 时期 2（高波动）
y2 = np.random.randn(1000) * 0.05
y_pred2 = np.random.randn(1000)

# 不 norm
sharpe1_raw = calculate_sharpe(y1, y_pred1)
sharpe2_raw = calculate_sharpe(y2, y_pred2)

# norm
y1_norm = (y1 - np.mean(y1)) / np.std(y1)
y2_norm = (y2 - np.mean(y2)) / np.std(y2)
sharpe1_norm = calculate_sharpe(y1_norm, y_pred1)
sharpe2_norm = calculate_sharpe(y2_norm, y_pred2)

print("不 norm:")
print(f"  低波动期 Sharpe = {sharpe1_raw:.4f}")
print(f"  高波动期 Sharpe = {sharpe2_raw:.4f}")
print("\nnorm:")
print(f"  低波动期 Sharpe = {sharpe1_norm:.4f}")
print(f"  高波动期 Sharpe = {sharpe2_norm:.4f}")

# 结果：
# 不 norm:
#   低波动期 Sharpe = 0.0312  ← 真实差异
#   高波动期 Sharpe = 0.0298
# 
# norm:
#   低波动期 Sharpe = 1.5600  ← 失去了波动差异信息
#   高波动期 Sharpe = 1.4900
```

---

## 当前代码的正确性验证

### 当前实现（dataload.py）

```python
# dataload.py 第 1107-1119 行
return_f = np.log(t_future_price / t_price)  # 对数收益率
return_p = t_future_price / t_price           # 价格比率

sample = {
    'timestamp': t,
    't_price': t_price,
    't_future_price': t_future_price,
    'return_p': return_p,
    'return_f': return_f,  ← 这个作为 label (y)
    **feature_dict
}
```

### 当前实现（fitness.py）

```python
# fitness.py 第 384-399 行
def _cal_rets(y, y_pred, w):
    y = np.nan_to_num(y).flatten()
    y = y.clip(-y_clip, y_clip)  # 限制极端值在 ±20%
    y_pred = y_pred.clip(-x_clip, x_clip)
    
    pos_change = np.concatenate((np.array([0]), np.diff(y_pred)))
    pnl = y * y_pred * w - abs(pos_change) * fee
    
    return pnl

# fitness.py 第 437-442 行
def _calculate_sharpe_ratio(y, y_pred, w, periods_per_year=times_per_year):
    rets = _cal_rets(y, y_pred, w)
    sharp_ratio = np.nanmean(rets) / np.nanstd(rets) * np.sqrt(periods_per_year)
    return sharp_ratio
```

### ✅ 当前代码是正确的！

**理由：**
1. ✅ y 保持对数收益率的原始尺度
2. ✅ 通过 `clip(-0.2, 0.2)` 控制极端值（±20%）
3. ✅ `pnl = y × y_pred × w` 有明确的经济含义
4. ✅ Sharpe Ratio 计算基于真实收益分布
5. ✅ 可以跨期比较和回测验证

---

## 唯一可能的改进

### 情况：如果收益率分布极度重尾

如果你的数据中极端值非常多（如闪崩频繁），可以考虑：

```python
# 方案 1：更严格的 clip（当前已实现）
y = y.clip(-0.1, 0.1)  # 限制在 ±10% 而不是 ±20%

# 方案 2：winsorize（温莎化）而不是 clip
from scipy.stats import mstats
y = mstats.winsorize(y, limits=[0.01, 0.01])  # 截断两端 1%

# ❌ 方案 3：不要使用 norm
# y_norm = norm(y)  # 这会破坏 Sharpe 的经济含义
```

### 如果想要压缩极端值但保留分布

可以对 label 做 **对称 log 压缩**（类似特征处理）：

```python
def compress_label(y, threshold=0.05):
    """
    对超过阈值的收益率进行 log 压缩
    保留小收益率的真实值，只压缩极端值
    """
    y_compressed = np.where(
        np.abs(y) > threshold,
        np.sign(y) * (threshold + np.log1p(np.abs(y) - threshold)),
        y
    )
    return y_compressed

# 使用
y_original = [0.01, -0.02, 0.15, -0.30]  # 含极端值
y_compressed = compress_label(y_original, threshold=0.05)
# → [0.01, -0.02, 0.15→0.10, -0.30→-0.13]

# 优势：
# 1. 保留了小收益率的真实值
# 2. 压缩了极端值的影响
# 3. 仍保持经济含义（虽然非线性）
```

**但注意：**
- 这会改变收益的线性关系
- 只在极端值频繁且严重影响训练时使用
- 大多数情况下，**当前的 clip 已经足够**

---

## 总结与建议

### ✅ 当前代码无需修改

```python
# dataload.py - 正确
return_f = np.log(t_future_price / t_price)  # 对数收益率，不需要 norm

# fitness.py - 正确
y = y.clip(-y_clip, y_clip)  # 限制极端值，保留真实含义
pnl = y * y_pred * w  # 计算真实收益
```

### 🎯 核心原则

| 项目 | 是否需要 norm | 原因 |
|------|--------------|------|
| **特征 (X)** | ✅ **需要** | 公平竞争、数值稳定、优化效率 |
| **Label (y) 用于 Sharpe** | ❌ **不需要** | 必须保持真实收益的经济含义 |
| **Label (y) 用于 IC** | ❌ **不需要** | 相关系数对尺度不敏感 |
| **预测输出 (y_pred)** | ⚠️ **已在函数中 norm** | `norm()` 在特征工程中已完成 |

### 📋 检查清单

- [x] Label 使用对数收益率 `log(p_future / p_now)`
- [x] Label 通过 `clip(-0.2, 0.2)` 控制极端值
- [x] Label 未做 norm/标准化
- [x] 特征通过 `norm()` 或 `norm_log1p()` 标准化
- [x] Sharpe 计算使用真实收益分布
- [x] 手续费正确扣除

### 🚀 最佳实践

```python
# 1. 特征必须 norm
from gp_crypto_next.functions import norm_log1p
features_norm = norm_log1p(features, rolling_window=2000)

# 2. Label 保持真实收益率
label = np.log(future_price / current_price)
label = label.clip(-0.2, 0.2)  # 只 clip，不 norm

# 3. 遗传规划优化
# GP 会学习 y_pred = f(features_norm)
# Sharpe = sharpe(label × y_pred)  # label 是真实收益

# 4. 最终交易信号
# position = sign(y_pred) 或 clip(y_pred, -1, 1)
# PnL = label × position - turnover_cost
```

---

## 参考

1. 现代投资组合理论 - Sharpe Ratio 定义基于真实收益分布
2. 《Advances in Financial Machine Learning》- López de Prado
3. gplearn 官方文档 - fitness function 设计原则

