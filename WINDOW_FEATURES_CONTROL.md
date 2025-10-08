# 窗口特征控制说明

## 🎯 问题背景

在 v2.0 中，我们同时有两套特征：

### 1. **原有窗口特征**（逐笔级一次性聚合）
```python
# 对过去 N 个 bar 的所有逐笔交易一次性聚合
window = [bar 0-9]  # 10个bar
features = extract_from_context(start=bar_0, end=bar_9)
# 输出: rv=0.025（10小时的总 RV）
```

### 2. **新增滚动统计特征**（bar 级 + 滚动）
```python
# 先提取每个 bar
bar_0: rv=0.002
bar_1: rv=0.0025
...
bar_9: rv=0.0028

# 再滚动统计（24个 bar）
bar_rv_w24_sum = 0.025   # ← 与窗口特征重复！
bar_rv_w24_mean = 0.00104
bar_rv_w24_trend = +0.40
bar_rv_w24_zscore = +1.2
...
```

## ⚠️ 重复问题

如果 `feature_window_bars = rolling_window_bars = 10`：
- `rv` (窗口特征) = 0.025
- `bar_rv_w10_sum` (滚动统计) = 0.025
- **完全重复！**

## ✅ 解决方案

新增 `enable_window_features` 参数来控制是否使用原有窗口特征。

### 默认配置（推荐）
```python
config = {
    'enable_rolling_stats': True,      # 启用滚动统计
    'rolling_window_bars': 24,
    'enable_window_features': False,   # ⚠️ 关闭窗口特征（默认）
}
```

### 为什么默认关闭窗口特征？

1. **避免重复**：滚动统计已经包含了聚合信息（如 `_sum`, `_mean`）
2. **更丰富**：滚动统计提供了 12 种统计量（mean, std, trend, zscore...）
3. **时间动态**：滚动统计能捕获趋势、加速度、相对位置等时间特征
4. **降低维度**：减少冗余特征，提升模型效率

---

## 📊 三种使用模式

### 模式 1：仅滚动统计（推荐）✅
```python
config = {
    'enable_rolling_stats': True,      # 启用
    'rolling_window_bars': 24,
    'enable_window_features': False,   # 关闭（默认）
}
```
**特征**：~600 个滚动统计特征  
**优势**：丰富的时间动态信息，无冗余

### 模式 2：仅窗口特征（传统方式）
```python
config = {
    'enable_rolling_stats': False,     # 关闭
    'enable_window_features': True,    # 启用
    'feature_window_bars': 10,
}
```
**特征**：~50 个窗口聚合特征  
**优势**：简单快速，适合快速测试

### 模式 3：两者都启用（对比实验）
```python
config = {
    'enable_rolling_stats': True,
    'rolling_window_bars': 24,
    'enable_window_features': True,    # 启用
    'feature_window_bars': 10,
}
```
**特征**：~50（窗口）+ ~600（滚动）= ~650  
**注意**：窗口特征会加上 `window_` 前缀以区分  
**用途**：对比两种方法的效果差异

---

## 🔧 参数说明

### `enable_rolling_stats` (bool, 默认 True)
- **True**: 启用 bar 级滚动统计特征
- **False**: 关闭滚动统计

### `rolling_window_bars` (int, 默认 24)
- 滚动统计的窗口大小（bar 数量）
- 示例：24 = 24小时（如果是小时bar）

### `enable_window_features` (bool, 默认 False)
- **True**: 启用原有的窗口级特征（会加 `window_` 前缀）
- **False**: 关闭窗口特征（推荐，避免冗余）

### `feature_window_bars` (int, 默认 10)
- 窗口特征的窗口大小（bar 数量）
- 仅在 `enable_window_features=True` 时生效

---

## 📈 特征命名规范

### 滚动统计特征
```
bar_{原始特征名}_w{窗口}_统计量
```
示例：
- `bar_rv_w24_mean`: RV 的24小时均值
- `bar_rv_w24_trend`: RV 的24小时趋势
- `bar_vpin_all_w24_zscore`: VPIN 的24小时 Z-score

### 窗口特征（如果启用）
```
window_{原始特征名}
```
示例：
- `window_rv`: 窗口内的 RV 聚合
- `window_vpin_all`: 窗口内的 VPIN 聚合

---

## 💡 使用建议

### 1. 新项目（推荐配置）
```python
config = {
    'enable_rolling_stats': True,
    'rolling_window_bars': 24,
    'enable_window_features': False,  # 默认关闭
}
```

### 2. 对比实验
```python
# 实验1：仅窗口特征
run_1 = pipeline.run_full_pipeline(
    enable_rolling_stats=False,
    enable_window_features=True,
)

# 实验2：仅滚动统计
run_2 = pipeline.run_full_pipeline(
    enable_rolling_stats=True,
    enable_window_features=False,
)

# 对比 IC 提升
print(f"窗口特征 IC: {run_1['evaluation']['summary']['pearson_ic_mean']:.4f}")
print(f"滚动统计 IC: {run_2['evaluation']['summary']['pearson_ic_mean']:.4f}")
```

### 3. 特征选择实验
```python
# 启用两者，让模型选择重要特征
run_full = pipeline.run_full_pipeline(
    enable_rolling_stats=True,
    enable_window_features=True,
)

# 查看特征重要性
importance = run_full['model'].get_feature_importance()
top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]

# 看看 window_ 和 bar_ 特征的重要性对比
window_features = [f for f, _ in top_features if f.startswith('window_')]
rolling_features = [f for f, _ in top_features if f.startswith('bar_')]
print(f"Top 20 中窗口特征: {len(window_features)}")
print(f"Top 20 中滚动特征: {len(rolling_features)}")
```

---

## 🎯 常见问题

### Q1: 为什么默认关闭窗口特征？
**A**: 滚动统计特征更丰富（12种统计量 vs 1个聚合值），且包含了窗口特征的信息。

### Q2: 什么时候应该启用窗口特征？
**A**: 
- 对比实验时
- 特征选择研究时
- 如果滚动统计效果不佳时

### Q3: 两者都启用会影响性能吗？
**A**: 会增加计算时间和特征维度，但不影响正确性。建议先用默认配置（仅滚动统计）。

### Q4: 窗口大小如何设置？
**A**:
- 滚动窗口: 通常设为 24（1天）或 168（7天）
- 窗口特征: 通常设为 10-20（如果启用的话）

---

## 📝 更新历史

- **2025-01-08**: 添加 `enable_window_features` 参数，默认关闭以避免特征重复
- **2025-01-08**: v2.0 发布，引入 bar 级滚动统计特征

---

**推荐配置**:
```python
enable_rolling_stats = True
rolling_window_bars = 24
enable_window_features = False  # 关闭窗口特征
```

