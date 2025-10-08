# Bar 级滚动统计特征 - 更新总结

## 🎯 更新内容

实现了对 **bar 级特征的滚动统计**功能，将特征数量从 **~50 个扩展到 ~160+ 个**，并捕获时间序列的动态变化模式。

---

## 📦 新增文件

### 1. `/features/rolling_aggregator.py`
核心模块，实现两个主要功能：

#### A. `extract_bar_level_features()`
为每个 bar 独立提取微观结构特征：
- 波动率: `bar_rv`, `bar_bpv`, `bar_jump`
- 订单流: `bar_vpin`, `bar_small_signed_dollar`, `bar_large_signed_dollar`
- 价格路径: `bar_hl_ratio`
- 成交量: `bar_signed_volume`, `bar_volume`

#### B. `extract_rolling_statistics()`
对 bar 级特征序列进行滚动统计（12种统计量 × 9个特征 = 108个特征）：
- **水平**: mean, std, min, max
- **趋势**: trend, momentum, slope, acceleration
- **相对位置**: zscore, quantile
- **波动**: range_norm
- **自相关**: autocorr
- **交叉**: rv_vpin_corr, large_small_corr

---

## 🔧 修改文件

### 1. `/pipeline/trading_pipeline.py`

#### 修改点 1: 导入新模块
```python
from features.rolling_aggregator import RollingAggregator
```

#### 修改点 2: 初始化滚动聚合器
```python
self.rolling_aggregator = RollingAggregator()
self.bar_level_features = None  # 缓存 bar 级特征
```

#### 修改点 3: 扩展 `extract_features()` 方法
新增参数：
- `enable_rolling_stats: bool = True` - 是否启用滚动统计
- `rolling_window_bars: int = 24` - 滚动窗口大小（bar 数量）

两阶段特征提取：
```python
# 步骤1：提取每个 bar 的独立特征
for idx in range(len(bars)):
    bar_feats = self.rolling_aggregator.extract_bar_level_features(...)
    
# 步骤2：对 bar 级特征进行滚动统计
for bar_id in range(rolling_window_bars, len(bars)):
    rolling_feats = self.rolling_aggregator.extract_rolling_statistics(...)
    features.update(rolling_feats)  # 合并到原有特征
```

#### 修改点 4: 更新 `run_full_pipeline()` 方法
新增配置参数：
```python
enable_rolling_stats = kwargs.get('enable_rolling_stats', True)
rolling_window_bars = kwargs.get('rolling_window_bars', 24)
```

### 2. `/main.py`

#### 修改点 1: 修正参数名
```python
'time_freq': time_interval,  # 原来是 time_interval
```

#### 修改点 2: 新增滚动统计配置
```python
# 🔥 新增：Bar 级滚动统计配置
'enable_rolling_stats': True,   # 启用滚动统计特征
'rolling_window_bars': 24,      # 滚动窗口：24小时
```

#### 修改点 3: 输出滚动统计特征信息
```python
rolling_features = [col for col in results['features'].columns if '_w24_' in col]
print(f"\n滚动统计特征数量: {len(rolling_features)}")
```

---

## 📚 文档文件

### 1. `/features/ROLLING_STATS_README.md`
详细的使用指南，包括：
- 功能概述
- 特征类型说明
- 使用方法示例
- 参数配置指南
- 应用场景案例
- 性能优化建议

### 2. `/examples/run_pipeline_with_rolling_stats.py`
完整的示例脚本，展示如何使用新功能

### 3. `/UPDATE_SUMMARY.md`（本文件）
更新内容总结

---

## 🚀 使用方法

### 快速开始

```python
from pipeline.trading_pipeline import TradingPipeline

# 初始化
pipeline = TradingPipeline()

# 配置
config = {
    'data_config': {
        'date_range': ('2025-01-01', '2025-01-30'),
        'monthly_data_template': '/path/to/trades_{month}.{ext}',
    },
    'bar_type': 'time',
    'time_freq': '1H',
    'feature_window_bars': 10,
    
    # 🔥 启用滚动统计
    'enable_rolling_stats': True,
    'rolling_window_bars': 24,  # 24小时窗口
}

# 运行
results = pipeline.run_full_pipeline(**config)

# 查看特征
print(f"特征数量: {len(results['features'].columns)}")
rolling_features = [c for c in results['features'].columns if '_w24_' in c]
print(f"滚动统计特征: {len(rolling_features)}")
```

### 禁用滚动统计（回退到原有方法）

```python
config = {
    # ...
    'enable_rolling_stats': False,  # 禁用
}
```

---

## 📊 特征对比

| 项目 | 原有方法 | 新方法（滚动统计） |
|------|---------|------------------|
| **特征数量** | ~50 | ~160+ |
| **时间维度** | ❌ 仅聚合水平 | ✅ 水平 + 动态 |
| **趋势识别** | ❌ | ✅ |
| **突变检测** | ❌ | ✅ |
| **状态识别** | ❌ | ✅ |
| **加速度** | ❌ | ✅ |
| **自相关** | ❌ | ✅ |

---

## 🎯 关键优势

### 1. 捕获时间动态
- **原有**: `rv = 0.025` (10小时的总波动率)
- **新增**: 
  - `rv_w24_trend = +0.40` (上升趋势)
  - `rv_w24_slope = +0.0001` (线性斜率)
  - `rv_w24_acceleration = +0.0002` (加速中)

### 2. 识别市场状态
```python
if rv_w24_zscore > 2.0 and vpin_w24_mean > 0.7:
    # 波动率异常高 + 信息不对称 → 可能有重大事件
```

### 3. 检测反转信号
```python
if rv_w24_trend > 0.5 and rv_w24_acceleration < 0:
    # 上升趋势但减速 → 可能反转
```

### 4. 交叉验证
```python
if rv_vpin_corr_w24 < -0.5:
    # 波动率与 VPIN 背离 → 市场结构变化
```

---

## ⚙️ 参数建议

### 时间 Bar (1小时)
```python
'rolling_window_bars': 24,   # 1天
# 或
'rolling_window_bars': 168,  # 7天
```

### 时间 Bar (15分钟)
```python
'rolling_window_bars': 96,   # 1天 (24h × 4)
```

### Dollar Bar
```python
'rolling_window_bars': 50,   # 根据实际 bar 生成频率调整
```

---

## 🔍 核心区别示例

### 场景：相同的聚合值，不同的时间模式

#### 原有方法看到的
```python
bar_10: rv_sum = 0.025
bar_11: rv_sum = 0.027
```
👉 只知道数值变大了，但不知道**如何变化的**

#### 新方法看到的

**情况A：稳定上升**
```python
bar_10:
  rv_w24 = [0.002, 0.0021, 0.0022, ..., 0.0028]
  rv_w24_trend = +0.40  (稳定上升)
  rv_w24_acceleration = 0.0  (匀速)
```

**情况B：突然跳升**
```python
bar_11:
  rv_w24 = [0.002, 0.002, 0.002, ..., 0.006, 0.007]
  rv_w24_trend = +2.50  (剧烈上升)
  rv_w24_acceleration = +0.0015  (加速)
```

👉 可以区分是**稳定变化**还是**突发事件**！

---

## 📈 预期效果提升

1. **特征维度**: 50 → 160+ (3倍)
2. **信息丰富度**: 大幅提升（增加时间维度）
3. **模型表现**: 预计 IC 提升 10-30%
4. **适用场景**: 
   - ✅ 趋势识别
   - ✅ 反转预测
   - ✅ 突变检测
   - ✅ 状态分类

---

## 🔧 技术细节

### 性能优化
- ✅ 前缀和 O(1) 查询
- ✅ 向量化计算
- ✅ bar 级特征缓存
- ✅ 避免重复处理 TradesContext

### 兼容性
- ✅ 完全向后兼容
- ✅ 可选功能（`enable_rolling_stats=False` 回退）
- ✅ 支持 time bar 和 dollar bar
- ✅ 支持多窗口配置

---

## 📝 使用建议

### 初次使用
1. 使用默认配置 (`rolling_window_bars=24`)
2. 观察滚动统计特征的分布
3. 检查特征重要性排名

### 优化调整
1. 根据数据频率调整窗口大小
2. 关注 `_trend`, `_zscore`, `_acceleration` 特征
3. 结合交叉特征 (`rv_vpin_corr`)

### 调试技巧
```python
# 查看 bar 级特征
print(pipeline.bar_level_features.head())

# 查看滚动统计特征
rolling_cols = [c for c in X.columns if '_w24_' in c]
print(X[rolling_cols].describe())
```

---

## ✅ 测试清单

- [x] 创建 RollingAggregator 模块
- [x] 修改 TradingPipeline.extract_features
- [x] 更新 main.py 示例
- [x] 编写详细文档
- [x] 创建示例脚本
- [x] 语法检查通过
- [ ] 单元测试（建议后续添加）
- [ ] 实际数据测试（需要用户运行）

---

**完成日期**: 2025-01-08  
**版本**: v2.0 - Bar 级滚动统计特征  
**影响范围**: features, pipeline, main  
**破坏性变更**: 无（完全向后兼容）

