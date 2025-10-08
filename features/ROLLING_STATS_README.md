# Bar 级滚动统计特征使用指南

## 📋 概述

新增的 `RollingAggregator` 模块实现了对 bar 级特征的滚动统计，可以捕获时间序列的动态变化模式。

## 🔄 两阶段特征提取

### 阶段 1：Bar 级特征聚合
为每个 bar 独立计算微观结构特征：
```python
bar_0: rv=0.0020, vpin=0.60, large_signed_dollar=5M
bar_1: rv=0.0025, vpin=0.62, large_signed_dollar=8M
bar_2: rv=0.0030, vpin=0.58, large_signed_dollar=3M
...
```

### 阶段 2：滚动统计
在 bar 级特征序列上进行时间序列统计：
```python
features_at_bar_24 = {
    # 水平统计
    'bar_rv_w24_mean': 0.0025,
    'bar_rv_w24_std': 0.00035,
    'bar_rv_w24_min': 0.0018,
    'bar_rv_w24_max': 0.0032,
    
    # 趋势统计
    'bar_rv_w24_trend': +0.40,      # 相对变化
    'bar_rv_w24_slope': +0.0001,    # 线性回归斜率
    'bar_rv_w24_momentum': +0.0003, # 后1/4 vs 前1/4
    'bar_rv_w24_acceleration': +0.0002,  # 二阶导数
    
    # 相对位置
    'bar_rv_w24_zscore': +1.2,      # 标准化分数
    'bar_rv_w24_quantile': 0.85,    # 分位数位置
    
    # 波动统计
    'bar_rv_w24_range_norm': 0.45,  # 归一化范围
    
    # 自相关
    'bar_rv_w24_autocorr': 0.65,    # 滞后1的自相关
    
    # 交叉相关
    'rv_vpin_corr_w24': -0.45,      # RV 与 VPIN 的相关性
}
```

## 🎯 提取的特征类型

### Bar 级基础特征（9个）
1. `bar_rv`: 已实现波动率
2. `bar_bpv`: 双幂变差
3. `bar_jump`: 跳跃成分
4. `bar_vpin`: VPIN
5. `bar_small_signed_dollar`: 小单签名金额
6. `bar_large_signed_dollar`: 大单签名金额
7. `bar_hl_ratio`: 高低幅度比
8. `bar_signed_volume`: 签名成交量
9. `bar_volume`: 成交量

### 每个基础特征的滚动统计（12个）
- **mean**: 均值
- **std**: 标准差
- **min**: 最小值
- **max**: 最大值
- **trend**: 相对变化趋势
- **momentum**: 动量（后1/4 vs 前1/4）
- **slope**: 线性回归斜率
- **zscore**: Z-score（标准化位置）
- **quantile**: 分位数位置
- **range_norm**: 归一化范围
- **acceleration**: 加速度（二阶导）
- **autocorr**: 自相关系数

### 交叉特征（2个）
- `rv_vpin_corr_w{window}`: RV 与 VPIN 的相关性
- `large_small_corr_w{window}`: 大单与小单的相关性

**总计**: 9 × 12 + 2 = **110 个滚动统计特征**

## 💻 使用方法

### 方法 1：通过 TradingPipeline（推荐）

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
    
    # 特征提取配置
    'feature_window_bars': 10,
    'enable_rolling_stats': True,      # 启用滚动统计
    'rolling_window_bars': 24,         # 24小时窗口
}

# 运行
results = pipeline.run_full_pipeline(**config)
```

### 方法 2：单独使用 RollingAggregator

```python
from features.rolling_aggregator import RollingAggregator

# 初始化
aggregator = RollingAggregator()

# 步骤1：提取每个 bar 的特征
bar_features_list = []
for idx in range(len(bars)):
    bar_feats = aggregator.extract_bar_level_features(
        bars, trades_context, idx
    )
    bar_feats['bar_id'] = idx
    bar_features_list.append(bar_feats)

bar_level_features = pd.DataFrame(bar_features_list).set_index('bar_id')

# 步骤2：对 bar 级特征进行滚动统计
for bar_id in range(24, len(bars)):
    rolling_feats = aggregator.extract_rolling_statistics(
        bar_level_features,
        window=24,
        current_idx=bar_id
    )
```

## 🔍 与原有方法的对比

| 维度 | 原有方法 | 新方法（滚动统计） |
|------|---------|------------------|
| **输入** | 窗口内所有逐笔交易 | 窗口内各 bar 的特征序列 |
| **输出** | 1个值/特征 | 12+个值/特征 |
| **信息内容** | 聚合水平 | 水平 + 趋势 + 动态 |
| **能否识别趋势** | ❌ | ✅ |
| **能否识别加速** | ❌ | ✅ |
| **能否识别突变** | ❌ | ✅ |
| **特征数量** | ~50 | ~160+ |

## 📊 应用场景

### 场景 1：趋势识别
```python
if features['bar_rv_w24_trend'] > 0.5 and features['bar_rv_w24_slope'] > 0.001:
    # RV 强上升趋势 → 波动率加剧
    prediction = 'high_volatility_period'
```

### 场景 2：突变检测
```python
if features['bar_rv_w24_zscore'] > 2.0:
    # 当前 RV 显著高于历史均值 → 可能有重大事件
    prediction = 'potential_event_driven_move'
```

### 场景 3：状态识别
```python
if features['bar_vpin_w24_mean'] > 0.7 and features['bar_vpin_w24_std'] < 0.1:
    # VPIN 高且稳定 → 持续的信息不对称
    prediction = 'informed_trading_regime'
```

### 场景 4：反转信号
```python
if (features['bar_rv_w24_trend'] > 0.8 and 
    features['bar_rv_w24_acceleration'] < -0.001):
    # RV 上升但加速度为负 → 可能反转
    prediction = 'potential_volatility_reversal'
```

## ⚙️ 参数配置

### `feature_window_bars`（默认 10）
- 逐笔级特征提取窗口
- 建议: 10-20 个 bar

### `rolling_window_bars`（默认 24）
- bar 级滚动统计窗口
- 建议:
  - 小时 bar: 24 (1天), 168 (7天)
  - 分钟 bar: 60 (1小时), 1440 (1天)

### `enable_rolling_stats`（默认 True）
- 是否启用滚动统计
- 设为 False 可回退到原有方法

## 📈 性能优化

1. **预计算前缀和**: 使用 `TradesContext` 的累计数组实现 O(1) 查询
2. **向量化计算**: 使用 numpy 数组操作避免循环
3. **缓存 bar 级特征**: 避免重复计算

## 🔧 扩展建议

### 增加更多窗口长度
```python
aggregator = RollingAggregator(windows=[6, 12, 24, 168])
```

### 自定义统计量
在 `RollingAggregator.extract_rolling_statistics` 中添加:
```python
# 偏度
features[f'{prefix}_skew'] = scipy.stats.skew(series)

# 峰度
features[f'{prefix}_kurtosis'] = scipy.stats.kurtosis(series)

# 分位数
features[f'{prefix}_q25'] = np.quantile(series, 0.25)
features[f'{prefix}_q75'] = np.quantile(series, 0.75)
```

## 📚 参考

- 论文: *High-Frequency Trading and Price Discovery* (Brogaard et al., 2014)
- 方法: Time Series Feature Engineering
- 应用: Market Microstructure Analysis

---

**最后更新**: 2025-01-08

