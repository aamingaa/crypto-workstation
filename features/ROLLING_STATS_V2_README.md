# Bar 级滚动统计特征 v2.0 - 完整特征复用

## 🎯 重要更新

**v2.0 版本**现在**完全复用**现有的特征提取器，而不是重新实现特征逻辑！

### ✅ v2.0 改进

- **复用现有特征提取器**：`RollingAggregator` 现在接受 `MicrostructureFeatureExtractor` 作为参数
- **自动识别所有特征**：自动对所有 bar 级特征进行滚动统计，无需硬编码
- **避免重复代码**：不再重新实现波动率、订单流等特征逻辑
- **更灵活的配置**：支持通过特征配置启用/禁用不同的特征组

---

## 📊 工作流程

### **v1.0（旧版）- 重复实现**
```python
# ❌ 在 RollingAggregator 中重新实现特征
def extract_bar_level_features():
    # 重新计算 rv, bpv, vpin...
    rv = sum(r²)
    vpin = calculate_vpin(...)
    # 只实现了9个特征
```

### **v2.0（新版）- 复用现有**
```python
# ✅ 使用现有的特征提取器
def extract_bar_level_features():
    # 调用已有的特征提取器
    features = feature_extractor.extract_from_context(
        ctx, start_ts, end_ts,
        bar_window_start_idx=bar_idx,
        bar_window_end_idx=bar_idx  # 单个 bar
    )
    # 自动获取所有启用的特征（可能50+个）
```

---

## 🔧 架构设计

### 特征提取器层次
```
MicrostructureFeatureExtractor (主提取器)
    ├─ BasicFeatureExtractor           # 基础 VWAP/强度
    ├─ VolatilityFeatureExtractor      # RV/BPV/Jump
    ├─ MomentumFeatureExtractor         # 动量/反转
    ├─ OrderFlowFeatureExtractor        # GOF/OFI
    ├─ ImpactFeatureExtractor           # Kyle/Amihud
    ├─ TailFeatureExtractor             # 大单尾部
    ├─ PathShapeFeatureExtractor        # 价格路径
    └─ BucketedFlowFeatureExtractor    # 分桶订单流/VPIN

RollingAggregator
    ├─ 接受 MicrostructureFeatureExtractor
    ├─ 为每个 bar 调用特征提取器（单 bar 模式）
    └─ 对 bar 级特征序列进行滚动统计
```

---

## 💻 使用方法

### 1. 在 TradingPipeline 中（自动配置）

```python
from pipeline.trading_pipeline import TradingPipeline

# 配置特征组（决定在单个 bar 上提取哪些特征）
config = {
    'features': {
        'basic': False,
        'volatility': True,       # 启用：RV, BPV, Jump
        'momentum': True,          # 启用：动量、反转
        'orderflow': False,
        'impact': False,
        'tail': True,              # 启用：大单尾部
        'path_shape': False,
        'bucketed_flow': True,    # 启用：分桶、VPIN
    }
}

pipeline = TradingPipeline(config)

# RollingAggregator 会自动使用上面配置的特征提取器
results = pipeline.run_full_pipeline(
    enable_rolling_stats=True,
    rolling_window_bars=24
)

# 🔥 现在会自动对所有启用的特征进行滚动统计！
```

### 2. 独立使用 RollingAggregator

```python
from features.microstructure_extractor import MicrostructureFeatureExtractor
from features.rolling_aggregator import RollingAggregator

# 配置特征提取器
feature_config = {
    'volatility': True,
    'bucketed_flow': True,
    # ... 其他配置
}

feature_extractor = MicrostructureFeatureExtractor(feature_config)

# 创建 RollingAggregator（传入特征提取器）
aggregator = RollingAggregator(feature_extractor=feature_extractor)

# 提取单个 bar 的特征
bar_features = aggregator.extract_bar_level_features(bars, ctx, bar_idx=0)
# 返回: {'bar_rv': 0.002, 'bar_bpv': 0.0018, 'bar_vpin_all': 0.65, ...}

# 对 bar 序列进行滚动统计
rolling_stats = aggregator.extract_rolling_statistics(
    bar_features_df, 
    window=24, 
    current_idx=24
)
# 返回: {'bar_rv_w24_mean': 0.0025, 'bar_rv_w24_trend': +0.40, ...}
```

---

## 🔥 特征自动识别

### 自动识别逻辑
```python
# extract_rolling_statistics() 自动识别所有 bar_ 开头的数值特征
stat_features = [col for col in bar_features.columns 
                if col.startswith('bar_') and 
                pd.api.types.is_numeric_dtype(bar_features[col])]

# 对每个识别到的特征进行12种统计
for feat in stat_features:
    compute_statistics(feat, window)
```

### 示例输出
```python
# 如果启用了 volatility + bucketed_flow
bar_features.columns = [
    'bar_rv', 'bar_bpv', 'bar_jump',                    # 波动率 (3)
    'bar_micro_dp_short', 'bar_micro_dp_zscore',       # 微动量 (2)
    'bar_hl_amplitude_ratio',                           # 高低幅度 (1)
    'bar_small_buy_dollar', 'bar_small_signed_dollar', # 小单 (2)
    'bar_large_buy_dollar', 'bar_large_signed_dollar', # 大单 (2)
    'bar_vpin_all', 'bar_small_vpin', 'bar_large_vpin', # VPIN (3)
    # ... 更多特征
]

# 滚动统计后
rolling_features.columns = [
    # 每个基础特征 × 12种统计量
    'bar_rv_w24_mean', 'bar_rv_w24_std', 'bar_rv_w24_trend', ...
    'bar_bpv_w24_mean', 'bar_bpv_w24_std', ...
    'bar_vpin_all_w24_mean', 'bar_vpin_all_w24_zscore', ...
    # ... 共 N × 12 个特征
]
```

---

## 📈 特征数量对比

| 启用的特征组 | 单 bar 特征数 | 滚动统计特征数 | 总特征数 |
|------------|-------------|---------------|----------|
| **仅 volatility** | 6 | 6 × 12 = 72 | ~72 |
| **volatility + bucketed_flow** | ~25 | 25 × 12 = 300 | ~300 |
| **全部启用** | ~50 | 50 × 12 = 600 | ~600+ |

---

## 🎯 配置策略

### 1. 轻量级配置（快速测试）
```python
config = {
    'features': {
        'volatility': True,    # 仅波动率
        'bucketed_flow': True, # 仅分桶订单流
    }
}
# 特征数: ~300
# 速度: 快
```

### 2. 标准配置（推荐）
```python
config = {
    'features': {
        'volatility': True,
        'bucketed_flow': True,
        'momentum': True,
        'tail': True,
    }
}
# 特征数: ~400
# 速度: 中等
```

### 3. 完整配置（最强性能）
```python
config = {
    'features': {
        'basic': True,
        'volatility': True,
        'momentum': True,
        'orderflow': True,
        'impact': True,
        'tail': True,
        'path_shape': True,
        'bucketed_flow': True,
    }
}
# 特征数: ~600+
# 速度: 慢（但特征最丰富）
```

---

## 🔄 工作流程详解

### 步骤 1：配置特征提取器
```python
feature_config = {
    'volatility': True,
    'bucketed_flow': True,
}

feature_extractor = MicrostructureFeatureExtractor(feature_config)
```

### 步骤 2：为每个 bar 提取特征
```python
# 自动调用启用的子提取器
for bar_idx in range(len(bars)):
    bar_features = feature_extractor.extract_from_context(
        ctx, 
        start_ts=bar.start_time,
        end_ts=bar.end_time,
        bar_window_start_idx=bar_idx,
        bar_window_end_idx=bar_idx  # 单个 bar！
    )
    # 输出: {
    #   'rv': 0.002,
    #   'bpv': 0.0018,
    #   'vpin_all': 0.65,
    #   'small_signed_dollar': 1000000,
    #   ...
    # }
```

### 步骤 3：添加 'bar_' 前缀
```python
bar_features = {f'bar_{k}': v for k, v in features.items()}
# 输出: {
#   'bar_rv': 0.002,
#   'bar_bpv': 0.0018,
#   'bar_vpin_all': 0.65,
#   ...
# }
```

### 步骤 4：构建 bar 级特征 DataFrame
```python
bar_0: bar_rv=0.002, bar_vpin_all=0.65, ...
bar_1: bar_rv=0.0025, bar_vpin_all=0.62, ...
...
bar_23: bar_rv=0.0028, bar_vpin_all=0.67, ...
```

### 步骤 5：滚动统计
```python
# 自动识别所有 bar_ 特征
for feature in ['bar_rv', 'bar_bpv', 'bar_vpin_all', ...]:
    compute_statistics(feature, window=24)
    # 输出:
    # - bar_rv_w24_mean
    # - bar_rv_w24_std
    # - bar_rv_w24_trend
    # - ... (12种统计量)
```

---

## 🎨 特征命名规范

### 单 bar 特征
```
bar_{原始特征名}
```

示例:
- `rv` → `bar_rv`
- `vpin_all` → `bar_vpin_all`
- `small_signed_dollar` → `bar_small_signed_dollar`

### 滚动统计特征
```
bar_{原始特征名}_w{窗口大小}_{统计量}
```

示例:
- `bar_rv_w24_mean`: RV 的24小时均值
- `bar_vpin_all_w24_trend`: VPIN 的24小时趋势
- `bar_small_signed_dollar_w24_zscore`: 小单金额的24小时 Z-score

---

## 📊 交叉特征

### 自动检测
```python
# RollingAggregator 自动查找匹配的特征对
rv_cols = [col for col in features if 'rv' in col.lower()]
vpin_cols = [col for col in features if 'vpin' in col.lower()]

if rv_cols and vpin_cols:
    corr = np.corrcoef(rv_series, vpin_series)[0, 1]
    features['rv_vpin_corr_w24'] = corr
```

### 支持的交叉特征
- `rv_vpin_corr_w{window}`: RV 与 VPIN 的相关性
- `large_small_corr_w{window}`: 大单与小单的相关性

---

## 💡 优势总结

### v1.0（旧版）
- ❌ 重复实现特征逻辑
- ❌ 仅支持9个硬编码特征
- ❌ 难以维护和扩展
- ❌ 无法利用现有特征组

### v2.0（新版）
- ✅ 完全复用现有特征提取器
- ✅ 自动支持所有启用的特征
- ✅ 易于维护（修改一处即可）
- ✅ 灵活配置（通过特征组开关）
- ✅ 特征数量可达 600+

---

## 🚀 迁移指南

如果你已经使用了 v1.0，迁移到 v2.0 非常简单：

### 不需要修改任何配置！
```python
# 这段代码在 v1.0 和 v2.0 中完全一样
pipeline = TradingPipeline(config)
results = pipeline.run_full_pipeline(
    enable_rolling_stats=True,
    rolling_window_bars=24
)
```

### 唯一的区别
- **v1.0**: 只会对9个硬编码特征进行滚动统计
- **v2.0**: 会对**所有启用的特征**进行滚动统计

---

## 🔧 高级用法

### 自定义特征提取器
```python
from features.microstructure_extractor import MicrostructureFeatureExtractor

class CustomFeatureExtractor(MicrostructureFeatureExtractor):
    def extract_from_context(self, ctx, start_ts, end_ts, **kwargs):
        features = super().extract_from_context(ctx, start_ts, end_ts, **kwargs)
        
        # 添加自定义特征
        features['custom_feature'] = calculate_custom(ctx, start_ts, end_ts)
        
        return features

# 使用自定义提取器
custom_extractor = CustomFeatureExtractor(config)
aggregator = RollingAggregator(feature_extractor=custom_extractor)

# 🔥 自定义特征也会自动进行滚动统计！
```

---

**版本**: v2.0  
**更新日期**: 2025-01-08  
**关键改进**: 完全复用现有特征提取器，避免重复代码

