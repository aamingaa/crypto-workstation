# 特征类型筛选使用指南

## 概述

`data_prepare_rolling` 函数现在支持针对特定类型的特征进行挖掘筛选，可以帮助你专注于特定领域的因子挖掘，如动量、大单、反转、tail等特征。

## 功能特性

### 1. 特征类型筛选
支持以下7种特征类型：

- **momentum**: 动量与反转特征
- **tail**: 大单特征  
- **orderflow**: 订单流特征
- **impact**: 价格冲击特征
- **volatility**: 波动率特征
- **basic**: 基础特征
- **path_shape**: 路径形状特征

### 2. 关键词筛选
支持基于关键词的特征筛选，如：
- `lambda`: 价格冲击相关特征
- `large`: 大单相关特征
- `signed`: 有向特征
- `imbalance`: 不平衡特征

## 使用方法

### 基本用法

```python
from gp_crypto_next.dataload import data_prepare_rolling

# 只使用动量特征
result = data_prepare_rolling(
    sym='BTCUSDT',
    freq='30min',
    start_date_train='2025-01-01',
    end_date_train='2025-01-15',
    start_date_test='2025-01-16', 
    end_date_test='2025-01-20',
    feature_types=['momentum'],  # 只筛选动量特征
    file_path='your_data_file.csv'
)
```

### 多类型组合筛选

```python
# 组合使用动量和大单特征
result = data_prepare_rolling(
    sym='BTCUSDT',
    freq='30min',
    start_date_train='2025-01-01',
    end_date_train='2025-01-15',
    start_date_test='2025-01-16',
    end_date_test='2025-01-20',
    feature_types=['momentum', 'tail'],  # 动量+大单特征
    file_path='your_data_file.csv'
)
```

### 关键词筛选

```python
# 筛选包含特定关键词的特征
result = data_prepare_rolling(
    sym='BTCUSDT',
    freq='30min',
    start_date_train='2025-01-01',
    end_date_train='2025-01-15',
    start_date_test='2025-01-16',
    end_date_test='2025-01-20',
    feature_keywords=['lambda', 'large'],  # 包含lambda或large的特征
    file_path='your_data_file.csv'
)
```

### 组合筛选

```python
# 同时使用类型和关键词筛选
result = data_prepare_rolling(
    sym='BTCUSDT',
    freq='30min',
    start_date_train='2025-01-01',
    end_date_train='2025-01-15',
    start_date_test='2025-01-16',
    end_date_test='2025-01-20',
    feature_types=['impact'],      # 价格冲击类型
    feature_keywords=['lambda'],   # 且包含lambda关键词
    file_path='your_data_file.csv'
)
```

## 特征类型详解

### 1. 动量特征 (momentum)
**关键词**: `mr_`, `momentum`, `reversion`, `dp_short`, `dp_zscore`

**典型特征**:
- `mr_rho1`: 一阶自相关系数
- `mr_strength`: 均值回复强度
- `dp_short`: 微动量（短期价格变化）
- `dp_zscore`: 微动量Z-score

**适用场景**: 预测价格趋势延续或反转

### 2. 大单特征 (tail)
**关键词**: `large_`, `q90`, `q95`, `q99`, `sweep`, `burstiness`

**典型特征**:
- `large_q90_dollar_sum`: 90分位大单金额
- `large_q95_imbalance_by_dollar`: 95分位大单不平衡
- `large_q90_sweep_count`: 大单扫盘次数
- `large_q95_burstiness`: 大单簇性指标

**适用场景**: 识别机构资金流向和大单冲击

### 3. 订单流特征 (orderflow)
**关键词**: `ofi_`, `gof_`, `signed_`, `imbalance`, `flow`

**典型特征**:
- `ofi_signed_qty_sum`: 有向成交量和
- `gof_by_volume`: 按成交量的Garman订单流
- `signed_dollar_sum`: 有向成交金额
- `imbalance_by_dollar`: 金额不平衡

**适用场景**: 分析买卖压力和市场情绪

### 4. 价格冲击特征 (impact)
**关键词**: `lambda`, `amihud`, `kyle`, `hasbrouck`, `impact`

**典型特征**:
- `kyle_lambda`: Kyle Lambda (价格冲击系数)
- `amihud_lambda`: Amihud Lambda (非流动性指标)
- `hasbrouck_lambda`: Hasbrouck Lambda (信息冲击系数)
- `impact_half_life`: 价格冲击半衰期

**适用场景**: 评估市场流动性和交易成本

### 5. 波动率特征 (volatility)
**关键词**: `rv`, `bpv`, `jump`, `volatility`, `std`

**典型特征**:
- `rv`: 已实现波动率
- `bpv`: 双幂变差
- `jump_rv_bpv`: 跳跃组件
- `hl_amplitude_ratio`: 高低幅度比

**适用场景**: 预测波动率变化和风险

### 6. 基础特征 (basic)
**关键词**: `int_trade_`, `vwap`, `volume`, `dollar`, `intensity`

**典型特征**:
- `int_trade_vwap`: 成交量加权平均价
- `int_trade_volume_sum`: 总成交量
- `int_trade_dollar_sum`: 总成交金额
- `int_trade_intensity`: 交易强度

**适用场景**: 基础市场状态分析

### 7. 路径形状特征 (path_shape)
**关键词**: `vwap_deviation`, `corr_`, `path`, `shape`, `deviation`

**典型特征**:
- `vwap_deviation`: VWAP偏离
- `corr_cumsum_signed_qty_logp`: 累计有向成交量与对数价格相关性
- `signed_vwap_deviation`: 有向VWAP偏离

**适用场景**: 分析价格路径形状和偏离程度

## 实际应用建议

### 1. 动量策略
```python
# 专注于动量和反转特征
feature_types=['momentum']
```

### 2. 大单跟踪策略
```python
# 专注于大单和订单流特征
feature_types=['tail', 'orderflow']
```

### 3. 流动性分析
```python
# 专注于价格冲击和波动率特征
feature_types=['impact', 'volatility']
```

### 4. 综合策略
```python
# 使用所有特征类型
feature_types=['momentum', 'tail', 'orderflow', 'impact', 'volatility', 'basic', 'path_shape']
```

## 性能优化

1. **特征数量控制**: 使用筛选功能可以减少特征维度，提高训练效率
2. **针对性挖掘**: 专注于特定类型的特征可以提高模型的专业性
3. **计算资源**: 减少特征数量可以降低内存使用和计算时间

## 注意事项

1. **数据文件**: 确保数据文件包含所需的特征列
2. **特征命名**: 特征筛选基于列名匹配，确保特征命名规范
3. **空结果**: 如果筛选后没有特征，函数会返回空的特征集
4. **兼容性**: 筛选功能与原有的滚动统计功能完全兼容

## 在GP分析中使用

### GPAnalyzer配置

在 `main_gp_new.py` 的 `GPAnalyzer` 类中，你可以通过设置以下属性来使用特征筛选：

```python
from gp_crypto_next.main_gp_new import GPAnalyzer

# 创建分析器
analyzer = GPAnalyzer()

# 基础配置
analyzer.sym = 'BTCUSDT'
analyzer.freq = '30min'
analyzer.data_source = 'rolling'  # 重要：使用滚动统计版
analyzer.file_path = 'your_data_file.csv'

# 特征筛选配置
analyzer.feature_types = ['momentum', 'tail']  # 动量+大单特征
analyzer.feature_keywords = ['lambda']         # 且包含lambda关键词

# 初始化数据
analyzer.initialize_data()

# 继续GP分析...
analyzer.run_analysis()
```

### 配置参数说明

- `feature_types`: 特征类型列表，如 `['momentum', 'tail']`
- `feature_keywords`: 关键词列表，如 `['lambda', 'large']`
- `data_source`: 必须设置为 `'rolling'` 才能使用特征筛选功能

## 示例脚本

### 1. 基础特征筛选示例
运行 `example_feature_filtering.py` 查看完整的使用示例：

```bash
python example_feature_filtering.py
```

### 2. GP分析中的特征筛选示例
运行 `example_gp_with_feature_filtering.py` 查看在GP分析中的使用方法：

```bash
python example_gp_with_feature_filtering.py
```

这些脚本演示了各种筛选方式的使用方法和效果。
