# Norm函数在粗粒度预计算中的数据泄露问题修复

## 🚨 问题描述

在 `dataload.py` 的粗粒度预计算优化中，存在严重的特征标准化问题：

### 问题根源

```python
# dataload.py 第1250行
base_feature = originalFeature.BaseFeature(coarse_bars.copy(), include_categories=include_categories)
features_df = base_feature.init_feature_df
```

每组粗粒度数据**独立计算特征**，特征计算中使用 `norm()` 函数：

```python
# functions.py 第154行
def norm(x, rolling_window=2000):
    factors_std = factors_data.rolling(window=rolling_window, min_periods=1).std()
    factor_value = (factors_data) / factors_std
    return np.nan_to_num(factor_value).flatten()
```

### 问题影响

1. **标准化基准不一致**：
   - 组0使用组0数据的rolling std
   - 组1使用组1数据的rolling std
   - 同一市场状态被标准化成不同值

2. **数据泄露**：
   - Rolling window使用未来数据
   - 前2000个点标准化不稳定

3. **特征不一致**：
   - TFT训练时会看到同一时刻的不同标准化值
   - 模型无法学到稳定的模式

## ✅ 解决方案

### 方案1：在粗粒度阶段不标准化（推荐）⭐

#### 步骤1：修改 `originalFeature.py`

在 `BaseFeature` 类中添加 `apply_norm` 参数：

```python
class BaseFeature:
    def __init__(self, data, include_categories=None, apply_norm=True):
        """
        参数:
            data: OHLCV数据
            include_categories: 要计算的特征类别
            apply_norm: 是否应用norm标准化（默认True，兼容原有代码）
        """
        self.data = data
        self.apply_norm = apply_norm
        self.include_categories = include_categories if include_categories else ['all']
        
        # 计算特征
        self.init_feature_df = self._calculate_all_features()
    
    def _apply_feature_func(self, feature_name, feature_func):
        """应用特征计算函数"""
        try:
            raw_values = feature_func(self.data)
            
            if self.apply_norm:
                # 原有逻辑：应用norm标准化
                return norm(raw_values)
            else:
                # 新逻辑：不标准化，返回原始值
                return np.nan_to_num(raw_values)
                
        except Exception as e:
            print(f"特征 {feature_name} 计算失败: {e}")
            return np.zeros(len(self.data))
```

#### 步骤2：修改 `dataload.py`

在粗粒度预计算时不应用标准化：

```python
# 第1250行左右
print(f"  - 计算BaseFeature（不标准化）...")

# ⚠️ 重要：粗粒度阶段不标准化
base_feature = originalFeature.BaseFeature(
    coarse_bars.copy(), 
    include_categories=include_categories,
    apply_norm=False  # 🔧 关键修改：不标准化！
)
features_df = base_feature.init_feature_df

print(f"  ✓ 组{i}完成: {len(features_df)} 个桶, {len(features_df.columns)} 个特征（原始值）")
```

#### 步骤3：在最终数据生成时统一标准化

在生成训练数据时，对所有特征统一标准化：

```python
# 在 dataload.py 的最后，生成最终数据时
def normalize_features_uniformly(df, feature_columns, rolling_window=2000):
    """
    统一标准化所有特征
    
    参数:
        df: 包含所有特征的DataFrame
        feature_columns: 需要标准化的特征列
        rolling_window: 滚动窗口大小
    """
    print(f"\n📊 统一标准化 {len(feature_columns)} 个特征...")
    
    normalized_df = df.copy()
    
    for col in feature_columns:
        if col in df.columns:
            normalized_df[col] = norm(df[col].values, rolling_window=rolling_window)
    
    print(f"✅ 标准化完成")
    return normalized_df


# 使用示例（在生成最终训练数据时）
# 假设 final_df 是合并了所有粗粒度特征的最终DataFrame
feature_cols = [col for col in final_df.columns if col.startswith('lgp_') or col.startswith('ori_')]

# 统一标准化
final_df = normalize_features_uniformly(final_df, feature_cols, rolling_window=2000)
```

### 方案2：使用全局标准化参数

#### 修改思路

1. 先用完整数据计算每个特征的rolling std
2. 保存这些参数
3. 在每组粗粒度计算时，使用预计算的参数

#### 实现代码

```python
# 新增：带参数的norm函数
def norm_with_params(x, global_std=None, rolling_window=2000):
    """
    使用全局参数进行标准化
    
    参数:
        x: 输入数据
        global_std: 预计算的标准差（如果为None则计算）
        rolling_window: 窗口大小
    
    返回:
        标准化后的数据, [标准差参数]
    """
    factors_data = pd.DataFrame(x, columns=['factor'])
    factors_data = factors_data.replace([np.inf, -np.inf, np.nan], 0.0)
    
    if global_std is None:
        # 计算模式：返回数据和参数
        factors_std = factors_data.rolling(window=rolling_window, min_periods=1).std()
        factor_value = factors_data / factors_std
        factor_value = factor_value.replace([np.inf, -np.inf, np.nan], 0.0)
        return np.nan_to_num(factor_value).flatten(), factors_std.values.flatten()
    else:
        # 使用模式：使用预计算的参数
        factors_std = pd.DataFrame(global_std, columns=['factor'])
        factor_value = factors_data / factors_std
        factor_value = factor_value.replace([np.inf, -np.inf, np.nan], 0.0)
        return np.nan_to_num(factor_value).flatten()
```

### 方案3：使用Expanding Window（最保守）

避免使用未来信息，改用expanding window：

```python
def norm_expanding(x, min_periods=100):
    """
    使用expanding window标准化（不使用未来信息）
    
    优点：
    - 不会有lookback bias
    - 每个点只使用历史数据
    
    缺点：
    - 前期标准化可能不稳定
    - 对市场regime变化适应慢
    """
    factors_data = pd.DataFrame(x, columns=['factor'])
    factors_data = factors_data.replace([np.inf, -np.inf, np.nan], 0.0)
    
    # 使用expanding而不是rolling
    factors_std = factors_data.expanding(min_periods=min_periods).std()
    factor_value = factors_data / factors_std
    factor_value = factor_value.replace([np.inf, -np.inf, np.nan], 0.0)
    
    return np.nan_to_num(factor_value).flatten()
```

## 🎯 推荐方案

**推荐使用方案1**，理由：

1. ✅ **实现简单**：只需修改几行代码
2. ✅ **效果最好**：所有数据使用统一标准化基准
3. ✅ **兼容性好**：不影响原有代码逻辑
4. ✅ **易于调试**：标准化在最后一步，容易检查

## 📋 完整修复流程

### 1. 备份原始代码

```bash
cp gp_crypto_next/originalFeature.py gp_crypto_next/originalFeature.py.backup
cp gp_crypto_next/dataload.py gp_crypto_next/dataload.py.backup
```

### 2. 修改 `originalFeature.py`

在 `BaseFeature.__init__()` 中添加 `apply_norm=True` 参数。

### 3. 修改 `dataload.py`

在第1250行左右，修改为：

```python
base_feature = originalFeature.BaseFeature(
    coarse_bars.copy(), 
    include_categories=include_categories,
    apply_norm=False  # 粗粒度不标准化
)
```

### 4. 在数据生成pipeline最后添加统一标准化

```python
# 在生成最终训练数据时
final_df = normalize_features_uniformly(final_df, feature_columns)
```

### 5. 测试验证

```python
# 测试代码
import pandas as pd
import numpy as np

# 生成测试数据
test_data = pd.DataFrame({
    'o': np.random.randn(1000),
    'h': np.random.randn(1000),
    'l': np.random.randn(1000),
    'c': np.random.randn(1000),
    'vol': np.random.randn(1000),
})

# 测试不标准化
features_raw = originalFeature.BaseFeature(test_data, apply_norm=False)
print("原始特征均值:", features_raw.init_feature_df.mean().mean())

# 测试标准化
features_norm = originalFeature.BaseFeature(test_data, apply_norm=True)
print("标准化特征均值:", features_norm.init_feature_df.mean().mean())
```

## 🔍 验证标准化一致性

检查修复后特征的一致性：

```python
def check_feature_consistency(df, feature_col):
    """检查特征在不同时间段的一致性"""
    # 分成3段
    n = len(df)
    seg1 = df[feature_col].iloc[:n//3]
    seg2 = df[feature_col].iloc[n//3:2*n//3]
    seg3 = df[feature_col].iloc[2*n//3:]
    
    print(f"特征 {feature_col} 的一致性检查:")
    print(f"  段1 均值/标准差: {seg1.mean():.4f} / {seg1.std():.4f}")
    print(f"  段2 均值/标准差: {seg2.mean():.4f} / {seg2.std():.4f}")
    print(f"  段3 均值/标准差: {seg3.mean():.4f} / {seg3.std():.4f}")
    
    # 理想情况：标准化后均值≈0，标准差≈1
```

## ⚠️ 注意事项

1. **训练TFT时的影响**：
   - 修复后特征分布会更稳定
   - 模型训练可能更快收敛
   - 预测性能可能提升

2. **历史模型兼容性**：
   - 用新方法训练的模型不能直接替换旧模型
   - 需要重新训练

3. **计算效率**：
   - 统一标准化只需计算一次
   - 比原方案更快

## 📚 参考资料

- [Feature Scaling Best Practices](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Time Series Data Leakage](https://machinelearningmastery.com/data-leakage-machine-learning/)
- [Rolling vs Expanding Windows](https://pandas.pydata.org/docs/reference/window.html)

---

**修复日期**: 2025-11-05  
**问题等级**: 🔴 严重  
**影响范围**: 所有使用粗粒度预计算的训练数据  
**修复优先级**: 高

