# Triple Barrier 使用说明

## 什么是 Triple Barrier？

**Triple Barrier（三重屏障标注法）** 是一种先进的金融时间序列标注方法，来自 Marcos López de Prado 的《Advances in Financial Machine Learning》一书。

### 三个"屏障"

1. **上屏障（Profit Taking）** - 止盈线
   - 当价格上涨达到目标收益率时触发
   - 例如：波动率的2倍

2. **下屏障（Stop Loss）** - 止损线  
   - 当价格下跌达到止损幅度时触发
   - 例如：波动率的2倍

3. **垂直屏障（Time Exit）** - 时间限制
   - 持有到最大时间后强制退出
   - 例如：最多持有4小时

**哪个屏障先被触碰，就在那个时刻退出交易。**

### 为什么使用 Triple Barrier？

相比固定周期收益率标注（如"预测未来1根K线的收益"），Triple Barrier 有以下优势：

1. **更贴近真实交易**：考虑了止盈、止损和持仓时间限制
2. **减少噪音**：避免标签受到未来过远时期无关波动的影响
3. **动态调整**：根据市场波动率自适应调整止盈止损幅度
4. **生成 Meta-Label**：可用于二分类模型（交易 vs 不交易）

---

## 在您的代码中如何使用

### 方式1：生成 Triple Barrier 标签（用于分析）

```python
from multi_model_main import QuantTradingStrategy

# 创建策略
yaml_path = 'gp_crypto_next/coarse_grain_parameters.yaml'
factor_csv_path = 'gp_models/ETHUSDT_15m_1_2025-01-01_2025-01-20_2025-01-20_2025-01-31.csv.gz'

strategy = QuantTradingStrategy.from_yaml(
    yaml_path=yaml_path,
    factor_csv_path=factor_csv_path
)

# 加载数据
strategy.load_data_from_dataload()

# 生成 Triple Barrier 标签
strategy.generate_triple_barrier_labels(
    pt_sl=[2, 2],        # [止盈倍数, 止损倍数]，相对于波动率
    max_holding=[0, 4]   # [天数, 小时数]，最多持有4小时
)

# 查看结果
print(f"盈利交易占比: {(strategy.meta_labels == 1).sum() / len(strategy.meta_labels):.2%}")
print(f"平均收益: {strategy.barrier_results['ret'].mean():.4f}")
```

### 方式2：使用 Triple Barrier 作为训练目标

```python
# 完整流程示例
strategy = (
    QuantTradingStrategy.from_yaml(yaml_path, factor_csv_path)
    .load_data_from_dataload()
    .load_factor_expressions()
    .evaluate_factor_expressions()
    .normalize_factors()
    .select_factors()
    
    # 生成 Triple Barrier 标签
    .generate_triple_barrier_labels(
        pt_sl=[2.5, 2.0],     # 止盈2.5倍波动率，止损2倍波动率
        max_holding=[0, 6]    # 最多持有6小时
    )
    
    # 使用 Triple Barrier 收益替代固定周期收益
    .use_triple_barrier_as_y()
    
    # 继续训练模型
    .prepare_training_data()
    .train_models()
    .make_predictions(weight_method='equal')
    .backtest_all_models()
)

# 查看结果
strategy.get_performance_summary()
strategy.plot_results('Ensemble')
```

### 方式3：Meta-Labeling（元标注）

Meta-Labeling 是一种二阶模型策略：
1. **一级模型**：预测价格方向（做多/做空）
2. **二级模型**：预测是否应该交易（基于 meta_labels）

```python
# 先训练一级模型
strategy.run_full_pipeline()

# 获取一级模型的预测方向
primary_predictions = np.sign(strategy.predictions['LinearRegression']['test'])
side_series = pd.Series(primary_predictions, index=strategy.z_index[len(strategy.y_train):])

# 生成 Triple Barrier 标签（使用一级模型的方向）
strategy.generate_triple_barrier_labels(
    pt_sl=[2, 2],
    max_holding=[0, 4],
    side_prediction=side_series  # 使用一级模型的方向预测
)

# meta_labels 可以作为二级分类模型的目标
# 1 = 交易会盈利，0 = 交易会亏损
# 训练二级分类模型来预测是否应该执行交易
```

---

## 参数调优建议

### pt_sl 参数（止盈/止损倍数）

```python
# 保守策略（更频繁触发）
pt_sl=[1.5, 1.5]  # 止盈止损各1.5倍波动率

# 平衡策略
pt_sl=[2.0, 2.0]  # 止盈止损各2倍波动率

# 激进策略（等待更大波动）
pt_sl=[3.0, 2.0]  # 止盈3倍，止损2倍（追求更高盈亏比）

# 非对称策略
pt_sl=[2.5, 1.5]  # 止盈2.5倍，止损1.5倍
```

### max_holding 参数（最大持仓时间）

```python
# 根据您的交易频率调整

# 15分钟K线
max_holding=[0, 2]   # 2小时 = 8根K线
max_holding=[0, 4]   # 4小时 = 16根K线
max_holding=[0, 8]   # 8小时 = 32根K线

# 1小时K线  
max_holding=[0, 12]  # 12小时
max_holding=[1, 0]   # 1天

# 4小时K线
max_holding=[2, 0]   # 2天
max_holding=[5, 0]   # 5天
```

### 如何选择参数？

可以使用网格搜索找到最佳参数组合：

```python
from label.triple_barrier import grid_pt_sl

# 定义参数范围
pt_list = [1.5, 2.0, 2.5, 3.0]
sl_list = [1.0, 1.5, 2.0, 2.5]

# 网格搜索
results = grid_pt_sl(
    pt=pt_list,
    sl=sl_list,
    close=strategy.close_series,
    enter=strategy.close_series.index,
    max_holding=[0, 4],
    side=None
)

# 查看哪个组合累计收益最高
print(results)
```

---

## 集成位置说明

### 在 multi_model_main.py 中

Triple Barrier 已经集成到 `QuantTradingStrategy` 类中，可以在以下位置调用：

```
数据加载
  ↓
load_data_from_dataload()
  ↓
load_factor_expressions()
  ↓
evaluate_factor_expressions()
  ↓
normalize_factors()
  ↓
select_factors()
  ↓
【在这里添加 Triple Barrier】← generate_triple_barrier_labels()
  ↓                          ↓
【可选】use_triple_barrier_as_y()
  ↓
prepare_training_data()
  ↓
train_models()
  ↓
make_predictions()
  ↓
backtest_all_models()
```

### 在 GP 遗传算法中

如果想在 GP 算法中使用 Triple Barrier，需要修改 `gp_crypto_next/main_gp_new.py`：

```python
# 在 GPAnalyzer 的 fitness 计算部分
# 将固定周期收益替换为 Triple Barrier 收益

from label.triple_barrier import get_barrier

# 在计算适应度函数时
barrier_results = get_barrier(
    close=close_prices,
    enter=enter_points,
    pt_sl=[2, 2],
    max_holding=[0, 4]
)

# 使用 barrier_results['ret'] 作为目标收益
```

---

## 可视化 Triple Barrier 结果

```python
from label.triple_barrier import get_wallet, plot

# 生成交易钱包
wallet = get_wallet(
    close=strategy.close_series,
    barrier=strategy.barrier_results,
    initial_money=10000
)

# 绘制交易结果（包含每笔交易的盈亏）
plot(
    close=strategy.close_series,
    barrier=strategy.barrier_results,
    wallet=wallet
)
```

---

## 注意事项

1. **数据泄露风险**：Triple Barrier 使用未来数据（知道何时触碰屏障），训练时要注意避免前视偏差

2. **计算成本**：Triple Barrier 需要遍历每个时间点计算退出时机，对于大数据集可能较慢

3. **参数敏感性**：不同的 pt_sl 和 max_holding 参数会显著影响标签分布

4. **适用场景**：
   - ✅ 适合：趋势跟踪、波段交易
   - ⚠️  需谨慎：高频交易（时间粒度太小）、长期持仓策略

---

## 参考资料

- 《Advances in Financial Machine Learning》 by Marcos López de Prado
- Chapter 3: Labeling (Triple-Barrier Method)
- Chapter 3: Meta-Labeling

---

## 示例：完整工作流程

```python
#!/usr/bin/env python3
"""
使用 Triple Barrier 的完整示例
"""

from multi_model_main import QuantTradingStrategy
import matplotlib.pyplot as plt

# 配置
yaml_path = 'gp_crypto_next/coarse_grain_parameters.yaml'
factor_csv_path = 'gp_models/ETHUSDT_15m_1_2025-01-01_2025-01-20_2025-01-20_2025-01-31.csv.gz'

# 策略配置
strategy_config = {
    'corr_threshold': 0.5,
    'max_factors': 30,
    'fees_rate': 0.0005,
    'model_save_path': './models',
}

# 创建策略并运行完整流程（使用 Triple Barrier）
strategy = (
    QuantTradingStrategy.from_yaml(
        yaml_path=yaml_path,
        factor_csv_path=factor_csv_path,
        strategy_config=strategy_config
    )
    .load_data_from_dataload()
    .load_factor_expressions()
    .evaluate_factor_expressions()
    .normalize_factors()
    .select_factors()
    
    # === Triple Barrier 标注 ===
    .generate_triple_barrier_labels(
        pt_sl=[2.0, 2.0],       # 止盈止损各2倍波动率
        max_holding=[0, 4]      # 最多持有4小时
    )
    .use_triple_barrier_as_y()  # 使用 Triple Barrier 收益作为目标
    
    # === 继续训练和回测 ===
    .prepare_training_data()
    .train_models()
    .make_predictions(weight_method='equal')
    .backtest_all_models()
)

# 显示结果
print("\n" + "="*60)
print("模型绩效汇总")
print("="*60)
strategy.get_performance_summary()

print("\n" + "="*60)
print("Triple Barrier 统计")
print("="*60)
print(f"总交易次数: {len(strategy.barrier_results)}")
print(f"盈利交易数: {(strategy.meta_labels == 1).sum()}")
print(f"胜率: {(strategy.meta_labels == 1).sum() / len(strategy.meta_labels):.2%}")
print(f"平均收益: {strategy.barrier_results['ret'].mean():.4f}")

# 绘制结果
strategy.plot_results('Ensemble')
plt.show(block=True)

# 保存模型
strategy.save_models()
```

运行这个脚本，您就可以看到使用 Triple Barrier 标注后的模型效果！

