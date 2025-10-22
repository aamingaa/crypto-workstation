# Triple Barrier 快速参考卡片

## 📌 快速开始（3行代码）

```python
strategy.load_data_from_dataload()
strategy.generate_triple_barrier_labels(pt_sl=[2, 2], max_holding=[0, 4])
# 现在 strategy.meta_labels 包含了 0/1 标签（0=亏损，1=盈利）
```

---

## 🎯 三个屏障是什么？

| 屏障类型 | 说明 | 示例 |
|---------|------|------|
| **上屏障**（Profit Taking） | 止盈线，达到目标收益退出 | 价格上涨2倍波动率 |
| **下屏障**（Stop Loss） | 止损线，达到止损幅度退出 | 价格下跌2倍波动率 |
| **垂直屏障**（Time Exit） | 时间限制，超时强制退出 | 最多持有4小时 |

**哪个先触碰，哪个退出！**

---

## 🔧 常用参数配置

### pt_sl（止盈/止损倍数）

```python
pt_sl=[1.5, 1.5]  # 保守：更频繁触发
pt_sl=[2.0, 2.0]  # 平衡：推荐起点
pt_sl=[3.0, 2.0]  # 激进：追求高盈亏比
pt_sl=[2.5, 1.5]  # 非对称：快止损慢止盈
```

### max_holding（最大持仓时间）

```python
# 15分钟K线
max_holding=[0, 2]   # 2小时 = 8根K线
max_holding=[0, 4]   # 4小时 = 16根K线（推荐）

# 1小时K线
max_holding=[0, 12]  # 12小时
max_holding=[1, 0]   # 1天

# 4小时K线
max_holding=[2, 0]   # 2天
```

---

## 📖 两种使用模式

### 模式1：仅生成标签（用于分析）

```python
strategy.generate_triple_barrier_labels(
    pt_sl=[2, 2],
    max_holding=[0, 4]
)

# 查看结果
print(strategy.meta_labels)        # 0/1 分类标签
print(strategy.barrier_results)    # 详细结果（收益、退出时间等）
```

### 模式2：替换训练目标（用于训练模型）

```python
strategy.generate_triple_barrier_labels(pt_sl=[2, 2], max_holding=[0, 4])
strategy.use_triple_barrier_as_y()  # ⚠️ 这会改变 y_train 和 y_test

# 现在模型训练的目标是 Triple Barrier 收益，而不是固定周期收益
```

---

## 💡 完整代码模板

### 模板1：添加到现有流程

```python
from multi_model_main import QuantTradingStrategy

strategy = (
    QuantTradingStrategy.from_yaml('config.yaml', 'factors.csv.gz')
    .load_data_from_dataload()
    .load_factor_expressions()
    .evaluate_factor_expressions()
    .normalize_factors()
    .select_factors()
    
    # ===== 在这里添加 Triple Barrier ===== #
    .generate_triple_barrier_labels(
        pt_sl=[2.0, 2.0],
        max_holding=[0, 4]
    )
    .use_triple_barrier_as_y()  # 可选
    # ===================================== #
    
    .prepare_training_data()
    .train_models()
    .make_predictions()
    .backtest_all_models()
)
```

### 模板2：运行示例脚本

```bash
# 方法1：直接运行示例
python triple_barrier_example.py

# 方法2：在 Python 中
from triple_barrier_example import example1_basic_usage
strategy = example1_basic_usage()
```

---

## 📊 结果分析

### 查看统计信息

```python
# 胜率
win_rate = (strategy.meta_labels == 1).sum() / len(strategy.meta_labels)
print(f"胜率: {win_rate:.2%}")

# 平均收益
avg_return = strategy.barrier_results['ret'].mean()
print(f"平均收益: {avg_return:.4f}")

# 收益分布
print(strategy.barrier_results['ret'].describe())
```

### 可视化

```python
from label.triple_barrier import get_wallet, plot

# 生成交易钱包
wallet = get_wallet(
    close=pd.Series(strategy.ohlc[:, 3], index=pd.to_datetime(strategy.z_index)),
    barrier=strategy.barrier_results,
    initial_money=10000
)

# 绘制交易记录
plot(close, strategy.barrier_results, wallet)
```

---

## ⚠️ 注意事项

| 问题 | 说明 | 解决方案 |
|------|------|----------|
| **前视偏差** | Triple Barrier 使用未来数据 | 训练时确保不使用未来信息 |
| **计算时间** | 需要遍历所有时间点 | 大数据集时可能较慢 |
| **参数敏感** | 不同参数影响很大 | 使用网格搜索优化 |
| **数据对齐** | 需要时间索引 | 确保数据有正确的时间戳 |

---

## 🔍 常见问题

**Q: 什么时候用 Triple Barrier？**
A: 当你想让模型学习"何时止盈止损"而不是"固定持有N期"。

**Q: 参数怎么选？**
A: 从 `pt_sl=[2, 2], max_holding=[0, 4]` 开始，根据回测结果调整。

**Q: meta_labels 是什么？**
A: 二分类标签（0=亏损，1=盈利），可用于训练分类模型预测"是否应该交易"。

**Q: 和固定周期收益有什么区别？**
A: 
- 固定周期：总是持有N期，无论盈亏
- Triple Barrier：达到止盈/止损/时间限制就退出，更贴近真实交易

**Q: 能在 GP 算法中用吗？**
A: 可以！在 `gp_crypto_next/main_gp_new.py` 的适应度函数中替换收益计算即可。

---

## 📚 相关文件

| 文件 | 说明 |
|------|------|
| `label/triple_barrier.py` | Triple Barrier 核心实现 |
| `multi_model_main.py` | 已集成 Triple Barrier 的策略类 |
| `triple_barrier_example.py` | 可运行的示例脚本 |
| `TRIPLE_BARRIER_使用说明.md` | 详细文档 |
| `TRIPLE_BARRIER_快速参考.md` | 本文档 |

---

## 🚀 立即尝试

```python
# 复制这段代码到 Python 或 Jupyter 中运行
from multi_model_main import QuantTradingStrategy

strategy = QuantTradingStrategy.from_yaml(
    'gp_crypto_next/coarse_grain_parameters.yaml',
    'gp_models/ETHUSDT_15m_1_2025-01-01_2025-01-20_2025-01-20_2025-01-31.csv.gz'
)

strategy.load_data_from_dataload()
strategy.generate_triple_barrier_labels(pt_sl=[2, 2], max_holding=[0, 4])

print(f"✅ 成功生成 {len(strategy.meta_labels)} 个标签")
print(f"胜率: {(strategy.meta_labels == 1).sum() / len(strategy.meta_labels):.2%}")
```

---

**需要帮助？** 查看 `TRIPLE_BARRIER_使用说明.md` 获取更详细的文档和示例。

