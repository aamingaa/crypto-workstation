"""
Triple Barrier 方法使用示例

Triple Barrier 是一种标签方法，用于机器学习中的金融时间序列预测。
它会创建以下三个"障碍"：
1. 盈利目标 (Profit Taking)
2. 止损 (Stop Loss)  
3. 垂直障碍 - 最大持仓期 (Vertical Barrier)

当价格触及其中任意一个障碍时，交易结束。
"""

import numpy as np
import pandas as pd
from label.triple_barrier import (
    get_barrier, 
    get_wallet, 
    get_metalabel,
    plot,
    show_results
)

# ==================== 1. 准备示例数据 ====================
print("=" * 60)
print("1. 准备示例价格数据")
print("=" * 60)

# 生成模拟价格数据
dates = pd.date_range('2025-01-01', periods=100, freq='1H')
np.random.seed(42)
prices = 100 * np.exp(np.cumsum(np.random.randn(100) * 0.01))
close = pd.Series(prices, index=dates)

print(f"价格数据范围: {len(close)} 个数据点")
print(f"起始价格: {close.iloc[0]:.2f}")
print(f"结束价格: {close.iloc[-1]:.2f}")
print(f"\n前5个数据点:")
print(close.head())

# ==================== 2. 定义入场点 ====================
print("\n" + "=" * 60)
print("2. 定义入场点（enter points）")
print("=" * 60)

# 每隔10个小时产生一个交易信号
enter_points = close.index[::10]
print(f"入场点数量: {len(enter_points)}")
print(f"\n前5个入场点:")
print(enter_points[:5])

# ==================== 3. 定义目标波动率 ====================
print("\n" + "=" * 60)
print("3. 定义目标波动率（target）")
print("=" * 60)

# 目标波动率：基于历史波动计算
target = close.rolling(window=20).std()
target = target.reindex(enter_points)
print(f"目标波动率（前5个）:")
print(target.head())

# ==================== 4. 定义交易方向 ====================
print("\n" + "=" * 60)
print("4. 定义交易方向（side）")
print("=" * 60)

# 1 = 做多, -1 = 做空
# 这里使用简单策略：价格上涨趋势时做多，下跌趋势时做空
returns = close.pct_change()
side = pd.Series(
    np.where(returns > 0, 1, -1),
    index=close.index
).reindex(enter_points).fillna(1)

print(f"交易方向（前5个）:")
print(side.head())

# ==================== 5. 应用 Triple Barrier ====================
print("\n" + "=" * 60)
print("5. 应用 Triple Barrier")
print("=" * 60)

# pt_sl: [盈利目标倍数, 止损倍数]
# 如果 target 未定义，pt_sl 是绝对价格变动
# 如果 target 已定义，pt_sl 是相对于 target 的倍数
pt_sl = [2.0, 1.0]  # 盈利目标是止损的2倍

# max_holding: [天数, 小时数] - 最大持仓期
max_holding = [0, 10]  # 最多持仓10小时

barrier = get_barrier(
    close=close,
    enter=enter_points,
    pt_sl=pt_sl,
    max_holding=max_holding,
    target=target,
    side=side
)

print(f"生成的 barrier DataFrame:")
print(barrier.head())
print(f"\nBarrier 形状: {barrier.shape}")
print(f"交易数量: {len(barrier)}")
print(f"有退出点的交易数: {barrier['exit'].notna().sum()}")

# 检查收益率
print(f"\n收益率统计:")
print(barrier['ret'].describe())

# ==================== 6. 计算钱包（模拟交易） ====================
print("\n" + "=" * 60)
print("6. 计算钱包（模拟交易）")
print("=" * 60)

# 初始资金
initial_money = 10000

# 每次交易金额
bet_size = pd.Series(1000, index=close.index)  # 每次交易1000元

wallet = get_wallet(
    close=close,
    barrier=barrier,
    initial_money=initial_money,
    bet_size=bet_size
)

print(f"钱包 DataFrame（前5行）:")
print(wallet.head())

# ==================== 7. 显示结果 ====================
print("\n" + "=" * 60)
print("7. 交易结果汇总")
print("=" * 60)
show_results(wallet)

# ==================== 8. 生成元标签 ====================
print("\n" + "=" * 60)
print("8. 生成元标签（Meta-label）")
print("=" * 60)
print("元标签用于训练二分类模型：是否应该执行这笔交易")

meta_label = get_metalabel(barrier)
print(f"元标签统计:")
print(f"盈利次数（1）: {meta_label.sum()}")
print(f"亏损次数（0）: {meta_label.shape[0] - meta_label.sum()}")
print(f"盈亏比: {(meta_label.sum() / (meta_label.shape[0] - meta_label.sum())):.2f}")
print(f"\n前10个标签:")
print(meta_label.head(10))

# ==================== 9. 可视化（可选） ====================
print("\n" + "=" * 60)
print("9. 可视化结果（取消注释以显示）")
print("=" * 60)
print("# 取消下面一行的注释以显示可视化结果:")
print("# plot(close, barrier, wallet)")

# ==================== 10. 参数优化示例 ====================
print("\n" + "=" * 60)
print("10. 不同参数的比较")
print("=" * 60)

# 测试不同的盈利目标/止损比例
test_configs = [
    ([2.0, 1.0], "盈利:止损 = 2:1"),
    ([1.5, 1.0], "盈利:止损 = 1.5:1"),
    ([1.0, 1.0], "盈利:止损 = 1:1"),
]

results = []
for pt_sl_val, description in test_configs:
    test_barrier = get_barrier(
        close=close,
        enter=enter_points,
        pt_sl=pt_sl_val,
        max_holding=max_holding,
        target=target,
        side=side
    )
    
    total_return = test_barrier['ret'].sum()
    win_rate = (test_barrier['ret'] > 0).sum() / len(test_barrier[test_barrier['ret'] != 0])
    sharpe = test_barrier['ret'].mean() / test_barrier['ret'].std() if test_barrier['ret'].std() != 0 else 0
    
    results.append({
        '配置': description,
        '总收益': total_return,
        '胜率': win_rate,
        '夏普比': sharpe
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

print("\n" + "=" * 60)
print("示例完成！")
print("=" * 60)
