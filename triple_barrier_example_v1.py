# 最小示例：triple barrier → meta-labeling → bet sizing
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from label.triple_barrier import get_barrier, get_metalabel, get_wallet, show_results

# 1) 构造示例价格数据
np.random.seed(42)
dates = pd.date_range('2025-01-01', periods=600, freq='15T')
# print(str(dates))

prices = 100 * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.005))
close = pd.Series(prices, index=dates)

# 2) 生成入场点与主方向（主模型信号，可替换为你的主模型）
enter = close.index[::20]  # 每20根入一次
ret1 = close.pct_change()
side = pd.Series(np.where(ret1.rolling(10).mean() > 0, 1, -1), index=close.index).reindex(enter).fillna(1)

# 3) 目标波动率与 triple barrier
target = close.pct_change().rolling(48).std()  # 以波动率为“目标”单位
target = target.reindex(enter)

pt_sl = [2.0, 1.0]      # 盈利:止损 = 2:1
max_holding = [0, 32]   # 最多持仓 8 小时（32*15分钟）
barrier = get_barrier(close=close, enter=enter, pt_sl=pt_sl, max_holding=max_holding, target=target, side=side)

# 4) 元标签（z=1胜、0负；剔除ret=0事件）
meta_label = get_metalabel(barrier)  # index 是发生触碰或到时的事件

# 5) 构造简单特征（你可以替换为项目中的特征）
def build_features(close: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame(index=close.index)
    df['ret_1'] = close.pct_change()
    df['ret_4'] = close.pct_change(4)
    df['ret_16'] = close.pct_change(16)
    df['vol_48'] = df['ret_1'].rolling(48).std()
    df['mom_24'] = close.pct_change(24)
    df['sma_fast'] = close.rolling(16).mean() / close - 1
    df['sma_slow'] = close.rolling(64).mean() / close - 1
    return df

X_full = build_features(close)
# 只拿事件发生时（enter）的特征；并与有标签的样本求交集
X_events = X_full.reindex(enter)
X_train = X_events.reindex(meta_label.index)  # 和 meta_label 对齐
y_train = meta_label.astype(int)

# 去掉缺失
train_mask = ~X_train.isna().any(axis=1)
X_train = X_train.loc[train_mask]
y_train = y_train.loc[train_mask]

# 6) 训练元模型（简单 LogisticRegression）
clf = Pipeline(steps=[
    ('scaler', StandardScaler(with_mean=False)),  # 稀疏/相对尺度稳定
    ('lr', LogisticRegression(max_iter=200, class_weight='balanced'))
])
clf.fit(X_train, y_train)

# 7) 对所有入场事件预测胜率 p̂
X_pred = X_events.copy()
pred_mask = ~X_pred.isna().any(axis=1)
p_hat = pd.Series(np.nan, index=enter, name='p_hat')
p_hat.loc[pred_mask] = clf.predict_proba(X_pred.loc[pred_mask])[:, 1]

# 8) Bet sizing（Kelly 或半 Kelly）
# 赔率 b = 盈利幅度 / 亏损幅度（与 pt_sl 对应）
b = pt_sl[0] / max(pt_sl[1], 1e-8)

def kelly_fraction(p: float, b: float, half: bool = True) -> float:
    if np.isnan(p):
        return 0.0
    f = (p * b - (1 - p)) / max(b, 1e-8)
    f = np.clip(f, 0.0, 1.0)
    return 0.5 * f if half else f

f_series = p_hat.apply(lambda p: kelly_fraction(p, b, half=True))  # 半 Kelly 更稳健

# 9) 将头寸规模映射成资金下注额（示例：以初始资金的比例下注）
initial_money = 100000.0
unit_bet = initial_money  # 以账户净值为基准
bet_size = pd.Series(0.0, index=close.index, name='bet_size')
bet_size.loc[enter] = (f_series * unit_bet).fillna(0.0)

# 可选：加入最小阈值，过滤低置信度交易
threshold = 0.55
bet_size.loc[enter] = np.where(p_hat >= threshold, bet_size.loc[enter], 0.0)

# 10) 回放钱包
wallet = get_wallet(close=close, barrier=barrier, initial_money=initial_money, bet_size=bet_size)
show_results(wallet)

print("\n前几个事件：")
out_df = pd.concat([
    barrier[['exit', 'ret', 'side']].head(10),
    p_hat.rename('p_hat').to_frame().head(10),
    f_series.rename('kelly_half').to_frame().head(10),
    bet_size.reindex(enter).rename('bet_amt').to_frame().head(10)
], axis=1)
print(out_df)