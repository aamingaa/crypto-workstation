# Crypto Trade - 基于cryptofeed的加密货币自动交易系统

## 🚀 系统架构

这是一个基于 `cryptofeed` 的实时加密货币自动交易系统，使用WebSocket实现策略信号广播和交易执行。

### 核心组件

- **`my_strategy.py`** - 基于cryptofeed的实时策略模块
- **`my_server.py`** - 集成策略的信号广播服务端
- **`my_trading_actions.py`** - Binance交易执行模块
- **`test_client.py`** - 测试客户端

## 📦 依赖安装

```bash
pip install -r requirements.txt
```

## 🔧 使用方法

### 1. 启动服务端（包含策略）

```bash
python my_server.py
```

服务端将：
- 启动WebSocket服务（端口8765）
- 订阅Binance的BTC-USDT实时K线数据
- 基于移动平均策略生成交易信号
- 每5秒广播最新信号

### 2. 启动测试客户端

```bash
python test_client.py
```

客户端将：
- 连接到服务端
- 接收并显示策略信号
- 模拟信号处理逻辑

### 3. 独立测试策略

```bash
python my_strategy.py
```

直接运行策略模块，查看实时K线数据和策略信号。

## 📊 策略说明

### 当前策略：移动平均交叉

- **买入条件**：5日均线上穿20日均线，且当前价格高于5日均线
- **卖出条件**：5日均线下穿20日均线，且当前价格低于5日均线
- **信号格式**：`BUY BTCUSDT 0.1` / `SELL BTCUSDT 0.1` / `HOLD BTCUSDT 0.0`

### 自定义策略

在 `my_strategy.py` 的 `OnBar` 方法中实现您的策略逻辑：

```python
def OnBar(self, kline_data: Dict) -> Optional[str]:
    # 您的策略逻辑
    if 买入条件:
        return f"BUY {self.symbol.replace('-', '')} 0.1"
    elif 卖出条件:
        return f"SELL {self.symbol.replace('-', '')} 0.1"
    return None
```

## 🔄 数据流

```
Binance WebSocket → cryptofeed → 策略分析 → 信号生成 → WebSocket广播 → 客户端接收 → 交易执行
```

## ⚠️ 注意事项

1. **API密钥**：当前使用公开数据，如需实盘交易请配置Binance API密钥
2. **风险控制**：建议在实盘使用前进行充分测试
3. **网络连接**：确保网络稳定，WebSocket连接可能因网络问题断开
4. **策略风险**：移动平均策略仅供参考，请根据实际情况调整

## 🛠️ 开发说明

### 添加新的交易所

```python
from cryptofeed.exchanges import Binance, Coinbase, Kraken

# 在CryptoStrategy中添加多个交易所
self.feed_handler.add_feed(
    Binance(symbols=['BTC-USDT'], channels=[CANDLES]),
    Coinbase(symbols=['BTC-USD'], channels=[CANDLES])
)
```

### 添加新的策略指标

```python
def calculate_rsi(self, prices, period=14):
    """计算RSI指标"""
    # RSI计算逻辑
    pass

def OnBar(self, kline_data):
    # 使用RSI等指标
    rsi = self.calculate_rsi(recent_prices)
    if rsi < 30:
        return f"BUY {self.symbol.replace('-', '')} 0.1"
```

## 📈 性能优化

1. **数据缓存**：策略模块自动缓存最近100根K线
2. **异步处理**：使用asyncio实现非阻塞操作
3. **线程分离**：策略运行在独立线程中，不影响WebSocket服务

## 🔍 故障排除

### 连接问题
- 检查网络连接
- 确认服务端端口8765未被占用
- 查看防火墙设置

### 数据问题
- 确认Binance API可用性
- 检查cryptofeed版本兼容性
- 查看错误日志

### 策略问题
- 验证策略逻辑
- 检查数据格式
- 测试信号生成
# crypto-trade
