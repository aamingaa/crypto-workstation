# Temporal Fusion Transformer (TFT) 使用说明

本项目实现了完整的 Temporal Fusion Transformer 模型，用于加密货币时间序列预测。

## 📋 目录

- [快速开始](#快速开始)
- [安装依赖](#安装依赖)
- [文件说明](#文件说明)
- [使用方法](#使用方法)
- [配置说明](#配置说明)
- [模型架构](#模型架构)
- [最佳实践](#最佳实践)
- [常见问题](#常见问题)

## 🚀 快速开始

### 1. 安装依赖

```bash
# 安装 PyTorch (根据你的系统选择)
# CUDA 版本 (NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPU 版本
pip install torch torchvision torchaudio

# Mac M1/M2
pip install torch torchvision torchaudio

# 安装其他依赖
pip install -r requirements.txt
```

### 2. 准备数据

确保你的数据包含：
- GP生成的因子列（如 `gp_0`, `gp_1`, ...）
- 目标变量列（如 `label`）
- 时间索引

示例数据格式：
```
timestamp,gp_0,gp_1,gp_2,...,label
2025-01-01 00:00:00,0.5,-0.3,1.2,...,0.002
2025-01-01 00:15:00,0.6,-0.2,1.1,...,0.003
...
```

### 3. 运行训练

```bash
python tft_main.py
```

就这么简单！模型会自动：
- 加载数据
- 预处理特征
- 训练模型
- 保存结果和可视化

## 📦 文件说明

```
crypto-workstation/
├── model/
│   ├── temporal_fusion_transformer.py  # TFT模型架构
│   └── tft_data_processor.py          # 数据预处理
├── tft_main.py                         # 训练主程序
├── tft_config.yaml                     # 配置文件示例
├── TFT_使用说明.md                     # 本文档
└── requirements.txt                    # 依赖包（已更新）
```

## 📖 使用方法

### 方法一：使用默认配置（推荐新手）

```python
python tft_main.py
```

### 方法二：修改配置文件

1. 编辑 `tft_config.yaml`
2. 修改参数（如batch_size, hidden_size等）
3. 在代码中加载配置：

```python
import yaml

with open('tft_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 使用配置
BATCH_SIZE = config['training']['batch_size']
HIDDEN_SIZE = config['model']['hidden_size']
```

### 方法三：自定义代码

```python
from model.temporal_fusion_transformer import TemporalFusionTransformer
from model.tft_data_processor import TFTDataProcessor
import pandas as pd

# 1. 加载数据
df = pd.read_csv('your_data.csv')

# 2. 数据预处理
processor = TFTDataProcessor(
    target_column='label',
    encoder_length=60,
    decoder_length=10,
    batch_size=64
)

# 3. 准备特征
processed_df, feature_config = processor.prepare_data_from_gp_factors(
    df,
    factor_columns=['gp_0', 'gp_1', 'gp_2'],  # 你的GP因子
)

# 4. 创建数据集
train_dataset, val_dataset = processor.create_datasets(
    processed_df,
    feature_config
)

train_loader, val_loader = processor.create_dataloaders(
    train_dataset,
    val_dataset
)

# 5. 创建模型
model = TemporalFusionTransformer(
    observed_inputs=len(feature_config['observed']),
    known_regular_inputs=len(feature_config['known']),
    hidden_size=128,
    lstm_layers=2,
    num_attention_heads=4,
    encoder_length=60,
    decoder_length=10,
)

# 6. 训练
from model.temporal_fusion_transformer import TFTTrainer

trainer = TFTTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device='cuda',
)

history = trainer.train()
```

## ⚙️ 配置说明

### 关键参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `encoder_length` | 历史窗口长度 | 30-120 (15分钟数据：30=7.5小时) |
| `decoder_length` | 预测窗口长度 | 5-20 (15分钟数据：10=2.5小时) |
| `hidden_size` | 隐藏层大小 | 64-256 (越大越强大但越慢) |
| `lstm_layers` | LSTM层数 | 1-3 |
| `num_attention_heads` | 注意力头数 | 4-8 (必须能整除hidden_size) |
| `dropout` | Dropout率 | 0.1-0.3 |
| `batch_size` | 批大小 | 32-128 |
| `learning_rate` | 学习率 | 1e-4 ~ 1e-3 |

### 数据参数

```yaml
data:
  encoder_length: 60      # 使用60个时间步的历史数据
  decoder_length: 10      # 预测未来10个时间步
  stride_train: 1         # 训练时每次滑动1步（更多样本）
  stride_val: 5           # 验证时每次滑动5步（更快）
```

### 模型参数

```yaml
model:
  hidden_size: 128        # 隐藏层维度
  lstm_layers: 2          # LSTM层数
  num_attention_heads: 4  # 注意力头数
  dropout: 0.2            # Dropout率
```

## 🏗️ 模型架构

TFT 由以下组件构成：

### 1. 变量选择网络 (Variable Selection Network)
- 自动选择最重要的输入特征
- 提供特征重要性解释

### 2. LSTM 编码器-解码器
- **编码器**: 处理历史观测数据
- **解码器**: 生成未来预测

### 3. 多头注意力机制
- 捕捉不同时间步之间的依赖关系
- 提供时间注意力可视化

### 4. 门控残差网络 (GRN)
- 非线性特征变换
- 门控机制控制信息流

### 5. 静态协变量处理
- 处理不随时间变化的特征
- 上下文向量增强

## 💡 最佳实践

### 1. 数据预处理

```python
# ✅ 推荐：使用 Robust Scaler（对异常值不敏感）
processor = TFTDataProcessor(scaler_method='robust')

# ❌ 避免：直接使用原始数据
```

### 2. 窗口长度选择

```python
# 15分钟数据建议配置
encoder_length = 60   # 15小时历史
decoder_length = 10   # 2.5小时预测

# 1小时数据建议配置
encoder_length = 24   # 1天历史
decoder_length = 6    # 6小时预测
```

### 3. 特征选择

```python
# 使用高质量的GP因子
# ✅ 推荐：使用GP筛选后的因子（IC > 0.05）
gp_factors = [col for col in df.columns if col.startswith('gp_') and ic[col] > 0.05]

# ❌ 避免：使用所有原始特征（可能包含噪声）
```

### 4. 超参数调优顺序

1. **先调大框架**: `encoder_length`, `decoder_length`
2. **再调模型容量**: `hidden_size`, `lstm_layers`
3. **最后调训练参数**: `learning_rate`, `batch_size`, `dropout`

### 5. 训练技巧

```python
# 使用梯度裁剪防止梯度爆炸
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 使用学习率调度器
scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# 使用早停防止过拟合
if val_loss > best_val_loss:
    epochs_without_improvement += 1
    if epochs_without_improvement >= patience:
        break
```

## 🎯 评估指标

模型会自动计算以下指标：

| 指标 | 说明 | 越小越好/越大越好 |
|------|------|-------------------|
| MSE | 均方误差 | ↓ 越小越好 |
| MAE | 平均绝对误差 | ↓ 越小越好 |
| RMSE | 均方根误差 | ↓ 越小越好 |
| MAPE | 平均绝对百分比误差 | ↓ 越小越好 |
| R² | 决定系数 | ↑ 越大越好 |

### 如何判断模型好坏？

```python
# 好的模型
R² > 0.3         # 解释了30%以上的方差
MAPE < 5%        # 平均误差小于5%
val_loss < train_loss * 1.2  # 验证损失不超过训练损失的20%

# 需要改进
R² < 0.1         # 预测能力较弱
MAPE > 10%       # 误差较大
val_loss > train_loss * 2    # 严重过拟合
```

## 📊 可视化输出

训练完成后会生成：

### 1. 训练历史图 (`training_history.png`)
- 训练/验证损失曲线
- 学习率变化曲线

### 2. 预测结果图 (`predictions.png`)
- 时间序列对比图（真实值 vs 预测值）
- 散点图（预测精度可视化）

### 3. 注意力权重图
- 展示模型关注的时间步
- 帮助理解模型决策

## ❓ 常见问题

### Q1: CUDA out of memory (显存不足)

**解决方案：**
```python
# 方法1: 减小batch_size
batch_size = 32  # 或 16

# 方法2: 减小hidden_size
hidden_size = 64

# 方法3: 使用梯度累积
gradient_accumulation_steps = 4
```

### Q2: 训练太慢

**解决方案：**
```python
# 方法1: 增大stride_train
stride_train = 2  # 或 5

# 方法2: 减少num_workers
num_workers = 0

# 方法3: 使用混合精度训练
from torch.cuda.amp import autocast, GradScaler
```

### Q3: 过拟合（验证损失远大于训练损失）

**解决方案：**
```python
# 方法1: 增大dropout
dropout = 0.3  # 或 0.4

# 方法2: 使用更多数据
# 增加训练样本数量

# 方法3: 减小模型容量
hidden_size = 64
lstm_layers = 1

# 方法4: 早停
patience = 10
```

### Q4: 预测结果不佳

**解决方案：**
```python
# 1. 检查数据质量
df.isnull().sum()  # 检查缺失值
df.describe()      # 检查分布

# 2. 检查特征相关性
correlation = df[gp_factors].corrwith(df['label'])
print(correlation.sort_values(ascending=False))

# 3. 增加历史窗口
encoder_length = 120  # 增加到120

# 4. 尝试不同的损失函数
criterion = nn.L1Loss()  # MAE损失
```

### Q5: 如何用于实际交易？

```python
# 1. 加载训练好的模型
model.load_state_dict(torch.load('best_model.pth'))

# 2. 准备最新数据
latest_data = get_latest_data()  # 获取最新60个时间步

# 3. 预测
model.eval()
with torch.no_grad():
    prediction = model(latest_data)

# 4. 生成交易信号
if prediction > threshold:
    signal = 'BUY'
elif prediction < -threshold:
    signal = 'SELL'
else:
    signal = 'HOLD'
```

## 🔧 高级用法

### 1. 多步预测

```python
# TFT原生支持多步预测
decoder_length = 20  # 预测未来20步

# 输出形状: (batch, 20, 1)
predictions = model(historical_inputs, future_inputs)
```

### 2. 概率预测（分位数回归）

```python
from model.temporal_fusion_transformer import QuantileLoss

# 使用分位数损失
criterion = QuantileLoss(quantiles=[0.1, 0.5, 0.9])

# 输出: 10%, 50%, 90% 分位数预测
# 可以构建预测区间
```

### 3. 迁移学习

```python
# 在新币种上微调已训练的模型
model.load_state_dict(torch.load('eth_model.pth'))

# 冻结部分层
for param in model.lstm_encoder.parameters():
    param.requires_grad = False

# 只训练注意力和输出层
trainer = TFTTrainer(model, train_loader, val_loader)
trainer.train()
```

### 4. 集成多个模型

```python
models = [load_model(f'model_{i}.pth') for i in range(5)]

# 平均预测
predictions = []
for model in models:
    pred = model(inputs)
    predictions.append(pred)

ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
```

## 📚 参考资料

- 论文: [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363)
- PyTorch文档: https://pytorch.org/docs/stable/index.html
- 时间序列预测: https://github.com/unit8co/darts

## 📞 支持

如有问题，请：
1. 查看本文档的"常见问题"部分
2. 检查 `tft_config.yaml` 配置是否正确
3. 查看训练日志和错误信息

## 🎉 开始使用吧！

```bash
# 一键启动
python tft_main.py

# 等待训练完成，查看结果！
```

祝你训练愉快！🚀

