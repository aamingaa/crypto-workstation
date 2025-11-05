# 🚀 Temporal Fusion Transformer (TFT) 实现

这是一个完整的、生产级的 Temporal Fusion Transformer 实现，专门为加密货币时间序列预测优化。

## ✨ 特性

- ✅ **完整的TFT架构实现**：包括变量选择网络、多头注意力、门控残差网络
- ✅ **与GP因子无缝集成**：自动处理GP生成的因子特征
- ✅ **多步预测**：支持预测未来多个时间步
- ✅ **可解释性**：提供注意力权重和变量重要性可视化
- ✅ **易于使用**：一行命令即可开始训练
- ✅ **灵活配置**：通过YAML文件轻松调整所有参数
- ✅ **完善的文档**：包含详细使用说明和最佳实践

## 📁 文件结构

```
crypto-workstation/
├── model/
│   ├── temporal_fusion_transformer.py  # TFT模型核心实现
│   └── tft_data_processor.py          # 数据预处理模块
├── tft_main.py                         # 完整训练主程序
├── tft_quick_start.py                  # 快速开始示例
├── tft_config.yaml                     # 配置文件模板
├── TFT_README.md                       # 本文件
└── TFT_使用说明.md                     # 详细使用文档
```

## 🏃 快速开始

### 1. 安装依赖

```bash
# 安装 PyTorch (根据你的系统选择合适的版本)
pip install torch torchvision torchaudio

# 安装其他依赖
pip install -r requirements.txt
```

### 2. 准备数据

确保你的数据包含：
- GP生成的因子列
- 目标变量列（如 `label`）

### 3. 运行训练

```bash
# 方式1: 完整训练
python tft_main.py

# 方式2: 快速测试
python tft_quick_start.py
```

就这么简单！🎉

## 📊 模型架构

TFT 是 Google 在 2019 年发表的先进时间序列模型，具有以下优势：

### 核心组件

1. **变量选择网络 (VSN)**
   - 自动识别重要特征
   - 减少噪声影响
   - 提供特征重要性解释

2. **静态协变量编码器**
   - 处理不随时间变化的特征
   - 提供全局上下文信息

3. **LSTM 编码器-解码器**
   - **编码器**: 提取历史模式
   - **解码器**: 生成未来预测

4. **多头注意力机制**
   - 捕捉时间依赖关系
   - 提供可解释的注意力权重

5. **门控残差网络 (GRN)**
   - 深层非线性变换
   - 门控机制控制信息流
   - 残差连接防止梯度消失

### 架构图

```
输入数据
  ↓
[变量选择网络] → 选择重要特征
  ↓
[静态上下文] → 全局信息
  ↓
[LSTM编码器] → 提取历史模式
  ↓
[LSTM解码器] → 生成初步预测
  ↓
[多头注意力] → 捕捉时间依赖
  ↓
[位置前馈网络] → 精细调整
  ↓
[输出层] → 最终预测
```

## 🎯 使用场景

### 1. 价格预测
```python
TARGET_COLUMN = 'close_return'  # 收益率
ENCODER_LENGTH = 60             # 15小时历史
DECODER_LENGTH = 10             # 2.5小时预测
```

### 2. 波动率预测
```python
TARGET_COLUMN = 'volatility'
ENCODER_LENGTH = 120
DECODER_LENGTH = 20
```

### 3. 量化因子预测
```python
TARGET_COLUMN = 'gp_best_factor'  # 最佳GP因子
ENCODER_LENGTH = 48
DECODER_LENGTH = 12
```

## ⚙️ 配置参数

### 快速配置

| 场景 | hidden_size | lstm_layers | batch_size | learning_rate |
|------|-------------|-------------|------------|---------------|
| 快速原型 | 64 | 1 | 128 | 1e-3 |
| 平衡性能 | 128 | 2 | 64 | 1e-3 |
| 高精度 | 256 | 3 | 32 | 1e-4 |
| 内存受限 | 32 | 1 | 16 | 1e-3 |

### 详细配置

查看 `tft_config.yaml` 获取所有可配置参数的详细说明。

## 📈 训练流程

### 自动化流程

```
1. 数据加载
   ├─ 读取CSV文件
   └─ 识别GP因子

2. 数据预处理
   ├─ 特征缩放（Robust Scaler）
   ├─ 时间特征生成
   ├─ 滑动窗口切分
   └─ 训练/验证集分割

3. 模型训练
   ├─ 自动早停
   ├─ 学习率调度
   ├─ 梯度裁剪
   └─ 最佳模型保存

4. 结果可视化
   ├─ 训练历史曲线
   ├─ 预测对比图
   ├─ 注意力权重热图
   └─ 特征重要性图

5. 模型保存
   ├─ 模型权重 (.pth)
   ├─ 数据缩放器 (.pkl)
   ├─ 配置文件 (.yaml)
   └─ 可视化图表 (.png)
```

## 📊 输出结果

训练完成后会生成：

```
tft_results_20250105_123456/
├── checkpoints/
│   └── best_model.pth              # 最佳模型权重
├── config.yaml                      # 训练配置
├── scaler.pkl                       # 数据缩放器
├── training_history.png             # 训练曲线
└── predictions.png                  # 预测结果
```

## 🎓 核心代码示例

### 最简示例

```python
from model.temporal_fusion_transformer import TemporalFusionTransformer
from model.tft_data_processor import TFTDataProcessor

# 1. 数据准备
processor = TFTDataProcessor(
    target_column='label',
    encoder_length=60,
    decoder_length=10
)

# 2. 模型创建
model = TemporalFusionTransformer(
    observed_inputs=20,
    known_regular_inputs=4,
    hidden_size=128
)

# 3. 训练
trainer = TFTTrainer(model, train_loader, val_loader)
history = trainer.train()
```

### 预测示例

```python
# 加载模型
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# 预测
with torch.no_grad():
    outputs = model(historical_inputs, future_inputs)
    predictions = outputs['predictions']
    attention = outputs['attention_weights']
```

## 🔬 高级功能

### 1. 注意力可视化

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 获取注意力权重
attention_weights = outputs['attention_weights']

# 绘制热图
plt.figure(figsize=(12, 8))
sns.heatmap(attention_weights[0].cpu().numpy(), cmap='viridis')
plt.title('Attention Weights')
plt.xlabel('Key Position')
plt.ylabel('Query Position')
plt.show()
```

### 2. 特征重要性分析

```python
# 获取变量选择权重
variable_weights = outputs['historical_variable_weights']

# 分析最重要的特征
importance = variable_weights.mean(dim=[0, 1])
top_features = torch.topk(importance, k=10)
```

### 3. 概率预测

```python
from model.temporal_fusion_transformer import QuantileLoss

# 使用分位数损失
criterion = QuantileLoss(quantiles=[0.1, 0.5, 0.9])

# 获得预测区间
predictions = model(inputs)
lower_bound = predictions[:, :, 0]  # 10%分位数
median = predictions[:, :, 1]       # 50%分位数
upper_bound = predictions[:, :, 2]  # 90%分位数
```

## 💡 最佳实践

### 数据准备

✅ **推荐做法**
- 使用高质量的GP筛选因子（IC > 0.05）
- 使用Robust Scaler处理异常值
- 进行Winsorization去除极端值
- 确保数据无缺失值

❌ **避免**
- 使用所有原始特征（包含噪声）
- 直接使用未标准化的数据
- 忽略异常值处理

### 模型训练

✅ **推荐做法**
- 使用早停防止过拟合
- 使用学习率调度器
- 使用梯度裁剪（max_norm=1.0）
- 监控训练/验证损失比例

❌ **避免**
- 训练过多epoch导致过拟合
- 使用过大的学习率
- 忽略梯度爆炸问题

### 超参数调优

✅ **推荐顺序**
1. 先调整数据窗口（encoder_length, decoder_length）
2. 再调整模型容量（hidden_size, lstm_layers）
3. 最后调整训练参数（learning_rate, dropout）

## 📚 技术细节

### 计算复杂度

- **时间复杂度**: O(L² · d · h)
  - L: 序列长度
  - d: 特征维度
  - h: 隐藏层大小

- **空间复杂度**: O(B · L · h)
  - B: 批大小

### 内存占用估算

| 配置 | 模型参数 | GPU内存 |
|------|----------|---------|
| 小型 (h=64) | ~100K | 2-3 GB |
| 中型 (h=128) | ~400K | 4-6 GB |
| 大型 (h=256) | ~1.6M | 8-12 GB |

## 🐛 常见问题

### Q: CUDA out of memory

**A:** 减小 `batch_size` 或 `hidden_size`，或使用梯度累积

### Q: 训练太慢

**A:** 增大 `stride_train`，减小 `num_workers`，或使用更小的模型

### Q: 过拟合

**A:** 增大 `dropout`，减小模型容量，或使用更多数据

### Q: 预测效果不好

**A:** 检查数据质量，增加历史窗口，或尝试不同的损失函数

详细解决方案请查看 `TFT_使用说明.md`

## 📖 参考文献

1. **TFT原论文**:
   - Lim, B., et al. (2021). "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting." International Journal of Forecasting.
   - 论文链接: https://arxiv.org/abs/1912.09363

2. **相关技术**:
   - Attention Is All You Need (Transformer)
   - LSTM Networks
   - Gated Residual Networks

## 🤝 贡献

欢迎提出改进建议！

## 📄 许可

本实现基于原论文和开源实现进行优化和改进。

---

## 🎯 下一步

1. **立即开始**: `python tft_quick_start.py`
2. **完整训练**: `python tft_main.py`
3. **阅读详细文档**: `TFT_使用说明.md`
4. **调整配置**: 编辑 `tft_config.yaml`
5. **部署生产**: 集成到你的交易系统

祝你使用愉快！🚀

---

**创建日期**: 2025-11-05  
**版本**: 1.0.0  
**作者**: AI Assistant

