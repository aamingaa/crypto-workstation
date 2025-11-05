"""
Temporal Fusion Transformer (TFT) 实现

基于论文: Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
参考: https://arxiv.org/abs/1912.09363

主要特性:
1. 多视野预测
2. 静态协变量、已知输入、观测输入的处理
3. 变量选择网络
4. 门控残差网络 (GRN)
5. 多头注意力机制
6. 可解释性输出
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class TimeDistributed(nn.Module):
    """将模块应用于时间序列的每个时间步"""
    
    def __init__(self, module: nn.Module, batch_first: bool = True):
        super().__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        # 重塑输入: (batch, time, features) -> (batch * time, features)
        x_reshape = x.contiguous().view(-1, x.size(-1))
        y = self.module(x_reshape)
        
        # 恢复时间维度: (batch * time, output) -> (batch, time, output)
        if self.batch_first:
            y = y.contiguous().view(x.size(0), x.size(1), -1)
        else:
            y = y.view(x.size(1), x.size(0), -1)
        return y


class GatedResidualNetwork(nn.Module):
    """
    门控残差网络 (Gated Residual Network, GRN)
    
    包含:
    - 非线性变换
    - 门控机制
    - 残差连接
    - Layer Normalization
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.1,
        context_size: Optional[int] = None,
        batch_first: bool = True
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.context_size = context_size
        self.batch_first = batch_first

        # 主网络
        if self.input_size != self.output_size:
            self.skip_layer = TimeDistributed(
                nn.Linear(self.input_size, self.output_size),
                batch_first=batch_first
            )
        
        # ELU激活的全连接层
        self.fc1 = TimeDistributed(
            nn.Linear(self.input_size, self.hidden_size),
            batch_first=batch_first
        )
        
        # 上下文处理
        if self.context_size is not None:
            self.context_layer = TimeDistributed(
                nn.Linear(self.context_size, self.hidden_size, bias=False),
                batch_first=batch_first
            )
        
        self.fc2 = TimeDistributed(
            nn.Linear(self.hidden_size, self.output_size),
            batch_first=batch_first
        )
        
        # 门控层
        self.gate = TimeDistributed(
            nn.Linear(self.hidden_size, self.output_size),
            batch_first=batch_first
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.output_size)

    def forward(self, x, context=None):
        # 残差连接
        if self.input_size != self.output_size:
            residual = self.skip_layer(x)
        else:
            residual = x

        # 主路径
        x = self.fc1(x)
        
        # 添加上下文
        if context is not None and self.context_size is not None:
            context = self.context_layer(context)
            x = x + context
            
        x = F.elu(x)
        x = self.fc2(x)
        
        # 门控机制
        gate = self.gate(x)
        gate = torch.sigmoid(gate)
        
        x = x * gate
        x = self.dropout(x)
        
        # 残差 + LayerNorm
        x = x + residual
        x = self.layer_norm(x)
        
        return x


class VariableSelectionNetwork(nn.Module):
    """
    变量选择网络
    
    使用softmax注意力机制选择最重要的变量
    """
    
    def __init__(
        self,
        input_size: int,
        num_inputs: int,
        hidden_size: int,
        dropout: float = 0.1,
        context_size: Optional[int] = None
    ):
        super().__init__()
        self.input_size = input_size
        self.num_inputs = num_inputs
        self.hidden_size = hidden_size
        self.context_size = context_size

        # 变量特征变换
        self.flattened_grn = GatedResidualNetwork(
            input_size=self.num_inputs * self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.num_inputs,
            dropout=dropout,
            context_size=self.context_size,
            batch_first=True
        )

        # 单个变量处理
        self.single_variable_grns = nn.ModuleList([
            GatedResidualNetwork(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                output_size=self.hidden_size,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(self.num_inputs)
        ])

    def forward(self, embedding, context=None):
        """
        embedding: (batch, time, num_inputs, input_size)
        """
        batch_size, time_steps, num_inputs, input_size = embedding.size()

        # 展平所有变量
        flatten = embedding.view(batch_size, time_steps, -1)
        
        # 变量选择权重
        mlp_outputs = self.flattened_grn(flatten, context)
        sparse_weights = F.softmax(mlp_outputs, dim=-1)
        sparse_weights = sparse_weights.unsqueeze(2)

        # 处理单个变量
        processed_inputs = []
        for i in range(num_inputs):
            processed_inputs.append(
                self.single_variable_grns[i](embedding[:, :, i, :])
            )
        
        processed_inputs = torch.stack(processed_inputs, dim=-1)

        # 加权组合
        outputs = processed_inputs * sparse_weights.transpose(-1, -2)
        outputs = outputs.sum(dim=-1)

        return outputs, sparse_weights


class InterpretableMultiHeadAttention(nn.Module):
    """
    可解释的多头注意力机制
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        self.qkv_linears = nn.Linear(d_model, 3 * d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # 生成 Q, K, V
        qkv = self.qkv_linears(x)
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, 3 * self.d_head)
        qkv = qkv.permute(0, 2, 1, 3)
        
        q, k, v = qkv.chunk(3, dim=-1)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_head)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力
        out = torch.matmul(attention_weights, v)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.reshape(batch_size, seq_len, self.d_model)
        out = self.output_linear(out)
        
        return out, attention_weights


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer 主模型
    
    参数:
        static_variables: 静态变量数量 (不随时间变化)
        known_regular_inputs: 已知输入变量数量 (未来已知)
        known_categorical_inputs: 已知类别变量数量
        observed_inputs: 观测输入变量数量 (只能在历史观测到)
        output_size: 输出维度
        hidden_size: 隐藏层大小
        lstm_layers: LSTM层数
        num_attention_heads: 注意力头数
        dropout: Dropout率
        encoder_length: 编码器长度 (历史窗口)
        decoder_length: 解码器长度 (预测窗口)
    """
    
    def __init__(
        self,
        static_variables: int = 0,
        known_regular_inputs: int = 0,
        known_categorical_inputs: int = 0,
        observed_inputs: int = 1,
        output_size: int = 1,
        hidden_size: int = 64,
        lstm_layers: int = 2,
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        encoder_length: int = 30,
        decoder_length: int = 5,
    ):
        super().__init__()
        
        self.static_variables = static_variables
        self.known_regular_inputs = known_regular_inputs
        self.known_categorical_inputs = known_categorical_inputs
        self.observed_inputs = observed_inputs
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        
        # 输入嵌入层
        self.input_size = observed_inputs + known_regular_inputs
        
        # 静态变量处理
        if static_variables > 0:
            self.static_context_grn = GatedResidualNetwork(
                input_size=static_variables,
                hidden_size=hidden_size,
                output_size=hidden_size,
                dropout=dropout,
                batch_first=True
            )
            self.static_enrichment_grn = GatedResidualNetwork(
                input_size=hidden_size,
                hidden_size=hidden_size,
                output_size=hidden_size,
                dropout=dropout,
                batch_first=True
            )
        
        # 历史输入的变量选择网络
        self.historical_variable_selection = VariableSelectionNetwork(
            input_size=1,
            num_inputs=self.input_size,
            hidden_size=hidden_size,
            dropout=dropout,
            context_size=hidden_size if static_variables > 0 else None
        )
        
        # 未来输入的变量选择网络
        if known_regular_inputs > 0:
            self.future_variable_selection = VariableSelectionNetwork(
                input_size=1,
                num_inputs=known_regular_inputs,
                hidden_size=hidden_size,
                dropout=dropout,
                context_size=hidden_size if static_variables > 0 else None
            )
        
        # LSTM编码器-解码器
        self.lstm_encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True
        )
        
        self.lstm_decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # 门控跳跃连接
        self.gate_add_norm = nn.ModuleList([
            GatedResidualNetwork(
                input_size=hidden_size,
                hidden_size=hidden_size,
                output_size=hidden_size,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(3)
        ])
        
        # 多头注意力
        self.multihead_attn = InterpretableMultiHeadAttention(
            d_model=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        # 位置前馈网络
        self.position_wise_grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout=dropout,
            batch_first=True
        )
        
        # 输出层
        self.output_layer = TimeDistributed(
            nn.Linear(hidden_size, output_size),
            batch_first=True
        )

    def forward(
        self,
        historical_inputs: torch.Tensor,
        future_inputs: Optional[torch.Tensor] = None,
        static_inputs: Optional[torch.Tensor] = None,
    ):
        """
        前向传播
        
        参数:
            historical_inputs: (batch, encoder_length, observed_inputs + known_inputs)
            future_inputs: (batch, decoder_length, known_inputs)
            static_inputs: (batch, static_variables)
            
        返回:
            predictions: (batch, decoder_length, output_size)
            attention_weights: 注意力权重
            variable_selection_weights: 变量选择权重
        """
        batch_size = historical_inputs.size(0)
        
        # 1. 处理静态变量
        static_context = None
        if self.static_variables > 0 and static_inputs is not None:
            static_context = self.static_context_grn(static_inputs)
            static_context = static_context.unsqueeze(1)  # 添加时间维度
        
        # 2. 历史变量选择
        # 将输入转换为 (batch, time, num_vars, 1) 格式
        historical_inputs_reshaped = historical_inputs.unsqueeze(-1)
        historical_features, hist_var_weights = self.historical_variable_selection(
            historical_inputs_reshaped,
            static_context
        )
        
        # 3. 未来变量选择
        if future_inputs is not None and self.known_regular_inputs > 0:
            future_inputs_reshaped = future_inputs.unsqueeze(-1)
            future_features, future_var_weights = self.future_variable_selection(
                future_inputs_reshaped,
                static_context
            )
        else:
            # 如果没有未来输入，创建零张量
            future_features = torch.zeros(
                batch_size,
                self.decoder_length,
                self.hidden_size,
                device=historical_inputs.device
            )
        
        # 4. LSTM处理
        # 编码器
        lstm_input = historical_features
        encoder_output, encoder_state = self.lstm_encoder(lstm_input)
        
        # 解码器
        decoder_output, _ = self.lstm_decoder(future_features, encoder_state)
        
        # 5. 门控跳跃连接
        lstm_output = torch.cat([encoder_output, decoder_output], dim=1)
        temporal_features = self.gate_add_norm[0](lstm_output)
        
        # 6. 静态变量增强
        if self.static_variables > 0 and static_inputs is not None:
            static_enrichment = self.static_enrichment_grn(static_context)
            static_enrichment = static_enrichment.expand(-1, temporal_features.size(1), -1)
            temporal_features = temporal_features + static_enrichment
        
        # 7. 多头注意力
        # 只对解码器时间步应用注意力
        attention_input = temporal_features[:, -self.decoder_length:, :]
        attention_output, attention_weights = self.multihead_attn(attention_input)
        
        # 门控跳跃连接
        attention_output = self.gate_add_norm[1](attention_output)
        
        # 8. 位置前馈网络
        output = self.position_wise_grn(attention_output)
        output = self.gate_add_norm[2](output)
        
        # 9. 输出投影
        predictions = self.output_layer(output)
        
        return {
            'predictions': predictions,
            'attention_weights': attention_weights,
            'historical_variable_weights': hist_var_weights,
        }

    def get_attention_weights(self):
        """获取注意力权重用于可解释性分析"""
        return self.attention_weights


class QuantileLoss(nn.Module):
    """
    分位数损失函数
    用于概率预测
    """
    
    def __init__(self, quantiles: List[float] = [0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        predictions: (batch, time, num_quantiles)
        targets: (batch, time, 1)
        """
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = targets - predictions[..., i:i+1]
            losses.append(torch.max((q - 1) * errors, q * errors))
        
        loss = torch.mean(torch.sum(torch.cat(losses, dim=-1), dim=-1))
        return loss

