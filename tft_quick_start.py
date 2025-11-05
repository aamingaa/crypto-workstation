"""
TFT 快速开始示例
这是一个简化的示例，展示如何快速使用TFT模型
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path

# 导入TFT模块
from model.temporal_fusion_transformer import TemporalFusionTransformer
from model.tft_data_processor import TFTDataProcessor, prepare_gp_data_for_tft

def quick_start_example():
    """快速开始示例"""
    
    print("=" * 80)
    print("TFT 快速开始示例")
    print("=" * 80)
    
    # ==================== 第1步: 加载数据 ====================
    print("\n[1/5] 加载数据...")
    
    # 替换为你的数据路径
    DATA_PATH = './gp_models/mom_ETHUSDT_15m_1_2025-01-01_2025-03-01_2025-03-01_2025-04-01.csv.gz'
    
    if not Path(DATA_PATH).exists():
        print(f"❌ 数据文件不存在: {DATA_PATH}")
        print("请修改 DATA_PATH 为你的数据文件路径")
        return
    
    # 加载数据
    if DATA_PATH.endswith('.gz'):
        df = pd.read_csv(DATA_PATH, compression='gzip')
    else:
        df = pd.read_csv(DATA_PATH)
    
    print(f"✅ 数据加载成功: {df.shape}")
    
    # ==================== 第2步: 数据预处理 ====================
    print("\n[2/5] 数据预处理...")
    
    TARGET_COLUMN = 'label'
    
    # 准备GP数据
    df, gp_factor_columns = prepare_gp_data_for_tft(
        df,
        target_col=TARGET_COLUMN,
        gp_factor_prefix='gp_'
    )
    
    print(f"✅ 发现 {len(gp_factor_columns)} 个GP因子")
    
    # ==================== 第3步: 创建数据集 ====================
    print("\n[3/5] 创建数据集...")
    
    # 配置参数
    ENCODER_LENGTH = 60   # 历史窗口
    DECODER_LENGTH = 10   # 预测窗口
    BATCH_SIZE = 64
    
    # 创建数据处理器
    processor = TFTDataProcessor(
        target_column=TARGET_COLUMN,
        encoder_length=ENCODER_LENGTH,
        decoder_length=DECODER_LENGTH,
        batch_size=BATCH_SIZE,
        val_ratio=0.2,
    )
    
    # 准备特征
    processed_df, feature_config = processor.prepare_data_from_gp_factors(
        df,
        factor_columns=gp_factor_columns[:20],  # 只用前20个因子演示
    )
    
    # 创建数据集和数据加载器
    train_dataset, val_dataset = processor.create_datasets(
        processed_df,
        feature_config
    )
    
    train_loader, val_loader = processor.create_dataloaders(
        train_dataset,
        val_dataset,
        num_workers=0
    )
    
    print(f"✅ 训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")
    
    # ==================== 第4步: 创建模型 ====================
    print("\n[4/5] 创建模型...")
    
    HIDDEN_SIZE = 64      # 隐藏层大小（快速演示用小模型）
    LSTM_LAYERS = 1       # LSTM层数
    NUM_HEADS = 4         # 注意力头数
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    model = TemporalFusionTransformer(
        static_variables=len(feature_config['static']),
        known_regular_inputs=len(feature_config['known']),
        observed_inputs=len(feature_config['observed']),
        output_size=1,
        hidden_size=HIDDEN_SIZE,
        lstm_layers=LSTM_LAYERS,
        num_attention_heads=NUM_HEADS,
        dropout=0.2,
        encoder_length=ENCODER_LENGTH,
        decoder_length=DECODER_LENGTH,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ 模型创建成功，参数量: {total_params:,}")
    
    # ==================== 第5步: 快速测试 ====================
    print("\n[5/5] 快速测试...")
    
    # 获取一个批次数据
    sample_batch = next(iter(train_loader))
    
    historical_inputs = sample_batch['historical_inputs'].to(device)
    future_inputs = sample_batch['future_inputs'].to(device)
    static_inputs = sample_batch['static_inputs'].to(device)
    targets = sample_batch['targets'].to(device)
    
    print(f"输入形状:")
    print(f"  历史输入: {historical_inputs.shape}")
    print(f"  未来输入: {future_inputs.shape}")
    print(f"  目标值: {targets.shape}")
    
    # 前向传播测试
    model.eval()
    with torch.no_grad():
        outputs = model(
            historical_inputs=historical_inputs,
            future_inputs=future_inputs,
            static_inputs=static_inputs,
        )
        predictions = outputs['predictions']
    
    print(f"\n输出形状:")
    print(f"  预测值: {predictions.shape}")
    
    # 计算损失
    criterion = torch.nn.MSELoss()
    loss = criterion(predictions, targets)
    print(f"  初始损失: {loss.item():.6f}")
    
    print("\n" + "=" * 80)
    print("✅ 快速测试完成！")
    print("=" * 80)
    
    print("\n下一步:")
    print("1. 如果测试成功，运行完整训练: python tft_main.py")
    print("2. 修改参数以获得更好性能（参考 tft_config.yaml）")
    print("3. 查看详细文档: TFT_使用说明.md")
    
    return model, processor, train_loader, val_loader


def simple_training_example(model, train_loader, val_loader, device='cuda', epochs=5):
    """简单训练示例（仅演示，不保存模型）"""
    
    print("\n" + "=" * 80)
    print("开始简单训练（仅5个epoch演示）")
    print("=" * 80)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        for i, batch in enumerate(train_loader):
            if i >= 10:  # 只训练10个批次演示
                break
            
            historical_inputs = batch['historical_inputs'].to(device)
            future_inputs = batch['future_inputs'].to(device)
            static_inputs = batch['static_inputs'].to(device)
            targets = batch['targets'].to(device)
            
            optimizer.zero_grad()
            outputs = model(historical_inputs, future_inputs, static_inputs)
            loss = criterion(outputs['predictions'], targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= min(10, len(train_loader))
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= 5:  # 只验证5个批次
                    break
                
                historical_inputs = batch['historical_inputs'].to(device)
                future_inputs = batch['future_inputs'].to(device)
                static_inputs = batch['static_inputs'].to(device)
                targets = batch['targets'].to(device)
                
                outputs = model(historical_inputs, future_inputs, static_inputs)
                loss = criterion(outputs['predictions'], targets)
                val_loss += loss.item()
        
        val_loss /= min(5, len(val_loader))
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    print("\n✅ 训练演示完成！")
    print("要进行完整训练，请运行: python tft_main.py")


if __name__ == '__main__':
    # 运行快速开始示例
    result = quick_start_example()
    
    if result is not None:
        model, processor, train_loader, val_loader = result
        
        # 询问是否进行简单训练演示
        print("\n" + "=" * 80)
        response = input("是否运行简单训练演示（5个epoch）？[y/N]: ")
        
        if response.lower() == 'y':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            simple_training_example(model, train_loader, val_loader, device=device, epochs=5)
        else:
            print("跳过训练演示。")
            print("运行完整训练: python tft_main.py")

