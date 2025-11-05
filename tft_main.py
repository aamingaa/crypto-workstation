"""
Temporal Fusion Transformer 训练主程序

功能:
1. 从CSV加载GP因子数据
2. 数据预处理和特征工程
3. TFT模型训练
4. 模型评估和可视化
5. 模型保存和加载
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置路径
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import yaml
import joblib
from datetime import datetime
import os
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

# 导入TFT相关模块
from model.temporal_fusion_transformer import TemporalFusionTransformer, QuantileLoss
from model.tft_data_processor import (
    TFTDataProcessor,
    prepare_gp_data_for_tft,
    TimeSeriesScaler
)

# 导入GP数据加载
try:
    from gp_crypto_next import dataload
except ImportError:
    import gp_crypto_next.dataload as dataload


class TFTTrainer:
    """TFT训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device: str = 'cuda',
        learning_rate: float = 1e-3,
        max_epochs: int = 100,
        patience: int = 10,
        save_dir: str = './tft_models',
    ):
        """
        参数:
            model: TFT模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            device: 设备 ('cuda' or 'cpu')
            learning_rate: 学习率
            max_epochs: 最大训练轮数
            patience: 早停耐心值
            save_dir: 模型保存目录
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # 学习率调度器
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
        }
        
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
    def train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            # 数据移到设备
            historical_inputs = batch['historical_inputs'].to(self.device)
            future_inputs = batch['future_inputs'].to(self.device)
            static_inputs = batch['static_inputs'].to(self.device) if 'static_inputs' in batch else None
            targets = batch['targets'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(
                historical_inputs=historical_inputs,
                future_inputs=future_inputs,
                static_inputs=static_inputs,
            )
            
            predictions = outputs['predictions']
            
            # 计算损失
            loss = self.criterion(predictions, targets)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / num_batches
    
    def validate(self) -> Tuple[float, Dict[str, float]]:
        """验证模型"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                historical_inputs = batch['historical_inputs'].to(self.device)
                future_inputs = batch['future_inputs'].to(self.device)
                static_inputs = batch['static_inputs'].to(self.device) if 'static_inputs' in batch else None
                targets = batch['targets'].to(self.device)
                
                outputs = self.model(
                    historical_inputs=historical_inputs,
                    future_inputs=future_inputs,
                    static_inputs=static_inputs,
                )
                
                predictions = outputs['predictions']
                loss = self.criterion(predictions, targets)
                
                total_loss += loss.item()
                num_batches += 1
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        avg_loss = total_loss / num_batches
        
        # 计算额外指标
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        metrics = self._calculate_metrics(all_predictions, all_targets)
        metrics['loss'] = avg_loss
        
        return avg_loss, metrics
    
    def _calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """计算评估指标"""
        # MSE
        mse = np.mean((predictions - targets) ** 2)
        
        # MAE
        mae = np.mean(np.abs(predictions - targets))
        
        # RMSE
        rmse = np.sqrt(mse)
        
        # MAPE
        mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
        
        # R2
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2,
        }
    
    def train(self):
        """完整训练流程"""
        print("=" * 80)
        print("开始训练 Temporal Fusion Transformer")
        print("=" * 80)
        
        for epoch in range(self.max_epochs):
            print(f"\nEpoch {epoch + 1}/{self.max_epochs}")
            print("-" * 80)
            
            # 训练
            train_loss = self.train_epoch()
            
            # 验证
            val_loss, val_metrics = self.validate()
            
            # 更新学习率
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rates'].append(current_lr)
            
            # 打印结果
            print(f"\n训练损失: {train_loss:.6f}")
            print(f"验证损失: {val_loss:.6f}")
            print(f"学习率: {current_lr:.2e}")
            print("\n验证指标:")
            for key, value in val_metrics.items():
                print(f"  {key}: {value:.6f}")
            
            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self.save_checkpoint('best_model.pth', epoch, val_metrics)
                print(f"\n✅ 保存最佳模型 (验证损失: {val_loss:.6f})")
            else:
                self.epochs_without_improvement += 1
            
            # 早停
            if self.epochs_without_improvement >= self.patience:
                print(f"\n早停: {self.patience} 个epoch无改进")
                break
        
        print("\n" + "=" * 80)
        print("训练完成!")
        print("=" * 80)
        
        return self.history
    
    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'metrics': metrics,
            'history': self.history,
        }
        
        save_path = self.save_dir / filename
        torch.save(checkpoint, save_path)
    
    def load_checkpoint(self, filename: str):
        """加载检查点"""
        load_path = self.save_dir / filename
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        print(f"加载检查点: {filename}")
        print(f"Epoch: {checkpoint['epoch']}")
        print(f"最佳验证损失: {self.best_val_loss:.6f}")


def plot_training_history(history: Dict, save_path: Optional[str] = None):
    """绘制训练历史"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 损失曲线
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 学习率曲线
    axes[1].plot(history['learning_rates'], color='green', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rate Schedule')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练历史图已保存至: {save_path}")
    
    plt.show()


def plot_predictions(
    model: nn.Module,
    val_loader,
    scaler: TimeSeriesScaler,
    target_column: str,
    device: str = 'cuda',
    num_samples: int = 100,
    save_path: Optional[str] = None
):
    """绘制预测结果"""
    model.eval()
    
    predictions_list = []
    targets_list = []
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_samples // batch['targets'].size(0):
                break
                
            historical_inputs = batch['historical_inputs'].to(device)
            future_inputs = batch['future_inputs'].to(device)
            static_inputs = batch['static_inputs'].to(device) if 'static_inputs' in batch else None
            targets = batch['targets'].to(device)
            
            outputs = model(
                historical_inputs=historical_inputs,
                future_inputs=future_inputs,
                static_inputs=static_inputs,
            )
            
            predictions = outputs['predictions'].cpu().numpy()
            targets = targets.cpu().numpy()
            
            predictions_list.append(predictions)
            targets_list.append(targets)
    
    predictions = np.concatenate(predictions_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    
    # 只取第一个预测步（或平均所有步）
    predictions_flat = predictions[:, 0, 0]  # (num_samples,)
    targets_flat = targets[:, 0, 0]
    
    # 反标准化
    predictions_unscaled = scaler.inverse_transform(predictions_flat, target_column)
    targets_unscaled = scaler.inverse_transform(targets_flat, target_column)
    
    # 绘图
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 时序图
    axes[0].plot(targets_unscaled[:200], label='True', alpha=0.7, linewidth=2)
    axes[0].plot(predictions_unscaled[:200], label='Predicted', alpha=0.7, linewidth=2)
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Value')
    axes[0].set_title('Predictions vs True Values')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 散点图
    axes[1].scatter(targets_unscaled, predictions_unscaled, alpha=0.5, s=20)
    axes[1].plot([targets_unscaled.min(), targets_unscaled.max()],
                 [targets_unscaled.min(), targets_unscaled.max()],
                 'r--', linewidth=2, label='Perfect Prediction')
    axes[1].set_xlabel('True Values')
    axes[1].set_ylabel('Predicted Values')
    axes[1].set_title('Prediction Scatter Plot')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"预测结果图已保存至: {save_path}")
    
    plt.show()


def main():
    """主函数"""
    # ==================== 配置参数 ====================
    
    # 数据配置
    DATA_PATH = './gp_models/mom_ETHUSDT_15m_1_2025-01-01_2025-03-01_2025-03-01_2025-04-01.csv.gz'
    TARGET_COLUMN = 'label'  # 目标列名
    
    # 模型配置
    ENCODER_LENGTH = 60      # 历史窗口长度（4小时 = 60 * 15分钟）
    DECODER_LENGTH = 10       # 预测窗口长度（2.5小时）
    HIDDEN_SIZE = 128         # 隐藏层大小
    LSTM_LAYERS = 2           # LSTM层数
    NUM_ATTENTION_HEADS = 4   # 注意力头数
    DROPOUT = 0.2             # Dropout率
    
    # 训练配置
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    MAX_EPOCHS = 100
    PATIENCE = 15
    VAL_RATIO = 0.2
    
    # 设备配置
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {DEVICE}")
    
    # 保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    SAVE_DIR = f'./tft_results_{timestamp}'
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    
    # ==================== 加载数据 ====================
    print("\n" + "=" * 80)
    print("1. 加载数据")
    print("=" * 80)
    
    if DATA_PATH.endswith('.gz'):
        df = pd.read_csv(DATA_PATH, compression='gzip')
    else:
        df = pd.read_csv(DATA_PATH)
    
    print(f"数据形状: {df.shape}")
    print(f"数据列: {df.columns.tolist()}")
    
    # ==================== 数据预处理 ====================
    print("\n" + "=" * 80)
    print("2. 数据预处理")
    print("=" * 80)
    
    # 准备GP数据
    df, gp_factor_columns = prepare_gp_data_for_tft(
        df,
        target_col=TARGET_COLUMN,
        gp_factor_prefix='gp_'
    )
    
    print(f"GP因子数量: {len(gp_factor_columns)}")
    print(f"样本数量: {len(df)}")
    
    # 创建数据处理器
    data_processor = TFTDataProcessor(
        target_column=TARGET_COLUMN,
        encoder_length=ENCODER_LENGTH,
        decoder_length=DECODER_LENGTH,
        batch_size=BATCH_SIZE,
        stride_train=1,
        stride_val=5,
        val_ratio=VAL_RATIO,
    )
    
    # 准备特征
    processed_df, feature_config = data_processor.prepare_data_from_gp_factors(
        df,
        factor_columns=gp_factor_columns,
        time_features=None,  # 自动生成时间特征
        static_features=None,
    )
    
    print(f"\n特征配置:")
    print(f"  观测特征: {len(feature_config['observed'])}")
    print(f"  已知特征: {len(feature_config['known'])}")
    print(f"  静态特征: {len(feature_config['static'])}")
    
    # 创建数据集
    train_dataset, val_dataset = data_processor.create_datasets(
        processed_df,
        feature_config
    )
    
    print(f"\n训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(val_dataset)}")
    
    # 创建数据加载器
    train_loader, val_loader = data_processor.create_dataloaders(
        train_dataset,
        val_dataset,
        num_workers=0  # Windows系统设为0，Linux/Mac可设为4
    )
    
    # 打印样本信息
    sample_batch = data_processor.get_sample_batch(train_loader)
    print("\n样本批次形状:")
    for key, value in sample_batch.items():
        print(f"  {key}: {value.shape}")
    
    # ==================== 创建模型 ====================
    print("\n" + "=" * 80)
    print("3. 创建模型")
    print("=" * 80)
    
    model = TemporalFusionTransformer(
        static_variables=len(feature_config['static']),
        known_regular_inputs=len(feature_config['known']),
        known_categorical_inputs=0,
        observed_inputs=len(feature_config['observed']),
        output_size=1,
        hidden_size=HIDDEN_SIZE,
        lstm_layers=LSTM_LAYERS,
        num_attention_heads=NUM_ATTENTION_HEADS,
        dropout=DROPOUT,
        encoder_length=ENCODER_LENGTH,
        decoder_length=DECODER_LENGTH,
    )
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型参数:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    # ==================== 训练模型 ====================
    print("\n" + "=" * 80)
    print("4. 训练模型")
    print("=" * 80)
    
    trainer = TFTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        learning_rate=LEARNING_RATE,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        save_dir=os.path.join(SAVE_DIR, 'checkpoints'),
    )
    
    history = trainer.train()
    
    # ==================== 可视化结果 ====================
    print("\n" + "=" * 80)
    print("5. 可视化结果")
    print("=" * 80)
    
    # 加载最佳模型
    trainer.load_checkpoint('best_model.pth')
    
    # 绘制训练历史
    plot_training_history(
        history,
        save_path=os.path.join(SAVE_DIR, 'training_history.png')
    )
    
    # 绘制预测结果
    plot_predictions(
        model=trainer.model,
        val_loader=val_loader,
        scaler=data_processor.scaler,
        target_column=TARGET_COLUMN,
        device=DEVICE,
        num_samples=500,
        save_path=os.path.join(SAVE_DIR, 'predictions.png')
    )
    
    # ==================== 保存配置和模型 ====================
    print("\n" + "=" * 80)
    print("6. 保存配置和模型")
    print("=" * 80)
    
    # 保存配置
    config = {
        'model_config': {
            'encoder_length': ENCODER_LENGTH,
            'decoder_length': DECODER_LENGTH,
            'hidden_size': HIDDEN_SIZE,
            'lstm_layers': LSTM_LAYERS,
            'num_attention_heads': NUM_ATTENTION_HEADS,
            'dropout': DROPOUT,
        },
        'data_config': {
            'target_column': TARGET_COLUMN,
            'gp_factors': gp_factor_columns,
            'feature_config': feature_config,
        },
        'training_config': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'max_epochs': MAX_EPOCHS,
            'patience': PATIENCE,
        }
    }
    
    with open(os.path.join(SAVE_DIR, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # 保存scaler
    joblib.dump(data_processor.scaler, os.path.join(SAVE_DIR, 'scaler.pkl'))
    
    print(f"\n✅ 所有结果已保存至: {SAVE_DIR}")
    print("\n" + "=" * 80)
    print("完成!")
    print("=" * 80)


if __name__ == '__main__':
    main()

