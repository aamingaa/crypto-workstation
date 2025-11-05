"""
TFT 数据预处理模块

功能:
1. 将时间序列数据转换为TFT输入格式
2. 处理GP因子特征
3. 创建滑动窗口
4. 数据标准化
5. 批量生成
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesScaler:
    """时间序列特征缩放器"""
    
    def __init__(self, method: str = 'standard'):
        """
        参数:
            method: 'standard', 'robust', 'minmax', 'none'
        """
        self.method = method
        self.scalers = {}
        
    def fit(self, data: pd.DataFrame, columns: List[str]):
        """拟合缩放器"""
        for col in columns:
            if self.method == 'standard':
                scaler = StandardScaler()
            elif self.method == 'robust':
                scaler = RobustScaler()
            elif self.method == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler(feature_range=(-1, 1))
            else:
                continue
                
            scaler.fit(data[[col]].values)
            self.scalers[col] = scaler
        
        return self
    
    def transform(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """转换数据"""
        transformed_data = data.copy()
        for col in columns:
            if col in self.scalers:
                transformed_data[col] = self.scalers[col].transform(data[[col]].values)
        return transformed_data
    
    def inverse_transform(self, data: Union[pd.DataFrame, np.ndarray], column: str) -> np.ndarray:
        """反转换"""
        if column in self.scalers:
            if isinstance(data, pd.DataFrame):
                return self.scalers[column].inverse_transform(data[[column]].values)
            else:
                return self.scalers[column].inverse_transform(data.reshape(-1, 1)).flatten()
        return data


class TFTDataset(Dataset):
    """
    TFT 数据集类
    
    将原始数据转换为TFT模型所需的格式:
    - 历史观测输入 (encoder_length)
    - 未来已知输入 (decoder_length)
    - 静态变量
    - 目标值
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        target_column: str,
        observed_columns: List[str],
        known_columns: Optional[List[str]] = None,
        static_columns: Optional[List[str]] = None,
        encoder_length: int = 30,
        decoder_length: int = 5,
        stride: int = 1,
        scale_target: bool = True,
        scaler: Optional[TimeSeriesScaler] = None,
    ):
        """
        参数:
            data: DataFrame containing all features
            target_column: 目标变量列名
            observed_columns: 只能在历史观测到的变量列名
            known_columns: 未来已知的变量列名（如时间特征、技术指标等）
            static_columns: 静态变量列名（不随时间变化）
            encoder_length: 编码器长度（历史窗口）
            decoder_length: 解码器长度（预测窗口）
            stride: 滑动窗口步长
            scale_target: 是否缩放目标变量
            scaler: 预训练的缩放器（用于测试集）
        """
        self.data = data.copy()
        self.target_column = target_column
        self.observed_columns = observed_columns if observed_columns else []
        self.known_columns = known_columns if known_columns else []
        self.static_columns = static_columns if static_columns else []
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.stride = stride
        self.scale_target = scale_target
        
        # 确保目标列在观测列中
        if self.target_column not in self.observed_columns:
            self.observed_columns.append(self.target_column)
        
        # 数据缩放
        if scaler is None:
            self.scaler = TimeSeriesScaler(method='robust')
            all_columns = self.observed_columns + self.known_columns
            self.scaler.fit(self.data, all_columns)
        else:
            self.scaler = scaler
        
        # 缩放数据
        all_columns = self.observed_columns + self.known_columns
        self.data = self.scaler.transform(self.data, all_columns)
        
        # 生成索引
        self.valid_indices = self._generate_indices()
        
    def _generate_indices(self) -> List[int]:
        """生成有效的样本索引"""
        total_length = self.encoder_length + self.decoder_length
        max_idx = len(self.data) - total_length
        
        indices = list(range(0, max_idx, self.stride))
        return indices
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        返回一个样本
        
        返回格式:
        {
            'historical_inputs': (encoder_length, num_observed + num_known),
            'future_inputs': (decoder_length, num_known),
            'static_inputs': (num_static,),
            'targets': (decoder_length, 1),
            'encoder_target': (encoder_length, 1)  # 用于teacher forcing
        }
        """
        start_idx = self.valid_indices[idx]
        encoder_end = start_idx + self.encoder_length
        decoder_end = encoder_end + self.decoder_length
        
        # 1. 历史观测输入 (encoder部分)
        historical_observed = self.data[self.observed_columns].iloc[start_idx:encoder_end].values
        
        # 2. 历史已知输入 (encoder部分)
        if self.known_columns:
            historical_known = self.data[self.known_columns].iloc[start_idx:encoder_end].values
            historical_inputs = np.concatenate([historical_observed, historical_known], axis=1)
        else:
            historical_inputs = historical_observed
        
        # 3. 未来已知输入 (decoder部分)
        if self.known_columns:
            future_inputs = self.data[self.known_columns].iloc[encoder_end:decoder_end].values
        else:
            future_inputs = np.zeros((self.decoder_length, 1))
        
        # 4. 静态输入
        if self.static_columns:
            static_inputs = self.data[self.static_columns].iloc[start_idx].values
        else:
            static_inputs = np.zeros(1)
        
        # 5. 目标值 (decoder部分)
        targets = self.data[self.target_column].iloc[encoder_end:decoder_end].values.reshape(-1, 1)
        
        # 6. 编码器目标 (用于某些训练策略)
        encoder_target = self.data[self.target_column].iloc[start_idx:encoder_end].values.reshape(-1, 1)
        
        # 转换为tensor
        return {
            'historical_inputs': torch.FloatTensor(historical_inputs),
            'future_inputs': torch.FloatTensor(future_inputs),
            'static_inputs': torch.FloatTensor(static_inputs),
            'targets': torch.FloatTensor(targets),
            'encoder_target': torch.FloatTensor(encoder_target),
        }


class TFTDataProcessor:
    """
    TFT数据处理器
    
    整合数据预处理、数据集创建、DataLoader生成等功能
    """
    
    def __init__(
        self,
        target_column: str = 'target',
        encoder_length: int = 30,
        decoder_length: int = 5,
        batch_size: int = 64,
        stride_train: int = 1,
        stride_val: int = 5,
        val_ratio: float = 0.2,
    ):
        """
        参数:
            target_column: 目标变量名
            encoder_length: 历史窗口长度
            decoder_length: 预测窗口长度
            batch_size: 批大小
            stride_train: 训练集滑动窗口步长
            stride_val: 验证集滑动窗口步长
            val_ratio: 验证集比例
        """
        self.target_column = target_column
        self.encoder_length = encoder_length
        self.decoder_length = decoder_length
        self.batch_size = batch_size
        self.stride_train = stride_train
        self.stride_val = stride_val
        self.val_ratio = val_ratio
        
        self.scaler = None
        self.feature_names = None
        
    def prepare_data_from_gp_factors(
        self,
        df: pd.DataFrame,
        factor_columns: List[str],
        time_features: Optional[List[str]] = None,
        static_features: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """
        从GP因子DataFrame准备数据
        
        参数:
            df: 包含所有特征的DataFrame
            factor_columns: GP生成的因子列名列表
            time_features: 时间特征列名（如hour, dayofweek等）
            static_features: 静态特征列名
            
        返回:
            processed_df: 处理后的DataFrame
            feature_config: 特征配置字典
        """
        processed_df = df.copy()
        
        # 确保有时间索引
        if not isinstance(processed_df.index, pd.DatetimeIndex):
            if 'timestamp' in processed_df.columns:
                processed_df.index = pd.to_datetime(processed_df['timestamp'])
            elif 'time' in processed_df.columns:
                processed_df.index = pd.to_datetime(processed_df['time'])
        
        # 添加时间特征
        if time_features is None:
            time_features = []
            processed_df['hour'] = processed_df.index.hour
            processed_df['dayofweek'] = processed_df.index.dayofweek
            processed_df['day'] = processed_df.index.day
            processed_df['month'] = processed_df.index.month
            time_features = ['hour', 'dayofweek', 'day', 'month']
        
        # 确保目标列存在
        if self.target_column not in processed_df.columns:
            raise ValueError(f"目标列 '{self.target_column}' 不在DataFrame中")
        
        # 处理缺失值
        processed_df = processed_df.fillna(method='ffill').fillna(method='bfill')
        
        # 特征配置
        feature_config = {
            'observed': factor_columns,  # GP因子作为观测输入
            'known': time_features,       # 时间特征作为已知输入
            'static': static_features if static_features else [],
        }
        
        # 保存特征名称
        self.feature_names = feature_config
        
        return processed_df, feature_config
    
    def create_datasets(
        self,
        df: pd.DataFrame,
        feature_config: Dict[str, List[str]],
        train_end_idx: Optional[int] = None,
    ) -> Tuple[TFTDataset, TFTDataset]:
        """
        创建训练集和验证集
        
        参数:
            df: 处理后的DataFrame
            feature_config: 特征配置
            train_end_idx: 训练集结束索引（如果为None，自动按比例分割）
            
        返回:
            train_dataset, val_dataset
        """
        if train_end_idx is None:
            train_end_idx = int(len(df) * (1 - self.val_ratio))
        
        train_df = df.iloc[:train_end_idx]
        val_df = df.iloc[train_end_idx:]
        
        print(f"训练集大小: {len(train_df)}, 验证集大小: {len(val_df)}")
        
        # 创建训练集
        train_dataset = TFTDataset(
            data=train_df,
            target_column=self.target_column,
            observed_columns=feature_config['observed'],
            known_columns=feature_config['known'],
            static_columns=feature_config['static'],
            encoder_length=self.encoder_length,
            decoder_length=self.decoder_length,
            stride=self.stride_train,
        )
        
        # 创建验证集（使用训练集的scaler）
        val_dataset = TFTDataset(
            data=val_df,
            target_column=self.target_column,
            observed_columns=feature_config['observed'],
            known_columns=feature_config['known'],
            static_columns=feature_config['static'],
            encoder_length=self.encoder_length,
            decoder_length=self.decoder_length,
            stride=self.stride_val,
            scaler=train_dataset.scaler,
        )
        
        self.scaler = train_dataset.scaler
        
        return train_dataset, val_dataset
    
    def create_dataloaders(
        self,
        train_dataset: TFTDataset,
        val_dataset: TFTDataset,
        num_workers: int = 4,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        创建DataLoader
        
        参数:
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            num_workers: 数据加载进程数
            
        返回:
            train_loader, val_loader
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        
        return train_loader, val_loader
    
    def get_sample_batch(self, dataloader: DataLoader) -> Dict[str, torch.Tensor]:
        """获取一个样本批次用于调试"""
        return next(iter(dataloader))


def prepare_gp_data_for_tft(
    df: pd.DataFrame,
    target_col: str = 'label',
    gp_factor_prefix: str = 'gp_',
) -> Tuple[pd.DataFrame, List[str]]:
    """
    准备GP生成的数据用于TFT
    
    参数:
        df: 包含GP因子的DataFrame
        target_col: 目标列名
        gp_factor_prefix: GP因子列名前缀
        
    返回:
        processed_df: 处理后的DataFrame
        gp_factor_columns: GP因子列名列表
    """
    # 识别GP因子列
    gp_factor_columns = [col for col in df.columns if col.startswith(gp_factor_prefix)]
    
    if len(gp_factor_columns) == 0:
        # 如果没有特定前缀，选择所有数值列（除了目标列）
        gp_factor_columns = [col for col in df.select_dtypes(include=[np.number]).columns 
                             if col != target_col]
    
    print(f"发现 {len(gp_factor_columns)} 个GP因子列")
    
    # 处理无穷值和极端值
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 对GP因子进行winsorization（去除极端值）
    for col in gp_factor_columns:
        q01 = df[col].quantile(0.01)
        q99 = df[col].quantile(0.99)
        df[col] = df[col].clip(q01, q99)
    
    # 填充缺失值
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    return df, gp_factor_columns

