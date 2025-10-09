#!/usr/bin/env python3
"""
滚动统计数据准备方法使用示例

这个示例展示了如何使用新的 data_prepare_rolling 方法：
1. 直接使用已聚合的bar级数据（包含OHLCV和微观结构因子）
2. 使用RollingAggregator进行滚动统计特征提取
3. 与现有系统完全兼容
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gp_crypto_next.dataload import data_prepare_rolling
import pandas as pd
import numpy as np

def example_usage():
    """使用示例"""
    
    # 示例参数
    sym = "BTCUSDT"
    freq = "1h"
    start_date_train = "2024-01-01"
    end_date_train = "2024-06-30"
    start_date_test = "2024-07-01"
    end_date_test = "2024-08-31"
    
    # 假设您有一个包含微观结构因子的bar数据文件
    # 文件应该包含：open, high, low, close, volume, 以及各种微观结构因子
    file_path = "/path/to/your/bars_with_microstructure_features.feather"
    
    # 滚动窗口设置
    rolling_windows = [5, 10, 20]  # 5bar, 10bar, 20bar窗口
    
    print("=" * 60)
    print("滚动统计数据准备示例")
    print("=" * 60)
    
    try:
        # 调用新的数据准备方法
        (X_all, X_train, y_train, ret_train, X_test, y_test, ret_test,
         feature_names, open_train, open_test, close_train, close_test,
         z_index, ohlc) = data_prepare_rolling(
            sym=sym,
            freq=freq,
            start_date_train=start_date_train,
            end_date_train=end_date_train,
            start_date_test=start_date_test,
            end_date_test=end_date_test,
            file_path=file_path,  # 已聚合的bar数据文件
            rolling_windows=rolling_windows,  # 滚动窗口列表
            use_rolling_aggregator=True,  # 使用滚动统计
            rolling_w=2000,  # 标签标准化窗口
            feature_types=None,  # 特征类型筛选（None表示使用所有特征）
            feature_keywords=None  # 关键词筛选（None表示使用所有特征）
        )
        
        print(f"\n✓ 数据准备完成！")
        print(f"训练集: {len(X_train)} 样本, {len(feature_names)} 特征")
        print(f"测试集: {len(X_test)} 样本, {len(feature_names)} 特征")
        print(f"特征数量: {len(feature_names)}")
        
        # 显示一些特征名称示例
        print(f"\n特征名称示例（前10个）:")
        for i, name in enumerate(feature_names[:10]):
            print(f"  {i+1:2d}. {name}")
        if len(feature_names) > 10:
            print(f"  ... 还有 {len(feature_names)-10} 个特征")
        
        # 显示数据统计
        print(f"\n数据统计:")
        print(f"训练集标签统计: mean={y_train.mean():.4f}, std={y_train.std():.4f}")
        print(f"测试集标签统计: mean={y_test.mean():.4f}, std={y_test.std():.4f}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_names
        }
        
    except FileNotFoundError:
        print(f"⚠️  文件未找到: {file_path}")
        print("请确保文件路径正确，或使用示例数据")
        
        # 使用示例数据（不指定file_path，会回退到传统K线数据）
        print("\n使用示例数据（回退模式）...")
        (X_all, X_train, y_train, ret_train, X_test, y_test, ret_test,
         feature_names, open_train, open_test, close_train, close_test,
         z_index, ohlc) = data_prepare_rolling(
            sym=sym,
            freq=freq,
            start_date_train=start_date_train,
            end_date_train=end_date_train,
            start_date_test=start_date_test,
            end_date_test=end_date_test,
            # file_path=None,  # 不指定文件，使用回退模式
            rolling_windows=rolling_windows,
            use_rolling_aggregator=True,
            rolling_w=2000,
            feature_types=None,  # 特征类型筛选（None表示使用所有特征）
            feature_keywords=None  # 关键词筛选（None表示使用所有特征）
        )
        
        print(f"✓ 回退模式数据准备完成！")
        print(f"训练集: {len(X_train)} 样本, {len(feature_names)} 特征")
        print(f"测试集: {len(X_test)} 样本, {len(feature_names)} 特征")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_names
        }

def create_sample_data():
    """创建示例数据文件（用于测试）"""
    
    # 生成示例bar数据
    dates = pd.date_range('2024-01-01', '2024-08-31', freq='1H')
    n_bars = len(dates)
    
    # 生成价格数据
    np.random.seed(42)
    price = 50000 + np.cumsum(np.random.randn(n_bars) * 100)
    
    # 生成OHLCV数据
    bars_data = {
        'timestamp': dates,
        'open': price + np.random.randn(n_bars) * 10,
        'high': price + np.abs(np.random.randn(n_bars)) * 20,
        'low': price - np.abs(np.random.randn(n_bars)) * 20,
        'close': price,
        'volume': np.random.exponential(1000, n_bars),
    }
    
    # 添加一些示例微观结构因子
    bars_data.update({
        'rv': np.random.exponential(0.001, n_bars),  # 已实现波动率
        'bpv': np.random.exponential(0.0008, n_bars),  # 双幂变差
        'jump': np.maximum(bars_data['rv'] - bars_data['bpv'], 0),  # 跳跃
        'vpin': np.random.beta(2, 5, n_bars),  # VPIN
        'small_signed_dollar': np.random.randn(n_bars) * 1000,  # 小单净流入
        'large_signed_dollar': np.random.randn(n_bars) * 5000,  # 大单净流入
        'hl_ratio': np.random.exponential(0.01, n_bars),  # 高低价比
        'signed_volume': np.random.randn(n_bars) * 500,  # 净成交量
        'amihud': np.random.exponential(0.0001, n_bars),  # Amihud非流动性
        'kyle_lambda': np.random.exponential(0.01, n_bars),  # Kyle Lambda
    })
    
    bars_df = pd.DataFrame(bars_data)
    
    # 保存为feather格式
    output_path = "/tmp/sample_bars_with_microstructure.feather"
    bars_df.to_feather(output_path)
    
    print(f"✓ 示例数据已保存到: {output_path}")
    print(f"数据形状: {bars_df.shape}")
    print(f"数据列: {list(bars_df.columns)}")
    
    return output_path

if __name__ == "__main__":
    print("滚动统计数据准备方法使用示例")
    print("=" * 60)
    
    # 创建示例数据
    sample_file = create_sample_data()
    
    # 使用示例数据
    print(f"\n使用示例数据: {sample_file}")
    
    # 修改示例中的文件路径
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(suffix='.feather', delete=False)
    temp_file.close()
    
    # 这里可以替换为您的实际数据文件路径
    # file_path = "/path/to/your/actual/bars_with_microstructure.feather"
    
    # 运行示例
    result = example_usage()
    
    print("\n" + "=" * 60)
    print("示例运行完成！")
    print("=" * 60)
