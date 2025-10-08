"""
使用 Bar 级滚动统计功能运行交易管道的示例
"""
import sys
sys.path.append('..')

from pipeline.trading_pipeline import TradingPipeline


def main():
    """运行完整的交易分析管道（包含 bar 级滚动统计）"""
    
    # 初始化管道
    config = {
        'features': {
            'basic': False,
            'orderflow': False,
            'impact': False,
            'volatility': True,      # 启用波动率特征
            'momentum': False,
            'path_shape': False,
            'tail': False,
            'bucketed_flow': True,   # 启用分桶订单流特征
        },
        'labels': {
            'horizon_bars': [1, 3, 5, 10]
        },
        'model': {
            'alpha': 1.0
        }
    }
    
    pipeline = TradingPipeline(config)
    
    # 运行管道配置
    pipeline_config = {
        # 数据加载配置
        'data_config': {
            'date_range': ('2025-01-01', '2025-01-30'),
            'monthly_data_template': '/path/to/trades_{month}.{ext}',  # 替换为实际路径
            # 或者直接提供 DataFrame
            # 'trades_data': your_dataframe
        },
        
        # Bar 构建配置
        'bar_type': 'time',         # 'time' 或 'dollar'
        'time_freq': '1H',          # 时间 bar 频率
        'dollar_threshold': 60000000,  # dollar bar 阈值（如果使用 dollar bar）
        'bar_cache_template': '/path/to/bars-2025-01-01-2025-01-30-1h.{ext}',
        
        # 特征提取配置
        'feature_window_bars': 10,           # 逐笔级特征窗口（10个bar）
        'enable_rolling_stats': True,        # 🔥 启用 bar 级滚动统计
        'rolling_window_bars': 24,           # 🔥 滚动统计窗口（24个bar = 24小时）
        'enable_window_features': False,     # ⚠️ 关闭原有窗口特征（避免重复）
        
        # 模型训练配置
        'model_type': 'ridge',
        'target_horizon': 5,
        'n_splits': 5,
        'embargo_bars': 3,
        
        # 可视化
        'save_plots': False,
    }
    
    print("="*60)
    print("开始运行交易分析管道（包含 Bar 级滚动统计）")
    print("="*60)
    
    # 运行完整管道
    results = pipeline.run_full_pipeline(**pipeline_config)
    
    print("\n" + "="*60)
    print("管道运行完成!")
    print("="*60)
    
    # 输出结果摘要
    print(f"\n特征数量: {len(results['features'].columns)}")
    print(f"样本数量: {len(results['features'])}")
    print(f"\n评估指标:")
    print(f"  Pearson IC: {results['evaluation']['summary']['pearson_ic_mean']:.4f}")
    print(f"  Spearman IC: {results['evaluation']['summary']['spearman_ic_mean']:.4f}")
    print(f"  RMSE: {results['evaluation']['summary']['rmse_mean']:.6f}")
    print(f"  方向准确率: {results['evaluation']['summary']['dir_acc_mean']:.4f}")
    
    # 查看滚动统计特征
    rolling_features = [col for col in results['features'].columns if '_w24_' in col]
    print(f"\n滚动统计特征数量: {len(rolling_features)}")
    if rolling_features:
        print("示例滚动统计特征:")
        for feat in rolling_features[:10]:
            print(f"  - {feat}")
    
    return results


if __name__ == '__main__':
    results = main()

