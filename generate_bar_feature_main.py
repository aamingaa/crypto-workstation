"""
主程序入口
"""
import os
from pipeline.trading_pipeline import TradingPipeline


def main():
    """主函数"""
    # 配置参数
    config = {
        'features': {
            'basic': False,
            'orderflow': False,
            'impact': False,
            'volatility': False,
            'path_shape': False,
            'tail': True,
            'bucketed_flow': {
                'enabled': False,
                'low_q': 0.2,
                'high_q': 0.8,
                'lag': 1,
                'vpin_bins': 10,
                'min_trades_alpha': 50,
            }
        },
        'model': {
            'random_state': 42
        },
        'data': {
            'load_mode': 'daily',  # 'daily', 'monthly', 'auto'（自动选择最优方案）
            'prefer_feather': True  # 优先使用 feather 格式
        }
    }

    # config = {
    #     'bar_type': 'time',           # 使用时间条
    #     'time_interval': '1h',        # 1小时间隔
    #     'data_config': {
    #         'trades_zip_path': 'path/to/your/trades.zip'
    #     },
    #     'bar_zip_path': 'output/bars_1h.zip',
    #     'feature_window_bars': 10,
    #     'model_type': 'ridge',
    #     'target_horizon': 5,
    # }
    
    # 数据配置
    start_date = '2025-01-01'
    end_date = '2025-09-01'
    dollar_threshold = 10000 * 60000
    dollar_threshold_str = str(dollar_threshold).replace("*", "_")
    
    # 文件路径
    bar_type = 'time'
    time_interval = '1h'
    crypto = 'ETHUSDT'
    # 原始数据路径配置
    data_base_path = '/Volumes/Ext-Disk/data/futures/um'
    daily_data_template = f'{data_base_path}/daily/trades/{crypto}/{crypto}-trades-{{date}}.{{ext}}'
    monthly_data_template = f'{data_base_path}/monthly/trades/{crypto}/{crypto}-trades-{{month}}.{{ext}}'
    
    # Bars 缓存路径（放在 output 目录）
    output_base_path = '/Users/aming/project/python/crypto-trade/output'
    plot_save_dir = '/Users/aming/project/python/crypto-trade/strategy/fusion/pic'
    
    # 生成 bars 缓存文件路径
    if bar_type == 'time':
        cache_key = f'{start_date}-{end_date}-{time_interval}'
    else:
        cache_key = f'{start_date}-{end_date}-{dollar_threshold_str}'
    
    bar_cache_template = f'{output_base_path}/bars-{cache_key}.{{ext}}'
    
    # 🔥 生成 bars_with_features 保存路径
    bars_with_features_save_path = f'{data_base_path}/{crypto}/bars_with_features-{cache_key}.feather'
    
    # 创建管道
    pipeline = TradingPipeline(config)
    
    # 运行参数
    run_config = {
        'data_config': {
            'date_range': (start_date, end_date),
            'daily_data_template': daily_data_template,
            'monthly_data_template': monthly_data_template
        },
        'bar_type': bar_type,           # 使用时间条
        'time_freq': time_interval,     # 1小时间隔
        'dollar_threshold': dollar_threshold,
        'bar_cache_template': bar_cache_template,
        'bars_with_features_save_path': bars_with_features_save_path,  # 🔥 保存 bars_with_features
        'feature_window_bars': 10,
        
        # 🔥 新增：Bar 级滚动统计配置
        'enable_rolling_stats': True,      # 启用滚动统计特征
        'rolling_window_bars': 24,         # 滚动窗口：24小时
        'enable_window_features': False,   # ⚠️ 关闭原有窗口特征（避免与滚动统计重复）
        'load_mode' : 'daily',
        'model_type': 'linear',
        'target_horizon': 5,
        'n_splits': 5,
        'embargo_bars': 3,
        'save_plots': True,
        'plot_save_dir': plot_save_dir
    }
    
    try:
        # 运行完整管道
        results = pipeline.run_full_pipeline(**run_config)
        
        # 输出结果摘要
        print("\n" + "="*50)
        print("分析结果摘要")
        print("="*50)
        summary = results['evaluation']['summary']
        print(f"平均Pearson IC: {summary['pearson_ic_mean']:.4f}")
        print(f"平均Spearman IC: {summary['spearman_ic_mean']:.4f}")
        print(f"平均RMSE: {summary['rmse_mean']:.4f}")
        print(f"平均方向准确率: {summary['dir_acc_mean']:.4f}")
        print(f"平均夏普比率: {summary['sharpe_net_mean']:.4f}")
        print(f"有效折数: {summary['n_splits_effective']}")
        
        # 特征重要性前10
        if hasattr(results['model'], 'get_feature_importance'):
            importance = results['model'].get_feature_importance()
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            print(f"\n前10重要特征:")
            for i, (feature, score) in enumerate(top_features, 1):
                print(f"{i:2d}. {feature}: {score:.6f}")
        
        print(f"\n数据统计:")
        print(f"样本数量: {len(results['features'])}")
        print(f"特征数量: {len(results['features'].columns)}")
        print(f"Bar数量: {len(results['bars'])}")
        
        # 查看滚动统计特征
        rolling_features = [col for col in results['features'].columns if '_w24_' in col or '_w' in col and '_mean' in col]
        if rolling_features:
            print(f"\n滚动统计特征数量: {len(rolling_features)}")
            print(f"示例滚动统计特征（前10个）:")
            for i, feat in enumerate(rolling_features[:10], 1):
                print(f"  {i:2d}. {feat}")
        
        return results
        
    except Exception as e:
        print(f"运行过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    main()
