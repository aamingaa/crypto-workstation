#!/usr/bin/env python3
"""
Triple Barrier 使用示例
演示如何在量化策略中使用 Triple Barrier 标注方法
"""

from pathlib import Path
import sys

# 设置路径
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from multi_model_main import QuantTradingStrategy
import matplotlib.pyplot as plt

def example1_basic_usage():
    """
    示例1：基本使用 - 生成 Triple Barrier 标签并分析
    """
    print("\n" + "="*80)
    print("示例1：生成 Triple Barrier 标签")
    print("="*80)
    
    yaml_path = 'gp_crypto_next/coarse_grain_parameters.yaml'
    factor_csv_path = 'gp_models/ETHUSDT_15m_1_2025-01-01_2025-01-20_2025-01-20_2025-01-31.csv.gz'
    
    strategy = QuantTradingStrategy.from_yaml(
        yaml_path=yaml_path,
        factor_csv_path=factor_csv_path
    )
    
    # 加载数据
    strategy.load_data_from_dataload()
    
    # 生成 Triple Barrier 标签
    strategy.generate_triple_barrier_labels(
        pt_sl=[2.0, 2.0],      # 止盈止损各2倍波动率
        max_holding=[0, 4]     # 最多持有4小时
    )
    
    # 分析结果
    print("\n📊 Triple Barrier 标签统计:")
    print(f"  总交易次数: {len(strategy.barrier_results)}")
    print(f"  盈利交易: {(strategy.meta_labels == 1).sum()} ({(strategy.meta_labels == 1).sum()/len(strategy.meta_labels):.2%})")
    print(f"  亏损交易: {(strategy.meta_labels == 0).sum()} ({(strategy.meta_labels == 0).sum()/len(strategy.meta_labels):.2%})")
    print(f"  平均收益: {strategy.barrier_results['ret'].mean():.4f}")
    print(f"  收益标准差: {strategy.barrier_results['ret'].std():.4f}")
    print(f"  最大单次收益: {strategy.barrier_results['ret'].max():.4f}")
    print(f"  最大单次亏损: {strategy.barrier_results['ret'].min():.4f}")
    
    return strategy


def example2_full_pipeline():
    """
    示例2：完整流程 - 使用 Triple Barrier 训练模型
    """
    print("\n" + "="*80)
    print("示例2：使用 Triple Barrier 收益训练模型")
    print("="*80)
    
    yaml_path = 'gp_crypto_next/coarse_grain_parameters.yaml'
    factor_csv_path = 'gp_models/ETHUSDT_15m_1_2025-01-01_2025-01-20_2025-01-20_2025-01-31.csv.gz'
    
    strategy_config = {
        'corr_threshold': 0.5,
        'max_factors': 30,
        'fees_rate': 0.0005,
        'model_save_path': './models_triple_barrier',
    }
    
    strategy = (
        QuantTradingStrategy.from_yaml(
            yaml_path=yaml_path,
            factor_csv_path=factor_csv_path,
            strategy_config=strategy_config
        )
        .load_data_from_dataload()
        .load_factor_expressions()
        .evaluate_factor_expressions()
        .normalize_factors()
        .select_factors()
        
        # 添加 Triple Barrier
        .generate_triple_barrier_labels(
            pt_sl=[2.5, 2.0],      # 止盈2.5倍，止损2倍
            max_holding=[0, 6]     # 最多持有6小时
        )
        .use_triple_barrier_as_y()  # 使用 Triple Barrier 收益替代固定周期收益
        
        # 继续训练
        .prepare_training_data()
        .train_models()
        .make_predictions(weight_method='equal')
        .backtest_all_models()
    )
    
    # 显示结果
    print("\n📈 模型绩效汇总:")
    strategy.get_performance_summary()
    
    # 绘制结果
    strategy.plot_results('Ensemble')
    
    return strategy


def example3_parameter_comparison():
    """
    示例3：参数对比 - 比较不同 Triple Barrier 参数的效果
    """
    print("\n" + "="*80)
    print("示例3：对比不同 Triple Barrier 参数")
    print("="*80)
    
    yaml_path = 'gp_crypto_next/coarse_grain_parameters.yaml'
    factor_csv_path = 'gp_models/ETHUSDT_15m_1_2025-01-01_2025-01-20_2025-01-20_2025-01-31.csv.gz'
    
    # 测试不同参数组合
    param_configs = [
        {'pt_sl': [1.5, 1.5], 'max_holding': [0, 4], 'name': '保守(1.5x, 4h)'},
        {'pt_sl': [2.0, 2.0], 'max_holding': [0, 4], 'name': '平衡(2.0x, 4h)'},
        {'pt_sl': [2.5, 2.0], 'max_holding': [0, 6], 'name': '激进(2.5x, 6h)'},
    ]
    
    results = []
    
    for config in param_configs:
        print(f"\n测试配置: {config['name']}")
        print(f"  pt_sl={config['pt_sl']}, max_holding={config['max_holding']}")
        
        strategy = QuantTradingStrategy.from_yaml(
            yaml_path=yaml_path,
            factor_csv_path=factor_csv_path
        )
        
        strategy.load_data_from_dataload()
        strategy.generate_triple_barrier_labels(
            pt_sl=config['pt_sl'],
            max_holding=config['max_holding']
        )
        
        # 收集统计信息
        win_rate = (strategy.meta_labels == 1).sum() / len(strategy.meta_labels)
        avg_return = strategy.barrier_results['ret'].mean()
        std_return = strategy.barrier_results['ret'].std()
        sharpe = avg_return / std_return if std_return > 0 else 0
        
        results.append({
            'name': config['name'],
            'win_rate': win_rate,
            'avg_return': avg_return,
            'std_return': std_return,
            'sharpe': sharpe
        })
        
        print(f"  胜率: {win_rate:.2%}")
        print(f"  平均收益: {avg_return:.4f}")
        print(f"  收益标准差: {std_return:.4f}")
        print(f"  夏普比率: {sharpe:.4f}")
    
    # 总结
    print("\n" + "="*80)
    print("参数对比总结:")
    print("="*80)
    for r in results:
        print(f"{r['name']:20s} | 胜率: {r['win_rate']:.2%} | 平均收益: {r['avg_return']:.4f} | 夏普: {r['sharpe']:.4f}")
    
    return results


def main():
    """主函数"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                Triple Barrier 使用示例                       ║
║                                                              ║
║  Triple Barrier 是一种先进的金融时间序列标注方法             ║
║  通过设置止盈、止损和时间限制三个屏障来生成标签              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # 选择要运行的示例
    print("请选择要运行的示例:")
    print("  1 - 基本使用：生成 Triple Barrier 标签")
    print("  2 - 完整流程：使用 Triple Barrier 训练模型")
    print("  3 - 参数对比：比较不同参数效果")
    print("  0 - 运行所有示例")
    
    choice = input("\n请输入选择 (0-3): ").strip()
    
    try:
        if choice == '1':
            strategy = example1_basic_usage()
            plt.show(block=True)
        elif choice == '2':
            strategy = example2_full_pipeline()
            plt.show(block=True)
        elif choice == '3':
            results = example3_parameter_comparison()
        elif choice == '0':
            example1_basic_usage()
            example3_parameter_comparison()
            strategy = example2_full_pipeline()
            plt.show(block=True)
        else:
            print("❌ 无效的选择")
            return
        
        print("\n✅ 示例运行完成！")
        
    except FileNotFoundError as e:
        print(f"\n❌ 文件未找到: {e}")
        print("\n💡 提示:")
        print("  1. 确保 YAML 配置文件存在")
        print("  2. 确保因子 CSV 文件存在")
        print("  3. 检查数据路径配置是否正确")
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

