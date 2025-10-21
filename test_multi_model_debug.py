#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
调试版本 - 用于诊断问题
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 设置路径
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
pkg_dir = project_root / "gp_crypto_next"
if str(pkg_dir) not in sys.path:
    sys.path.insert(0, str(pkg_dir))

from multi_model_main import QuantTradingStrategy
import matplotlib.pyplot as plt

def main():
    print("="*60)
    print("开始调试 Multi-Model 策略")
    print("="*60)
    
    # YAML 配置文件
    yaml_path = 'gp_crypto_next/coarse_grain_parameters.yaml'
    
    # 因子文件
    factor_csv_path = 'gp_models/ETHUSDT_15m_1_2025-01-01_2025-01-20_2025-01-20_2025-01-31.csv.gz'
    
    # 策略配置
    strategy_config = {
        'corr_threshold': 0.5,
        'max_factors': 30,
        'fees_rate': 0.0005,
        'model_save_path': './models',
    }
    
    try:
        # 创建策略
        print("\n1. 创建策略实例...")
        strategy = QuantTradingStrategy.from_yaml(
            yaml_path=yaml_path,
            factor_csv_path=factor_csv_path,
            strategy_config=strategy_config
        )
        
        print("\n2. 加载数据...")
        strategy.load_data_from_dataload()
        
        print(f"\n数据加载检查:")
        print(f"  X_all shape: {strategy.X_all.shape}")
        print(f"  训练集大小: {len(strategy.y_train)}")
        print(f"  测试集大小: {len(strategy.y_test)}")
        print(f"  open_train type: {type(strategy.open_train)}, shape: {strategy.open_train.shape if hasattr(strategy.open_train, 'shape') else len(strategy.open_train)}")
        print(f"  open_test type: {type(strategy.open_test)}, shape: {strategy.open_test.shape if hasattr(strategy.open_test, 'shape') else len(strategy.open_test)}")
        
        print("\n3. 加载因子表达式...")
        strategy.load_factor_expressions()
        
        print("\n4. 评估因子...")
        strategy.evaluate_factor_expressions()
        
        print("\n5. 标准化和筛选因子...")
        strategy.normalize_factors()
        strategy.select_factors()
        
        print("\n6. 准备训练数据...")
        strategy.prepare_training_data()
        
        print("\n7. 训练模型...")
        strategy.train_models()
        
        print("\n8. 生成预测...")
        strategy.make_predictions(weight_method='equal')
        
        print("\n9. 回测...")
        strategy.backtest_all_models()
        
        print("\n10. 性能汇总:")
        summary = strategy.get_performance_summary()
        
        print("\n11. 绘制结果...")
        strategy.plot_results('Ensemble')
        
        print("\n" + "="*60)
        print("调试完成！")
        print("="*60)
        
        # 显示图表
        plt.show(block=True)
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

