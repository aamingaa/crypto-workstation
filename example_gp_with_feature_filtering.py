#!/usr/bin/env python3
"""
在GP分析中使用特征筛选的示例

演示如何在main_gp_new.py中配置和使用特征类型筛选功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gp_crypto_next.main_gp_new import GPAnalyzer

def example_momentum_strategy():
    """示例：动量策略 - 只使用动量特征"""
    print("\n" + "="*60)
    print("示例：动量策略 - 只使用动量特征")
    print("="*60)
    
    # 创建GP分析器
    analyzer = GPAnalyzer()
    
    # 基础配置
    analyzer.sym = 'BTCUSDT'
    analyzer.freq = '30min'
    analyzer.start_date_train = '2025-01-01'
    analyzer.end_date_train = '2025-01-15'
    analyzer.start_date_test = '2025-01-16'
    analyzer.end_date_test = '2025-01-20'
    analyzer.data_source = 'rolling'  # 使用滚动统计版数据准备
    analyzer.file_path = 'gp_crypto_next/gp_models/BTCUSDT_30min_1_2025-01_2025-04_2025-04_2025-05.csv'
    
    # 特征筛选配置 - 只使用动量特征
    analyzer.feature_types = ['momentum']
    analyzer.feature_keywords = None
    
    print(f"配置信息:")
    print(f"  交易对: {analyzer.sym}")
    print(f"  频率: {analyzer.freq}")
    print(f"  特征类型: {analyzer.feature_types}")
    print(f"  数据文件: {analyzer.file_path}")
    
    try:
        # 初始化数据
        analyzer.initialize_data()
        print(f"✓ 数据初始化成功")
        print(f"  训练集特征数量: {len(analyzer.feature_names)}")
        print(f"  特征列表: {analyzer.feature_names}")
        
        # 这里可以继续GP分析...
        # analyzer.run_analysis()
        
    except Exception as e:
        print(f"✗ 初始化失败: {e}")

def example_tail_flow_strategy():
    """示例：大单+订单流策略"""
    print("\n" + "="*60)
    print("示例：大单+订单流策略")
    print("="*60)
    
    analyzer = GPAnalyzer()
    
    # 基础配置
    analyzer.sym = 'BTCUSDT'
    analyzer.freq = '30min'
    analyzer.start_date_train = '2025-01-01'
    analyzer.end_date_train = '2025-01-15'
    analyzer.start_date_test = '2025-01-16'
    analyzer.end_date_test = '2025-01-20'
    analyzer.data_source = 'rolling'
    analyzer.file_path = 'gp_crypto_next/gp_models/BTCUSDT_30min_1_2025-01_2025-04_2025-04_2025-05.csv'
    
    # 特征筛选配置 - 大单+订单流特征
    analyzer.feature_types = ['tail', 'orderflow']
    analyzer.feature_keywords = None
    
    print(f"配置信息:")
    print(f"  交易对: {analyzer.sym}")
    print(f"  频率: {analyzer.freq}")
    print(f"  特征类型: {analyzer.feature_types}")
    print(f"  数据文件: {analyzer.file_path}")
    
    try:
        analyzer.initialize_data()
        print(f"✓ 数据初始化成功")
        print(f"  训练集特征数量: {len(analyzer.feature_names)}")
        print(f"  特征列表: {analyzer.feature_names}")
        
    except Exception as e:
        print(f"✗ 初始化失败: {e}")

def example_lambda_keyword_strategy():
    """示例：关键词筛选 - 只使用包含lambda的特征"""
    print("\n" + "="*60)
    print("示例：关键词筛选 - 只使用包含lambda的特征")
    print("="*60)
    
    analyzer = GPAnalyzer()
    
    # 基础配置
    analyzer.sym = 'BTCUSDT'
    analyzer.freq = '30min'
    analyzer.start_date_train = '2025-01-01'
    analyzer.end_date_train = '2025-01-15'
    analyzer.start_date_test = '2025-01-16'
    analyzer.end_date_test = '2025-01-20'
    analyzer.data_source = 'rolling'
    analyzer.file_path = 'gp_crypto_next/gp_models/BTCUSDT_30min_1_2025-01_2025-04_2025-04_2025-05.csv'
    
    # 特征筛选配置 - 关键词筛选
    analyzer.feature_types = None
    analyzer.feature_keywords = ['lambda']
    
    print(f"配置信息:")
    print(f"  交易对: {analyzer.sym}")
    print(f"  频率: {analyzer.freq}")
    print(f"  关键词: {analyzer.feature_keywords}")
    print(f"  数据文件: {analyzer.file_path}")
    
    try:
        analyzer.initialize_data()
        print(f"✓ 数据初始化成功")
        print(f"  训练集特征数量: {len(analyzer.feature_names)}")
        print(f"  特征列表: {analyzer.feature_names}")
        
    except Exception as e:
        print(f"✗ 初始化失败: {e}")

def example_combined_strategy():
    """示例：组合筛选 - 价格冲击类型 + lambda关键词"""
    print("\n" + "="*60)
    print("示例：组合筛选 - 价格冲击类型 + lambda关键词")
    print("="*60)
    
    analyzer = GPAnalyzer()
    
    # 基础配置
    analyzer.sym = 'BTCUSDT'
    analyzer.freq = '30min'
    analyzer.start_date_train = '2025-01-01'
    analyzer.end_date_train = '2025-01-15'
    analyzer.start_date_test = '2025-01-16'
    analyzer.end_date_test = '2025-01-20'
    analyzer.data_source = 'rolling'
    analyzer.file_path = 'gp_crypto_next/gp_models/BTCUSDT_30min_1_2025-01_2025-04_2025-04_2025-05.csv'
    
    # 特征筛选配置 - 组合筛选
    analyzer.feature_types = ['impact']
    analyzer.feature_keywords = ['lambda']
    
    print(f"配置信息:")
    print(f"  交易对: {analyzer.sym}")
    print(f"  频率: {analyzer.freq}")
    print(f"  特征类型: {analyzer.feature_types}")
    print(f"  关键词: {analyzer.feature_keywords}")
    print(f"  数据文件: {analyzer.file_path}")
    
    try:
        analyzer.initialize_data()
        print(f"✓ 数据初始化成功")
        print(f"  训练集特征数量: {len(analyzer.feature_names)}")
        print(f"  特征列表: {analyzer.feature_names}")
        
    except Exception as e:
        print(f"✗ 初始化失败: {e}")

def main():
    """主函数：运行所有示例"""
    print("GP分析中特征筛选功能示例")
    print("=" * 60)
    
    # 检查数据文件是否存在
    data_file = 'gp_crypto_next/gp_models/BTCUSDT_30min_1_2025-01_2025-04_2025-04_2025-05.csv'
    if not os.path.exists(data_file):
        print(f"⚠️  数据文件不存在: {data_file}")
        print("请确保数据文件存在后再运行示例")
        return
    
    # 运行各种策略示例
    examples = [
        ("动量策略", example_momentum_strategy),
        ("大单+订单流策略", example_tail_flow_strategy),
        ("关键词筛选策略", example_lambda_keyword_strategy),
        ("组合筛选策略", example_combined_strategy),
    ]
    
    for example_name, example_func in examples:
        print(f"\n正在运行: {example_name}")
        example_func()
    
    print("\n" + "="*60)
    print("所有示例运行完成！")
    print("="*60)
    print("\n使用说明:")
    print("1. 在GPAnalyzer中设置 feature_types 和 feature_keywords 属性")
    print("2. 确保 data_source 设置为 'rolling'")
    print("3. 调用 initialize_data() 进行数据初始化")
    print("4. 继续执行GP分析流程")

if __name__ == "__main__":
    main()
