#!/usr/bin/env python3
"""
特征类型筛选示例脚本

演示如何使用data_prepare_rolling函数进行针对性特征挖掘：
- 动量特征挖掘
- 大单特征挖掘  
- 订单流特征挖掘
- 价格冲击特征挖掘
- 组合特征挖掘
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gp_crypto_next.dataload import data_prepare_rolling
import pandas as pd
import numpy as np

def test_momentum_features():
    """测试动量特征筛选"""
    print("\n" + "="*60)
    print("测试动量特征筛选")
    print("="*60)
    
    try:
        # 只筛选动量相关特征
        result = data_prepare_rolling(
            sym='BTCUSDT',
            freq='30min',
            start_date_train='2025-01-01',
            end_date_train='2025-01-15', 
            start_date_test='2025-01-16',
            end_date_test='2025-01-20',
            feature_types=['momentum'],
            file_path='gp_crypto_next/gp_models/BTCUSDT_30min_1_2025-01_2025-04_2025-04_2025-05.csv'
        )
        
        X_all, X_train, y_train, ret_train, X_test, y_test, ret_test, feature_names, open_train, open_test, close_train, close_test, index, bars_df = result
        
        print(f"✓ 动量特征筛选成功")
        print(f"筛选出的动量特征数量: {len(feature_names)}")
        print(f"动量特征列表: {feature_names}")
        print(f"训练集形状: {X_train.shape}")
        print(f"测试集形状: {X_test.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 动量特征筛选失败: {e}")
        return False

def test_tail_features():
    """测试大单特征筛选"""
    print("\n" + "="*60)
    print("测试大单特征筛选")
    print("="*60)
    
    try:
        # 只筛选大单相关特征
        result = data_prepare_rolling(
            sym='BTCUSDT',
            freq='30min',
            start_date_train='2025-01-01',
            end_date_train='2025-01-15',
            start_date_test='2025-01-16', 
            end_date_test='2025-01-20',
            feature_types=['tail'],
            file_path='gp_crypto_next/gp_models/BTCUSDT_30min_1_2025-01_2025-04_2025-04_2025-05.csv'
        )
        
        X_all, X_train, y_train, ret_train, X_test, y_test, ret_test, feature_names, open_train, open_test, close_train, close_test, index, bars_df = result
        
        print(f"✓ 大单特征筛选成功")
        print(f"筛选出的大单特征数量: {len(feature_names)}")
        print(f"大单特征列表: {feature_names}")
        print(f"训练集形状: {X_train.shape}")
        print(f"测试集形状: {X_test.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 大单特征筛选失败: {e}")
        return False

def test_orderflow_features():
    """测试订单流特征筛选"""
    print("\n" + "="*60)
    print("测试订单流特征筛选")
    print("="*60)
    
    try:
        # 只筛选订单流相关特征
        result = data_prepare_rolling(
            sym='BTCUSDT',
            freq='30min',
            start_date_train='2025-01-01',
            end_date_train='2025-01-15',
            start_date_test='2025-01-16',
            end_date_test='2025-01-20',
            feature_types=['orderflow'],
            file_path='gp_crypto_next/gp_models/BTCUSDT_30min_1_2025-01_2025-04_2025-04_2025-05.csv'
        )
        
        X_all, X_train, y_train, ret_train, X_test, y_test, ret_test, feature_names, open_train, open_test, close_train, close_test, index, bars_df = result
        
        print(f"✓ 订单流特征筛选成功")
        print(f"筛选出的订单流特征数量: {len(feature_names)}")
        print(f"订单流特征列表: {feature_names}")
        print(f"训练集形状: {X_train.shape}")
        print(f"测试集形状: {X_test.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 订单流特征筛选失败: {e}")
        return False

def test_impact_features():
    """测试价格冲击特征筛选"""
    print("\n" + "="*60)
    print("测试价格冲击特征筛选")
    print("="*60)
    
    try:
        # 只筛选价格冲击相关特征
        result = data_prepare_rolling(
            sym='BTCUSDT',
            freq='30min',
            start_date_train='2025-01-01',
            end_date_train='2025-01-15',
            start_date_test='2025-01-16',
            end_date_test='2025-01-20',
            feature_types=['impact'],
            file_path='gp_crypto_next/gp_models/BTCUSDT_30min_1_2025-01_2025-04_2025-04_2025-05.csv'
        )
        
        X_all, X_train, y_train, ret_train, X_test, y_test, ret_test, feature_names, open_train, open_test, close_train, close_test, index, bars_df = result
        
        print(f"✓ 价格冲击特征筛选成功")
        print(f"筛选出的价格冲击特征数量: {len(feature_names)}")
        print(f"价格冲击特征列表: {feature_names}")
        print(f"训练集形状: {X_train.shape}")
        print(f"测试集形状: {X_test.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 价格冲击特征筛选失败: {e}")
        return False

def test_keyword_filtering():
    """测试关键词筛选"""
    print("\n" + "="*60)
    print("测试关键词筛选")
    print("="*60)
    
    try:
        # 使用关键词筛选包含'lambda'的特征
        result = data_prepare_rolling(
            sym='BTCUSDT',
            freq='30min',
            start_date_train='2025-01-01',
            end_date_train='2025-01-15',
            start_date_test='2025-01-16',
            end_date_test='2025-01-20',
            feature_keywords=['lambda'],
            file_path='gp_crypto_next/gp_models/BTCUSDT_30min_1_2025-01_2025-04_2025-04_2025-05.csv'
        )
        
        X_all, X_train, y_train, ret_train, X_test, y_test, ret_test, feature_names, open_train, open_test, close_train, close_test, index, bars_df = result
        
        print(f"✓ 关键词筛选成功")
        print(f"筛选出的包含'lambda'的特征数量: {len(feature_names)}")
        print(f"特征列表: {feature_names}")
        print(f"训练集形状: {X_train.shape}")
        print(f"测试集形状: {X_test.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 关键词筛选失败: {e}")
        return False

def test_combined_filtering():
    """测试组合筛选"""
    print("\n" + "="*60)
    print("测试组合筛选（动量+大单特征）")
    print("="*60)
    
    try:
        # 组合筛选：动量特征 + 大单特征
        result = data_prepare_rolling(
            sym='BTCUSDT',
            freq='30min',
            start_date_train='2025-01-01',
            end_date_train='2025-01-15',
            start_date_test='2025-01-16',
            end_date_test='2025-01-20',
            feature_types=['momentum', 'tail'],
            file_path='gp_crypto_next/gp_models/BTCUSDT_30min_1_2025-01_2025-04_2025-04_2025-05.csv'
        )
        
        X_all, X_train, y_train, ret_train, X_test, y_test, ret_test, feature_names, open_train, open_test, close_train, close_test, index, bars_df = result
        
        print(f"✓ 组合筛选成功")
        print(f"筛选出的动量+大单特征数量: {len(feature_names)}")
        print(f"特征列表: {feature_names}")
        print(f"训练集形状: {X_train.shape}")
        print(f"测试集形状: {X_test.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ 组合筛选失败: {e}")
        return False

def main():
    """主函数：运行所有测试"""
    print("开始测试特征类型筛选功能...")
    
    # 检查数据文件是否存在
    data_file = 'gp_crypto_next/gp_models/BTCUSDT_30min_1_2025-01_2025-04_2025-04_2025-05.csv'
    if not os.path.exists(data_file):
        print(f"⚠️  数据文件不存在: {data_file}")
        print("请确保数据文件存在后再运行测试")
        return
    
    # 运行各项测试
    tests = [
        ("动量特征筛选", test_momentum_features),
        ("大单特征筛选", test_tail_features), 
        ("订单流特征筛选", test_orderflow_features),
        ("价格冲击特征筛选", test_impact_features),
        ("关键词筛选", test_keyword_filtering),
        ("组合筛选", test_combined_filtering),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n正在运行: {test_name}")
        success = test_func()
        results.append((test_name, success))
    
    # 汇总结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    for test_name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{test_name}: {status}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\n总计: {passed}/{total} 项测试通过")

if __name__ == "__main__":
    main()
