'''
数据读取、降频处理和计算收益率模块
'''

import pandas as pd
import numpy as np
from pathlib import Path
import originalFeature
from scipy.optimize import minimize
import time
import talib as ta
from enum import Enum
import re
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

import sys
import matplotlib.pyplot as plt
from scipy.stats import zscore, kurtosis, skew, yeojohnson, boxcox
from scipy.stats import tukeylambda, mstats
from sklearn.preprocessing import RobustScaler
import zipfile
from io import BytesIO

# 微结构（bar级）稳健代理函数
# from utils.microstructure_features import (
#     get_roll_measure,
#     get_corwin_schultz_estimator,
#     get_bekker_parkinson_vol,
#     get_bar_based_kyle_lambda,
#     get_bar_based_amihud_lambda,
#     get_bar_based_hasbrouck_lambda,
# )

# 可选：基于逐笔数据构建 bars 并提取 features/ 目录下的微结构特征
from data.time_bars import TimeBarBuilder
from data.dollar_bars import DollarBarBuilder
from data.trades_processor import TradesProcessor
from features.orderflow_features import OrderFlowFeatureExtractor
from features.impact_features import PriceImpactFeatureExtractor
from features.tail_features import TailFeatureExtractor
from features.bucketed_flow_features import BucketedFlowFeatureExtractor
from features.microstructure_extractor import MicrostructureFeatureExtractor
from pipeline.trading_pipeline import TradingPipeline


class DataFrequency(Enum):
    """数据频率枚举"""
    MONTHLY = 'monthly'  # 月度数据
    DAILY = 'daily'      # 日度数据


def data_load(sym: str) -> pd.DataFrame:
    '''数据读取模块（原版）'''
    file_name = '/home/etern/crypto/data/merged/merged/' + sym + '-merged-without-rfr-1m.csv'  
    z = pd.read_csv(file_name, index_col=1)[
        ['o', 'h', 'l', 'c', 'vol', 'vol_ccy', 'trades',
               'oi', 'oi_ccy', 'toptrader_count_lsr', 'toptrader_oi_lsr', 'count_lsr',
               'taker_vol_lsr']]
    return z


def _generate_date_range(start_date: str, end_date: str, read_frequency: DataFrequency = DataFrequency.MONTHLY) -> List[str]:
    """
    生成日期范围列表
    
    参数:
    start_date: 起始日期
        - 月度格式: 'YYYY-MM' (如 '2020-01') 或 'YYYY-MM-DD' (自动转换为 'YYYY-MM')
        - 日度格式: 'YYYY-MM-DD' (如 '2020-01-01')
    end_date: 结束日期，格式同上
    frequency: 数据频率（月度或日度）
    
    返回:
    日期字符串列表
    """
    if read_frequency == DataFrequency.MONTHLY:
        # 兼容 'YYYY-MM' 和 'YYYY-MM-DD' 两种格式
        # 如果是 'YYYY-MM-DD' 格式，自动截取为 'YYYY-MM'
        new_start_date = start_date
        new_end_date = end_date
        if len(start_date) == 10:  # 'YYYY-MM-DD' 格式
            new_start_date = start_date[:7]
        if len(end_date) == 10:
            new_end_date = end_date[:7]
            
        start_dt = datetime.strptime(new_start_date, '%Y-%m')
        end_dt = datetime.strptime(new_end_date, '%Y-%m')
        
        date_list = []
        current_dt = start_dt
        while current_dt <= end_dt:
            date_list.append(current_dt.strftime('%Y-%m'))
            # 移动到下一个月
            if current_dt.month == 12:
                current_dt = current_dt.replace(year=current_dt.year + 1, month=1)
            else:
                current_dt = current_dt.replace(month=current_dt.month + 1)
        
        return date_list
    
    elif read_frequency == DataFrequency.DAILY:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        date_list = []
        current_dt = start_dt
        while current_dt <= end_dt:
            date_list.append(current_dt.strftime('%Y-%m-%d'))
            current_dt += timedelta(days=1)
        
        return date_list
    
    else:
        raise ValueError(f"不支持的数据频率: {frequency}")


def _build_file_paths(sym: str, date_str: str, data_dir: str, timeframe: str = '1m', 
                      frequency: DataFrequency = DataFrequency.MONTHLY) -> Tuple[str, str, str]:
    """
    构建文件路径
    
    参数:
    sym: 交易对符号
    date_str: 日期字符串
    data_dir: 数据目录
    timeframe: 时间周期 (如 '1m', '5m', '1h')
    frequency: 数据频率
    
    返回:
    (file_base_name, feather_path, zip_path) 元组
    """
    if frequency == DataFrequency.MONTHLY:
        file_base_name = f"{sym}-{timeframe}-{date_str}"
    elif frequency == DataFrequency.DAILY:
        file_base_name = f"{sym}-{timeframe}-{date_str}"
    else:
        raise ValueError(f"不支持的数据频率: {frequency}")
    
    # /Volumes/Ext-Disk/data/futures/um/monthly/klines/ETHUSDT/15m/2025/ETHUSDT-15m-2025-01.feather
    year = date_str.split('-')[0]
    feather_path = os.path.join(f'{data_dir}/{year}', f"{file_base_name}.feather")
    zip_path = os.path.join(f'{data_dir}/{year}', f"{file_base_name}.zip")
    
    return file_base_name, feather_path, zip_path


def _read_feather_file(feather_path: str) -> Optional[pd.DataFrame]:
    """
    读取 feather 格式文件
    
    参数:
    feather_path: feather 文件路径
    
    返回:
    DataFrame 或 None（如果读取失败）
    """
    if not os.path.exists(feather_path):
        return None
    
    try:
        df = pd.read_feather(feather_path)
        print(f"✓ 成功读取 feather: {os.path.basename(feather_path)}, 行数: {len(df)}")
        return df
    except Exception as e:
        print(f"✗ 读取 feather 文件失败: {os.path.basename(feather_path)}, 错误: {str(e)}")
        return None


def _read_zip_file(zip_path: str, file_base_name: str, save_feather: bool = True) -> Optional[pd.DataFrame]:
    """
    读取 zip 格式文件（内含 CSV）
    
    参数:
    zip_path: zip 文件路径
    file_base_name: 文件基础名称（不含扩展名）
    save_feather: 是否保存为 feather 格式以加速后续读取
    
    返回:
    DataFrame 或 None（如果读取失败）
    """
    if not os.path.exists(zip_path):
        return None
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 获取 zip 中的 csv 文件名
            csv_filename = f"{file_base_name}.csv"
            
            if csv_filename not in zip_ref.namelist():
                # 如果找不到，尝试使用第一个 csv 文件
                csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                if csv_files:
                    csv_filename = csv_files[0]
                else:
                    print(f"✗ 在 {os.path.basename(zip_path)} 中找不到 CSV 文件")
                    return None
            
            # 读取 CSV 数据
            with zip_ref.open(csv_filename) as csv_file:
                df = pd.read_csv(csv_file)
                print(f"✓ 成功读取 zip: {os.path.basename(zip_path)}, 行数: {len(df)}")
                
                # 可选：保存为 feather 格式以加速后续读取
                if save_feather:
                    feather_path = zip_path.replace('.zip', '.feather')
                    try:
                        df.to_feather(feather_path)
                        print(f"  → 已缓存为 feather 格式")
                    except Exception as e:
                        print(f"  → 保存 feather 文件失败: {str(e)}")
                
                return df
    
    except Exception as e:
        print(f"✗ 读取 zip 文件失败: {os.path.basename(zip_path)}, 错误: {str(e)}")
        return None


def _read_single_period_data(sym: str, date_str: str, data_dir: str, timeframe: str = '1m',
                             frequency: DataFrequency = DataFrequency.MONTHLY) -> Optional[pd.DataFrame]:
    """
    读取单个时间段的数据（优先 feather，其次 zip）
    
    参数:
    sym: 交易对符号
    date_str: 日期字符串
    data_dir: 数据目录
    timeframe: 时间周期
    frequency: 数据频率
    
    返回:
    DataFrame 或 None
    """
    file_base_name, feather_path, zip_path = _build_file_paths(sym, date_str, data_dir, timeframe, frequency)
    
    # 优先读取 feather
    df = _read_feather_file(feather_path)
    if df is not None:
        return df
    
    # 如果 feather 不存在，读取 zip
    df = _read_zip_file(zip_path, file_base_name, save_feather=True)
    if df is not None:
        return df
    
    # 两种文件都不存在
    print(f"⚠ 警告：文件不存在，跳过: {file_base_name}")
    return None


def _read_tick_data_file(file_path: str) -> pd.DataFrame:
    """
    读取 tick 级别交易数据文件（支持 feather / zip / csv）
    
    参数:
    file_path: 文件路径
    
    返回:
    Tick 数据 DataFrame（包含 time, price, quantity, side 等列）
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Tick 数据文件不存在: {file_path}")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.feather':
            df = pd.read_feather(file_path)
            print(f"✓ 成功读取 tick feather 文件，行数: {len(df):,}")
        elif file_ext == '.zip':
            df = pd.read_csv(file_path, compression='zip')
            print(f"✓ 成功读取 tick zip 文件，行数: {len(df):,}")
        elif file_ext == '.csv':
            df = pd.read_csv(file_path)
            print(f"✓ 成功读取 tick csv 文件，行数: {len(df):,}")
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}，仅支持 .feather / .zip / .csv")
        
        # 确保时间列是 datetime 类型
        if 'time' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['time']):
                df['time'] = pd.to_datetime(df['time'], unit='ms', errors='coerce')
        
        print(f"Tick 数据时间范围: {df['time'].min()} 至 {df['time'].max()}")
        
        return df
        
    except Exception as e:
        raise ValueError(f"读取 Tick 数据文件失败: {file_path}\n错误: {str(e)}")


def _read_direct_file(file_path: str) -> pd.DataFrame:
    """
    直接读取指定路径的 K线数据文件（支持 feather / zip / csv）
    
    参数:
    file_path: 文件路径
    
    返回:
    标准化后的 K线 DataFrame
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    print(f"\n{'='*60}")
    print(f"直接读取 K线文件: {os.path.basename(file_path)}")
    print(f"{'='*60}\n")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.feather':
            df = pd.read_feather(file_path)
            print(f"✓ 成功读取 feather 文件，行数: {len(df):,}")
        elif file_ext == '.zip':
            df = pd.read_csv(file_path, compression='zip')
            print(f"✓ 成功读取 zip 文件，行数: {len(df):,}")
        elif file_ext == '.csv':
            df = pd.read_csv(file_path)
            print(f"✓ 成功读取 csv 文件，行数: {len(df):,}")
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}，仅支持 .feather / .zip / .csv")
        
        # 标准化列名和索引
        standardized_df = _standardize_dataframe_columns(df)
        print(f"数据时间范围: {standardized_df.index.min()} 至 {standardized_df.index.max()}")
        print(f"{'='*60}\n")
        
        return standardized_df
        
    except Exception as e:
        raise ValueError(f"读取文件失败: {file_path}\n错误: {str(e)}")


def _standardize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    标准化 DataFrame 列名并设置索引
    
    参数:
    df: 原始 DataFrame（包含 Binance 格式的列名）
    
    返回:
    标准化后的 DataFrame
    """
    # 将 open_time 转换为 datetime 并设置为索引
    df = df.copy()
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)

    # df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    # df.set_index('close_time', inplace=True)
    
    # 列名映射：新列名 -> 旧列名
    # 新列名: open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore
    # 旧列名: o, h, l, c, vol, vol_ccy, trades, oi, oi_ccy, toptrader_count_lsr, toptrader_oi_lsr, count_lsr, taker_vol_lsr
    column_mapping = {
        'open': 'o',
        'high': 'h',
        'low': 'l',
        'close': 'c',
        'volume': 'vol',
        'quote_volume': 'vol_ccy',
        'count': 'trades',
        'close_time': 'close_time',
    }
    
    df = df.rename(columns=column_mapping)
    
    # 选择需要的列，对于缺失的列用 0 填充
    required_columns = ['o', 'h', 'l', 'c', 'vol', 'vol_ccy', 'trades',
                       'oi', 'oi_ccy', 'toptrader_count_lsr', 'toptrader_oi_lsr', 'count_lsr',
                       'taker_vol_lsr', 'close_time']
    
    # 为缺失的列添加默认值 0
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0
            print(f"⚠ 警告：列 '{col}' 不存在，已填充为 0")
    
    return df[required_columns]


def data_load_v2(sym: str, data_dir: str, start_date: str, end_date: str, 
                 timeframe: str = '1h', read_frequency: str = 'monthly',
                 file_path: Optional[str] = None) -> pd.DataFrame:
    """
    数据读取模块 V2 - 支持从多种时间粒度的数据文件读取
    
    参数:
    sym: 交易对符号，例如 'BTCUSDT'
    data_dir: 数据目录路径，例如 '/Volumes/Ext-Disk/data/futures/um/monthly/klines/BTCUSDT/1m'
    start_date: 起始日期
        - 月度格式: 'YYYY-MM' (如 '2020-01')
        - 日度格式: 'YYYY-MM-DD' (如 '2020-01-01')
    end_date: 结束日期，格式同上
    timeframe: 时间周期，默认 '1m'，可选 '5m', '1h' 等
    frequency: 数据频率，'monthly'（月度）或 'daily'（日度）
    file_path: 直接指定文件路径（支持 .feather / .zip / .csv），指定后将忽略其他参数
    
    返回:
    包含标准化列名的 DataFrame
    
    文件读取优先级:
    1. 如果指定 file_path，直接读取该文件
    2. 否则按日期范围读取，优先读取 .feather 格式文件（如果存在）
    3. 如果 .feather 不存在，则读取 .zip 文件，并自动缓存为 .feather
    
    示例:
    # 读取月度数据
    df = data_load_v2('BTCUSDT', '/path/to/monthly', '2020-01', '2024-09', frequency='monthly')
    
    # 读取日度数据
    df = data_load_v2('BTCUSDT', '/path/to/daily', '2020-01-01', '2020-01-31', frequency='daily')
    
    # 直接读取单个文件
    df = data_load_v2('BTCUSDT', '', '', '', file_path='/path/to/data.feather')
    """
    
    # 如果指定了直接文件路径，直接读取
    if file_path:
        return _read_direct_file(file_path)
    
    # 解析频率参数
    try:
        freq_enum = DataFrequency(read_frequency.lower())
    except ValueError:
        raise ValueError(f"不支持的数据频率: {read_frequency}，仅支持 'monthly' 或 'daily'")
    
    # 生成日期范围
    date_list = _generate_date_range(start_date, end_date, freq_enum)
    
    # 读取所有时间段的数据
    df_list = []
    success_count = 0
    failed_count = 0
    
    for date_str in date_list:
        df = _read_single_period_data(sym, date_str, data_dir, timeframe, freq_enum)
        if df is not None:
            df_list.append(df)
            success_count += 1
        else:
            failed_count += 1
    
    # 检查是否成功读取到数据
    if not df_list:
        raise ValueError(f"未能成功读取任何数据文件，请检查路径和日期范围\n路径: {data_dir}\n日期: {start_date} ~ {end_date}")
    
    print(f"\n{'='*60}")
    print(f"读取完成: 成功 {success_count} 个，失败 {failed_count} 个")
    print(f"{'='*60}\n")
    
    # 合并所有数据
    merged_df = pd.concat(df_list, ignore_index=True)
    print(f"合并后总行数: {len(merged_df):,}")
    
    # 标准化列名和索引
    standardized_df = _standardize_dataframe_columns(merged_df)
    
    print(f"数据时间范围: {standardized_df.index.min()} 至 {standardized_df.index.max()}")
    print(f"{'='*60}\n")
    
    return standardized_df



def removed_zero_vol_dataframe(df):
    """
    打印并且返回-
    1. volume这一列为0的行组成的df
    2. low这一列的最小值
    3. volume这一列的最小值
    5. 去除掉volume=0的行的dataframe
    -------

    """
    # 将DataFrame的索引列设置为'datetime'
    df.index = pd.to_datetime(df.index)

    # 1. volume这一列为0的行组成的df
    volume_zero_df = df[df['vol'] == 0]
    print(f"Volume为0的行组成的DataFrame: {len(volume_zero_df)}")

    # 2. low这一列的最小值
    min_low = df['l'].min()
    print(f"Low这一列的最小值: {min_low}")

    # 3. volume这一列的最小值
    min_volume = df['vol'].min()
    print(f"Volume这一列的最小值: {min_volume}")

    # 5. 去除掉volume=0的行的dataframe
    removed_zero_vol_df = df[df['vol'] != 0]
    print(f"去除掉Volume为0的行之前的DataFrame length: {len(df)}")
    print(f"去除掉Volume为0的行之后的DataFrame length: {len(removed_zero_vol_df)}")

    return removed_zero_vol_df


def resample(z: pd.DataFrame, freq: str) -> pd.DataFrame:
    '''
    这是不支持vwap的，默认读入的数据是没有turnover信息，自然也没有vwap的信息，不需要获取sym的乘数
    '''
    if freq == '15m':
        return z
    
    if freq != '1min' or freq != '1m':
        z.index = pd.to_datetime(z.index)
        # 注意closed和label参数
        z = z.resample(freq, closed='left', label='left').agg({'o': 'first',
                                                               'h': 'max',
                                                               'l': 'min',
                                                               'c': 'last',
                                                               'vol': 'sum',
                                                               'vol_ccy': 'sum',
                                                               'trades': 'sum',
                                                               'oi': 'last', 
                                                               'oi_ccy': 'last', 
                                                               'toptrader_count_lsr':'last', 
                                                               'toptrader_oi_lsr':'last', 
                                                               'count_lsr':'last',
                                                               'taker_vol_lsr':'last'})
        # 注意resample后,比如以10min为resample的freq，9:00的数据是指9:00到9:10的数据~~
        z = z.fillna(method='ffill')   
        z.columns = ['o', 'h', 'l', 'c', 'vol', 'vol_ccy','trades',
               'oi', 'oi_ccy', 'toptrader_count_lsr', 'toptrader_oi_lsr', 'count_lsr',
               'taker_vol_lsr']

        # 重要，这个删掉0成交的操作，不能给5分钟以内的freq进行操作，因为这种情况还是挺容易出现没有成交的，这会改变本身的分布
        # 使用正则表达式提取开头的数值部分, 判断freq的周期
        match = re.match(r"(\d+)", freq)
        if match:
            int_freq = int(match.group(1))
        if int_freq > 5:
            z = removed_zero_vol_dataframe(z)
    return z


def data_prepare(sym, freq, start_date_train, end_date_train, start_date_test, end_date_test, y_train_ret_period=1,
                 rolling_w=2000, output_format='ndarry', _compute_transformed_series=False, _check_zscore_window_series=False, data_dir='', read_frequency='', timeframe='', file_path=None):
    '''
    内置了一些对于label的分析，比较关键, 但只需要研究和对比时才会开启

        # 对于Label的分析的指导目标，是希望它能够接近正态分布，偏度，峰度接近于0
    # 方案1，先对log_return做clip，完全去除了outlier，再看偏度峰度，决定后续是否rolling_zscore.
    # 方案2，先对log_return做rolling_zscore,
    # (rolling窗口值是2000，暂时当做经验性的参数，取值的自由度来源于詹森不等式和大数定理，都是用数据算出来的)
        # 1. 参数和单点夏普的关系不明确，但是和几万个因子的夏普只和，他们的关系应该存在一定的凸性；
        # 2. 参数的设置应该在维持1的前提下兼顾大数定理；
        # 3. samples拆分成为几个class后样本量仍然符合大数定理；
    # 如上两种方案的对比，当前认为是应该第二种方式，应该是能够保留一部分outlier的信息，相对平衡的减轻outlier的影响

    Note - 最终要把窗口还没积累完全的部分，删除掉这些样本，否则会影响训练的结果。往往是都生成了feature之后，最后处理好label，再做切割。
    '''

    # -----------------------------------------
    # z = data_load(sym)
    z = data_load_v2(sym, data_dir=data_dir, start_date=start_date_train, end_date=end_date_test, timeframe=timeframe, read_frequency=read_frequency, file_path=file_path)
    # 切分数据，只取需要的部分 - train and test
    z.index = pd.to_datetime(z.index)
    print(f'开始处理 {sym} 的历史数据')   
    print(f'len of z before select = {len(z)}')
    z = z[(z.index >= pd.to_datetime(start_date_train)) & (
        z.index <= pd.to_datetime(end_date_test))]  # 只截取参数指定部分dataframe
    print(f'len of z after select = {len(z)}')
    ohlcva_df = resample(z, freq)

    print(f'len of resample_z = {len(ohlcva_df)}')
    # --------------------------------------------------------
    if _compute_transformed_series:
        # 分析label的分布，画出label的各类处理后的 三阶矩，四阶矩
        compute_transformed_series(z.c)
    if _check_zscore_window_series:
        # 画图，展现各种窗口下的label的log_return
        check_zscore_window_series(z, 'c')
    # --------------------------------------------------------
    print('开始生成初始特征')
    base_feature = originalFeature.BaseFeature(ohlcva_df.copy())
    z = base_feature.init_feature_df

    # -----------------------------------------

    # 关键 - 生成ret这一列，这是label数值，整个因子评估体系的基础，要注意分析label分布的skewness, kurtosis等.
    # note - 需要把空值处理掉，因为测试集中的最后的几个空值可能刚好影响测试的持仓效果.
    # 注意使用滑动窗口时，对于没填满的区域，和最后空空值区域，也要有类似的考量，防止刚好碰到极值label引起失真影响。
    print('开始生成ret')
    z['return_f'] = np.log(z['c']).diff(
        y_train_ret_period).shift(-y_train_ret_period)
    z['return_f'] = z['return_f'].fillna(0)
    z['r'] = np.log(z['c']).diff()
    z['r'] = z['r'].fillna(0)


    # ---方案2， 先对label做rolling_zscore---------------
    def norm_ret(x, window=rolling_w):  # 不再用L2 norm，恢复到之前的zscore，然后这里需要做的是给他增加一个周期

        # 注意这个函数是用在return上面的，log1p最小的数值是-1，用于return合适
        x = np.log1p(np.asarray(x))
        factors_data = pd.DataFrame(x, columns=['factor'])
        factors_data = factors_data.replace([np.inf, -np.inf, np.nan], 0.0)
        # factors_mean = factors_data.rolling(
        #     window=window, min_periods=1).mean()
        factors_std = factors_data.rolling(window=window, min_periods=1).std()
        factor_value = factors_data / factors_std
        factor_value = factor_value.replace([np.inf, -np.inf, np.nan], 0.0)
        # factor_value = factor_value.clip(-6, 6)
        return np.nan_to_num(factor_value).flatten()


    # Note - 先强行使用norm_ret看效果
    z['ret_rolling_zscore'] = norm_ret(z['return_f'])

    # 此时，所有的features和label，都用相同窗口做完了rolling处理，为了训练模型的准确性，可以开始删除掉还没有存满窗口的那些行了。去除前window行
    # 重要！！ 如果执行如下这句，会z与上面的ohlcva_df不一致，导致originalFeature.BaseFeature(ohlcva_df)初始化的ohlcva_df与做eval的feature_data不一致
    # z = z.iloc[window-1:]


    kept_index = z.index
    ohlcva_df = ohlcva_df.loc[kept_index]
    open_train = ohlcva_df['o'][(ohlcva_df['o'].index >= pd.to_datetime(start_date_train)) & (
            ohlcva_df['o'].index < pd.to_datetime(end_date_train))]
    open_test = ohlcva_df['o'][(ohlcva_df['o'].index >= pd.to_datetime(start_date_test)) & (
            ohlcva_df['o'].index <= pd.to_datetime(end_date_test))]
    close_train = ohlcva_df['c'][(ohlcva_df['c'].index >= pd.to_datetime(start_date_train)) & (
            ohlcva_df['c'].index < pd.to_datetime(end_date_train))]
    close_test = ohlcva_df['c'][(ohlcva_df['c'].index >= pd.to_datetime(start_date_test)) & (
            ohlcva_df['c'].index <= pd.to_datetime(end_date_test))]

    print('ret_rolling_zscore skew = {}'.format(
        z['ret_rolling_zscore'].skew()))
    print('ret_rolling_zscore kurtosis = {}'.format(
        z['ret_rolling_zscore'].kurtosis()))

    print('return skew = {}'.format(z['return_f'].skew()))
    print('return kurtosis = {}'.format(z['return_f'].kurtosis()))


    # 切分为train和test两个数据集，但是注意，test数据集其实带入了之前的数据的窗口, 是要特意这么做的。
    z_train = z[(z.index >= pd.to_datetime(start_date_train)) & (
        z.index < pd.to_datetime(end_date_train))]  # 只截取参数指定部分dataframe
    z_test = z[(z.index >= pd.to_datetime(start_date_test)) & (
        z.index <= pd.to_datetime(end_date_test))]
    # ------------<label 分析>-------------------------

    # 对于Label的分析的指导目标，是希望它能够接近正态分布，偏度，峰度接近于0
    # 方案1，先对log_return做clip，完全去除了outlier，再看偏度峰度，决定后续是否rolling_zscore.
    # 方案2，先对log_return做rolling_zscore,
    # (rolling窗口值是2000，暂时当做经验性的参数，取值的自由度来源于詹森不等式和大数定理，都是用数据算出来的)
    # 1. 参数和单点夏普的关系不明确，但是和几万个因子的夏普只和，他们的关系应该存在一定的凸性；
    # 2. 参数的设置应该在维持1的前提下兼顾大数定理；
    # 3. samples拆分成为几个class后样本量仍然符合大数定理；
    # 如上两种方案的对比，当前认为是应该第二种方式，应该是能够保留一部分outlier的信息，相对平衡的减轻outlier的影响



    if output_format == 'ndarry':
        y_dataset_train = z_train['ret_rolling_zscore'].values
        y_dataset_test = z_test['ret_rolling_zscore'].values
        ret_dataset_train = z_train['return_f'].values
        ret_dataset_test = z_test['return_f'].values
        # 重要！要删除掉包含未来信息的字段，ret，ret_rolling_zscore
        z_train.drop(['return_f', 'ret_rolling_zscore'], axis=1, inplace=True)
        z_test.drop(['return_f', 'ret_rolling_zscore'], axis=1, inplace=True)
        z.drop(['return_f', 'ret_rolling_zscore'], axis=1, inplace=True)

        X_all = np.where(np.isnan(z), 0, z)
        X_dataset_train = np.where(np.isnan(z_train), 0, z_train)
        X_dataset_test = np.where(np.isnan(z_test), 0, z_test)

    elif output_format == 'dataframe':
        y_dataset_train = z_train['ret_rolling_zscore']
        y_dataset_test = z_test['ret_rolling_zscore']
        ret_dataset_train = z_train['return_f']
        ret_dataset_test = z_test['return_f']
        # 重要！要删除掉包含未来信息的字段，ret，ret_rolling_zscore
        z_train.drop(['return_f', 'ret_rolling_zscore'], axis=1, inplace=True)
        z_test.drop(['return_f', 'ret_rolling_zscore'], axis=1, inplace=True)
        z.drop(['return_f', 'ret_rolling_zscore'], axis=1, inplace=True)

        X_all = z.fillna(0)
        X_dataset_train = z_train.fillna(0)
        X_dataset_test = z_test.fillna(0)

    else:
        print(
            'output_format of data_prepare should be "ndarry" or "dataframe", pls check it ')
        exit(1)

    feature_names = z_train.columns

    print('检查x all是不是等于 x train和y train相加，再检查trian和test以及close和open的形状是否一致')
    # X_all 专门是为做batch prediction的时候，要用X_all生成test集要用到的factor_df, 因为factor的计算需要之前一段window中的feature值
    print(f'检查X_all的形状 {X_all.shape}')
    print(f'检查x dataset train的形状 {X_dataset_train.shape}')
    print(f'检查y dataset train的形状 {y_dataset_train.shape}')
    print(f'检查x all是不是等于train和test相加 {len(X_all)},{len(X_dataset_test)+len(X_dataset_train)}')
    print(f'检查open train的形状 {open_train.shape}')
    print(f'检查close train的形状 {close_train.shape}')
    print(f'检查x dataset test的形状 {X_dataset_test.shape}')
    print(f'检查y dataset test的形状 {y_dataset_test.shape}')
    print(f'检查open test的形状 {open_test.shape}')
    print(f'检查close test的形状 {close_test.shape}')

    # X_all 专门是为做batch prediction的时候，要用X_all生成test集要用到的factor_df, 因为factor的计算需要之前一段window中的feature值
    return X_all, X_dataset_train, y_dataset_train,ret_dataset_train, X_dataset_test, y_dataset_test,ret_dataset_test, feature_names,open_train,open_test,close_train,close_test, z.index ,ohlcva_df


def data_thick_rolling_prepare(sym, freq, start_date_train, end_date_train, start_date_test, end_date_test, y_train_ret_period=1,
                 rolling_w=2000, output_format='ndarry', _compute_transformed_series=False, _check_zscore_window_series=False, data_dir='', read_frequency='', timeframe='', file_path=None):
    '''
    内置了一些对于label的分析，比较关键, 但只需要研究和对比时才会开启

        # 对于Label的分析的指导目标，是希望它能够接近正态分布，偏度，峰度接近于0
    # 方案1，先对log_return做clip，完全去除了outlier，再看偏度峰度，决定后续是否rolling_zscore.
    # 方案2，先对log_return做rolling_zscore,
    # (rolling窗口值是2000，暂时当做经验性的参数，取值的自由度来源于詹森不等式和大数定理，都是用数据算出来的)
        # 1. 参数和单点夏普的关系不明确，但是和几万个因子的夏普只和，他们的关系应该存在一定的凸性；
        # 2. 参数的设置应该在维持1的前提下兼顾大数定理；
        # 3. samples拆分成为几个class后样本量仍然符合大数定理；
    # 如上两种方案的对比，当前认为是应该第二种方式，应该是能够保留一部分outlier的信息，相对平衡的减轻outlier的影响

    Note - 最终要把窗口还没积累完全的部分，删除掉这些样本，否则会影响训练的结果。往往是都生成了feature之后，最后处理好label，再做切割。
    '''

    # -----------------------------------------
    # z = data_load(sym)
    z = data_load_v2(sym, data_dir=data_dir, start_date=start_date_train, end_date=end_date_test, timeframe=timeframe, read_frequency=read_frequency, file_path=file_path)
    # 切分数据，只取需要的部分 - train and test
    z.index = pd.to_datetime(z.index)
    print(f'开始处理 {sym} 的历史数据')   
    print(f'len of z before select = {len(z)}')
    z = z[(z.index >= pd.to_datetime(start_date_train)) & (
        z.index <= pd.to_datetime(end_date_test))]  # 只截取参数指定部分dataframe
    print(f'len of z after select = {len(z)}')
    ohlcva_df = resample(z, freq)

    print(f'len of resample_z = {len(ohlcva_df)}')
    # --------------------------------------------------------
    if _compute_transformed_series:
        # 分析label的分布，画出label的各类处理后的 三阶矩，四阶矩
        compute_transformed_series(z.c)
    if _check_zscore_window_series:
        # 画图，展现各种窗口下的label的log_return
        check_zscore_window_series(z, 'c')
    # --------------------------------------------------------
    print('开始生成初始特征')
    base_feature = originalFeature.BaseFeature(ohlcva_df.copy())
    z = base_feature.init_feature_df

    # -----------------------------------------

    # 关键 - 生成ret这一列，这是label数值，整个因子评估体系的基础，要注意分析label分布的skewness, kurtosis等.
    # note - 需要把空值处理掉，因为测试集中的最后的几个空值可能刚好影响测试的持仓效果.
    # 注意使用滑动窗口时，对于没填满的区域，和最后空空值区域，也要有类似的考量，防止刚好碰到极值label引起失真影响。
    print('开始生成ret')
    z['return_f'] = np.log(z['c']).diff(
        y_train_ret_period).shift(-y_train_ret_period)
    z['return_f'] = z['return_f'].fillna(0)
    z['r'] = np.log(z['c']).diff()
    z['r'] = z['r'].fillna(0)


    # ---方案2， 先对label做rolling_zscore---------------
    def norm_ret(x, window=rolling_w):  # 不再用L2 norm，恢复到之前的zscore，然后这里需要做的是给他增加一个周期

        # 注意这个函数是用在return上面的，log1p最小的数值是-1，用于return合适
        x = np.log1p(np.asarray(x))
        factors_data = pd.DataFrame(x, columns=['factor'])
        factors_data = factors_data.replace([np.inf, -np.inf, np.nan], 0.0)
        # factors_mean = factors_data.rolling(
        #     window=window, min_periods=1).mean()
        factors_std = factors_data.rolling(window=window, min_periods=1).std()
        factor_value = factors_data / factors_std
        factor_value = factor_value.replace([np.inf, -np.inf, np.nan], 0.0)
        # factor_value = factor_value.clip(-6, 6)
        return np.nan_to_num(factor_value).flatten()


    # Note - 先强行使用norm_ret看效果
    z['ret_rolling_zscore'] = norm_ret(z['return_f'])

    # 此时，所有的features和label，都用相同窗口做完了rolling处理，为了训练模型的准确性，可以开始删除掉还没有存满窗口的那些行了。去除前window行
    # 重要！！ 如果执行如下这句，会z与上面的ohlcva_df不一致，导致originalFeature.BaseFeature(ohlcva_df)初始化的ohlcva_df与做eval的feature_data不一致
    # z = z.iloc[window-1:]


    kept_index = z.index
    ohlcva_df = ohlcva_df.loc[kept_index]
    open_train = ohlcva_df['o'][(ohlcva_df['o'].index >= pd.to_datetime(start_date_train)) & (
            ohlcva_df['o'].index < pd.to_datetime(end_date_train))]
    open_test = ohlcva_df['o'][(ohlcva_df['o'].index >= pd.to_datetime(start_date_test)) & (
            ohlcva_df['o'].index <= pd.to_datetime(end_date_test))]
    close_train = ohlcva_df['c'][(ohlcva_df['c'].index >= pd.to_datetime(start_date_train)) & (
            ohlcva_df['c'].index < pd.to_datetime(end_date_train))]
    close_test = ohlcva_df['c'][(ohlcva_df['c'].index >= pd.to_datetime(start_date_test)) & (
            ohlcva_df['c'].index <= pd.to_datetime(end_date_test))]

    print('ret_rolling_zscore skew = {}'.format(
        z['ret_rolling_zscore'].skew()))
    print('ret_rolling_zscore kurtosis = {}'.format(
        z['ret_rolling_zscore'].kurtosis()))

    print('return skew = {}'.format(z['return_f'].skew()))
    print('return kurtosis = {}'.format(z['return_f'].kurtosis()))


    # 切分为train和test两个数据集，但是注意，test数据集其实带入了之前的数据的窗口, 是要特意这么做的。
    z_train = z[(z.index >= pd.to_datetime(start_date_train)) & (
        z.index < pd.to_datetime(end_date_train))]  # 只截取参数指定部分dataframe
    z_test = z[(z.index >= pd.to_datetime(start_date_test)) & (
        z.index <= pd.to_datetime(end_date_test))]
    # ------------<label 分析>-------------------------

    # 对于Label的分析的指导目标，是希望它能够接近正态分布，偏度，峰度接近于0
    # 方案1，先对log_return做clip，完全去除了outlier，再看偏度峰度，决定后续是否rolling_zscore.
    # 方案2，先对log_return做rolling_zscore,
    # (rolling窗口值是2000，暂时当做经验性的参数，取值的自由度来源于詹森不等式和大数定理，都是用数据算出来的)
    # 1. 参数和单点夏普的关系不明确，但是和几万个因子的夏普只和，他们的关系应该存在一定的凸性；
    # 2. 参数的设置应该在维持1的前提下兼顾大数定理；
    # 3. samples拆分成为几个class后样本量仍然符合大数定理；
    # 如上两种方案的对比，当前认为是应该第二种方式，应该是能够保留一部分outlier的信息，相对平衡的减轻outlier的影响



    if output_format == 'ndarry':
        y_dataset_train = z_train['ret_rolling_zscore'].values
        y_dataset_test = z_test['ret_rolling_zscore'].values
        ret_dataset_train = z_train['return_f'].values
        ret_dataset_test = z_test['return_f'].values
        # 重要！要删除掉包含未来信息的字段，ret，ret_rolling_zscore
        z_train.drop(['return_f', 'ret_rolling_zscore'], axis=1, inplace=True)
        z_test.drop(['return_f', 'ret_rolling_zscore'], axis=1, inplace=True)
        z.drop(['return_f', 'ret_rolling_zscore'], axis=1, inplace=True)

        X_all = np.where(np.isnan(z), 0, z)
        X_dataset_train = np.where(np.isnan(z_train), 0, z_train)
        X_dataset_test = np.where(np.isnan(z_test), 0, z_test)

    elif output_format == 'dataframe':
        y_dataset_train = z_train['ret_rolling_zscore']
        y_dataset_test = z_test['ret_rolling_zscore']
        ret_dataset_train = z_train['return_f']
        ret_dataset_test = z_test['return_f']
        # 重要！要删除掉包含未来信息的字段，ret，ret_rolling_zscore
        z_train.drop(['return_f', 'ret_rolling_zscore'], axis=1, inplace=True)
        z_test.drop(['return_f', 'ret_rolling_zscore'], axis=1, inplace=True)
        z.drop(['return_f', 'ret_rolling_zscore'], axis=1, inplace=True)

        X_all = z.fillna(0)
        X_dataset_train = z_train.fillna(0)
        X_dataset_test = z_test.fillna(0)

    else:
        print(
            'output_format of data_prepare should be "ndarry" or "dataframe", pls check it ')
        exit(1)

    feature_names = z_train.columns

    print('检查x all是不是等于 x train和y train相加，再检查trian和test以及close和open的形状是否一致')
    # X_all 专门是为做batch prediction的时候，要用X_all生成test集要用到的factor_df, 因为factor的计算需要之前一段window中的feature值
    print(f'检查X_all的形状 {X_all.shape}')
    print(f'检查x dataset train的形状 {X_dataset_train.shape}')
    print(f'检查y dataset train的形状 {y_dataset_train.shape}')
    print(f'检查x all是不是等于train和test相加 {len(X_all)},{len(X_dataset_test)+len(X_dataset_train)}')
    print(f'检查open train的形状 {open_train.shape}')
    print(f'检查close train的形状 {close_train.shape}')
    print(f'检查x dataset test的形状 {X_dataset_test.shape}')
    print(f'检查y dataset test的形状 {y_dataset_test.shape}')
    print(f'检查open test的形状 {open_test.shape}')
    print(f'检查close test的形状 {close_test.shape}')

    # X_all 专门是为做batch prediction的时候，要用X_all生成test集要用到的factor_df, 因为factor的计算需要之前一段window中的feature值
    return X_all, X_dataset_train, y_dataset_train,ret_dataset_train, X_dataset_test, y_dataset_test,ret_dataset_test, feature_names,open_train,open_test,close_train,close_test, z.index ,ohlcva_df


def data_prepare_coarse_grain_rolling(
        sym: str, 
        freq: str,  # 预测周期，例如 '2h' 表示预测未来2小时收益
        start_date_train: str, 
        end_date_train: str,
        start_date_test: str, 
        end_date_test: str,
        coarse_grain_period: str = '2h',  # 粗粒度特征桶周期
        feature_lookback_bars: int = 8,    # 特征回溯桶数（8个2h = 16小时）
        rolling_step: str = '10min',       # 滚动步长
        y_train_ret_period: int = 1,       # 预测周期（以coarse_grain为单位，1表示1个2h）
        rolling_w: int = 2000,
        output_format: str = 'ndarry',
        data_dir: str = '',
        read_frequency: str = '',
        timeframe: str = '',
        file_path: Optional[str] = None
    ):
    """
    粗粒度特征 + 细粒度滚动的数据准备方法（滑动窗口版本）
    
    核心思想：
    - 特征使用粗粒度周期（如2小时）聚合，减少噪声
    - 特征窗口使用固定时间长度（如8个2小时 = 16小时）
    - 预测起点以细粒度步长滚动（如15分钟），产生高频样本
    - **关键改进**：每个滚动时间点都独立计算其专属的滑动窗口特征，避免多个样本重复使用相同的粗粒度桶
    - 预测目标是未来N个粗粒度周期的收益（如未来2小时）
    
    参数说明：
    - sym: 交易对符号
    - freq: 用于兼容，实际预测周期由 y_train_ret_period * coarse_grain_period 决定
    - coarse_grain_period: 粗粒度特征桶周期，如 '2h', '1h', '30min'
    - feature_lookback_bars: 特征回溯的粗粒度桶数量（如8表示8个2h桶）
    - rolling_step: 滚动步长，如 '15min', '10min', '5min'
    - y_train_ret_period: 预测周期数（以coarse_grain_period为单位）
    
    示例场景（滑动窗口）：
    - coarse_grain_period='2h', feature_lookback_bars=8, rolling_step='15min'
    - 在9:00时刻：从原始数据提取 [前一天17:00, 9:00] 的数据，重采样为2h桶，计算特征，预测9:00-11:00收益
    - 在9:15时刻：从原始数据提取 [前一天17:15, 9:15] 的数据，重采样为2h桶，计算特征，预测9:15-11:15收益
    - 在9:30时刻：从原始数据提取 [前一天17:30, 9:30] 的数据，重采样为2h桶，计算特征，预测9:30-11:30收益
    
    优势：
    - 每个时间点的特征窗口都是独立的，避免了数据泄露和样本相关性问题
    - 滚动步长可以任意设置，不受粗粒度周期限制
    - 特征更加精细，更能反映实时市场状态
    
    返回与 data_prepare 相同的接口
    """
    
    print(f"\n{'='*60}")
    print(f"粗粒度特征 + 细粒度滚动数据准备（滑动窗口版本）")
    print(f"品种: {sym}")
    print(f"粗粒度周期: {coarse_grain_period}")
    print(f"特征窗口: {feature_lookback_bars} × {coarse_grain_period} = {feature_lookback_bars * pd.Timedelta(coarse_grain_period).total_seconds() / 3600:.1f}小时")
    print(f"滚动步长: {rolling_step}")
    print(f"预测周期: {y_train_ret_period} × {coarse_grain_period}")
    print(f"注意：每个时间点都会独立计算其滑动窗口特征，避免重复使用相同的粗粒度桶")
    print(f"{'='*60}\n")
    
    # ========== 第一步：读取原始数据（细粒度） ==========
    z_raw = data_load_v2(sym, data_dir=data_dir, start_date=start_date_train, end_date=end_date_test,
                         timeframe=timeframe, read_frequency=read_frequency, file_path=file_path)
    z_raw.index = pd.to_datetime(z_raw.index)
    
    # 扩展数据范围以容纳特征窗口
    feature_window_timedelta = pd.Timedelta(coarse_grain_period) * feature_lookback_bars
    extended_start = pd.to_datetime(start_date_train) - feature_window_timedelta - pd.Timedelta('1d')  # 多留1天buffer
    
    z_raw = z_raw[(z_raw.index >= extended_start) & (z_raw.index <= pd.to_datetime(end_date_test))]
    print(f"读取原始数据: {len(z_raw)} 行，时间范围 {z_raw.index.min()} 至 {z_raw.index.max()}")
    
    # ========== 第二步：生成粗粒度OHLCV桶（仅用于标签计算） ==========
    print(f"\n生成粗粒度OHLCV桶（周期={coarse_grain_period}，用于标签计算）...")
    coarse_bars = resample(z_raw, coarse_grain_period)
    print(f"粗粒度桶数量: {len(coarse_bars)}")
    
    # 注意：不再预先计算所有桶的特征，而是为每个滚动时间点动态计算滑动窗口特征
    
    # ========== 第三步：生成细粒度滚动时间网格 ==========
    print(f"\n生成细粒度滚动时间网格（步长={rolling_step}）...")
    
    # 从训练集开始到测试集结束，按rolling_step生成时间点
    grid_start = pd.to_datetime(start_date_train)
    grid_end = pd.to_datetime(end_date_test)
    
    # 生成时间网格
    fine_grain_timestamps = pd.date_range(start=grid_start, end=grid_end, freq=rolling_step)
    print(f"生成 {len(fine_grain_timestamps)} 个时间点")
    
    # ========== 第四步：为每个细粒度时间点提取滑动窗口特征和标签 ==========
    print(f"\n为每个时间点提取滑动窗口特征和标签...")
    print(f"注意：采用滑动窗口方案，每个时间点都独立计算特征")
    
    samples = []
    valid_count = 0
    skipped_count = 0
    
    coarse_period_td = pd.Timedelta(coarse_grain_period)
    prediction_horizon_td = coarse_period_td * y_train_ret_period
    
    for idx, t in enumerate(fine_grain_timestamps):
        if idx % 50 == 0:
            print(f"  处理进度: {idx}/{len(fine_grain_timestamps)} ({100*idx/len(fine_grain_timestamps):.1f}%)")
        
        # ========== 滑动窗口特征提取 ==========
        # 确定特征窗口：t时刻前的 feature_lookback_bars * coarse_grain_period 时间
        feature_window_start = t - feature_window_timedelta
        feature_window_end = t
        
        # 检查数据范围
        if feature_window_start < z_raw.index.min():
            skipped_count += 1
            continue
        
        if feature_window_end > z_raw.index.max():
            skipped_count += 1
            continue
        
        # 从原始数据中提取这个时间点专属的窗口数据
        window_raw_data = z_raw[(z_raw.index >= feature_window_start) & 
                                (z_raw.index < feature_window_end)]
        
        if len(window_raw_data) < 10:  # 至少需要一些数据点
            skipped_count += 1
            continue
        
        # 对窗口数据进行粗粒度重采样
        window_coarse_bars = resample(window_raw_data, coarse_grain_period)
        
        # 检查是否有足够的粗粒度桶
        if len(window_coarse_bars) < feature_lookback_bars * 0.5:  # 容忍50%缺失
            skipped_count += 1
            continue
        
        # 为这个窗口的粗粒度桶提取特征
        try:
            base_feature = originalFeature.BaseFeature(window_coarse_bars.copy())
            window_features_df = base_feature.init_feature_df
        except Exception as e:
            # 特征提取失败，跳过这个样本
            skipped_count += 1
            continue
        
        # 对窗口内的特征进行聚合（多种统计量）
        feature_dict = {}
        for col in window_features_df.columns:
            if col in ['c', 'v', 'o', 'h', 'l', 'vol']:
                continue
            if pd.api.types.is_numeric_dtype(window_features_df[col]):
                col_data = window_features_df[col]
                n = len(col_data)
                
                # 基础统计量
                feature_dict[f'{col}_mean'] = col_data.mean()
                feature_dict[f'{col}_std'] = col_data.std()
                feature_dict[f'{col}_max'] = col_data.max()
                feature_dict[f'{col}_min'] = col_data.min()
                feature_dict[f'{col}_last'] = col_data.iloc[-1] if n > 0 else 0
                
                # 高阶统计量
                feature_dict[f'{col}_skew'] = col_data.skew() if n > 2 else 0  # 偏度（需要至少3个点）
                feature_dict[f'{col}_kurt'] = col_data.kurtosis() if n > 3 else 0  # 峰度（需要至少4个点）
                
                # 分位数（更稳健）
                feature_dict[f'{col}_median'] = col_data.median()  # 中位数
                feature_dict[f'{col}_q25'] = col_data.quantile(0.25) if n > 0 else 0  # 25%分位数
                feature_dict[f'{col}_q75'] = col_data.quantile(0.75) if n > 0 else 0  # 75%分位数
        
        # 计算t时刻的价格（用于标签计算）
        # 找到t时刻最近的粗粒度桶
        closest_bar_idx = coarse_bars.index.searchsorted(t, side='right') - 1
        if closest_bar_idx < 0 or closest_bar_idx >= len(coarse_bars):
            skipped_count += 1
            continue
        
        t_price = coarse_bars.iloc[closest_bar_idx]['c']
        
        # 计算t+prediction_horizon时刻的价格
        t_future = t + prediction_horizon_td
        future_bar_idx = coarse_bars.index.searchsorted(t_future, side='right') - 1
        if future_bar_idx < 0 or future_bar_idx >= len(coarse_bars):
            skipped_count += 1
            continue
        
        t_future_price = coarse_bars.iloc[future_bar_idx]['c']
        
        # 计算对数收益
        log_return = np.log(t_future_price / t_price)
        
        # 记录样本
        sample = {
            'timestamp': t,
            't_price': t_price,
            't_future_price': t_future_price,
            'return_f': log_return,
            **feature_dict
        }
        samples.append(sample)
        valid_count += 1
    
    print(f"\n✓ 生成样本完成: 有效 {valid_count} 个，跳过 {skipped_count} 个")
    
    # ========== 第六步：构建DataFrame并处理 ==========
    df_samples = pd.DataFrame(samples)
    df_samples.set_index('timestamp', inplace=True)
    
    print(f"样本时间范围: {df_samples.index.min()} 至 {df_samples.index.max()}")
    print(f"样本数量: {len(df_samples)}")
    print(f"特征维度: {len([c for c in df_samples.columns if c not in ['t_price', 't_future_price', 'return_f']])}")
    
    # 应用滚动标准化到标签
    def norm_ret(x, window=rolling_w):
        x = np.log1p(np.asarray(x))
        factors_data = pd.DataFrame(x, columns=['factor'])
        factors_data = factors_data.replace([np.inf, -np.inf, np.nan], 0.0)
        factors_std = factors_data.rolling(window=window, min_periods=1).std()
        factor_value = factors_data / factors_std
        factor_value = factor_value.replace([np.inf, -np.inf, np.nan], 0.0)
        return np.nan_to_num(factor_value).flatten()
    
    df_samples['ret_rolling_zscore'] = norm_ret(df_samples['return_f'].values)
    
    print(f"\n标签统计:")
    print(f"return_f - 偏度: {df_samples['return_f'].skew():.4f}, 峰度: {df_samples['return_f'].kurtosis():.4f}")
    print(f"ret_rolling_zscore - 偏度: {df_samples['ret_rolling_zscore'].skew():.4f}, 峰度: {df_samples['ret_rolling_zscore'].kurtosis():.4f}")
    
    # ========== 第七步：分割训练集和测试集 ==========
    train_mask = (df_samples.index >= pd.to_datetime(start_date_train)) & \
                 (df_samples.index < pd.to_datetime(end_date_train))
    test_mask = (df_samples.index >= pd.to_datetime(start_date_test)) & \
                (df_samples.index <= pd.to_datetime(end_date_test))
    
    # 提取特征列
    feature_cols = [c for c in df_samples.columns if c not in ['t_price', 't_future_price', 'return_f', 'ret_rolling_zscore']]
    
    X_all = df_samples[feature_cols].fillna(0)
    X_train = X_all[train_mask]
    X_test = X_all[test_mask]
    
    y_train = df_samples.loc[train_mask, 'ret_rolling_zscore'].fillna(0).values
    y_test = df_samples.loc[test_mask, 'ret_rolling_zscore'].fillna(0).values
    ret_train = df_samples.loc[train_mask, 'return_f'].fillna(0).values
    ret_test = df_samples.loc[test_mask, 'return_f'].fillna(0).values
    
    # 价格数据（用于回测）
    open_train = df_samples.loc[train_mask, 't_price']
    close_train = df_samples.loc[train_mask, 't_price']  # 简化：开盘价=当前价
    open_test = df_samples.loc[test_mask, 't_price']
    close_test = df_samples.loc[test_mask, 't_price']
    
    feature_names = feature_cols
    
    # 格式转换
    if output_format == 'ndarry':
        X_all = X_all.values
        X_train = X_train.values
        X_test = X_test.values
    elif output_format == 'dataframe':
        pass  # 保持DataFrame格式
    else:
        raise ValueError(f"output_format 应为 'ndarry' 或 'dataframe'，当前为 {output_format}")
    
    print(f"\n{'='*60}")
    print(f"数据分割完成:")
    print(f"  训练集: {len(X_train)} 样本")
    print(f"  测试集: {len(X_test)} 样本")
    print(f"  特征数: {len(feature_names)}")
    print(f"{'='*60}\n")
    
    # 构建 ohlc DataFrame（用于后续分析，如IC decay）
    # 包含每个样本的价格信息，与 X_all 对齐
    ohlc_aligned = pd.DataFrame({
        'c': df_samples['t_price'],  # 当前价格
        'close': df_samples['t_price']  # 兼容性别名
    }, index=df_samples.index)
    
    # 返回接口与 data_prepare 保持一致
    return (X_all, X_train, y_train, ret_train, X_test, y_test, ret_test,
            feature_names, open_train, open_test, close_train, close_test,
            df_samples.index, ohlc_aligned)


def data_prepare_micro(sym: str, freq: str,
                       start_date_train: str, end_date_train: str,
                       start_date_test: str, end_date_test: str,
                       y_train_ret_period: int = 1, rolling_w: int = 2000,
                       output_format: str = 'ndarry', data_dir: str = '',
                       read_frequency: str = '', timeframe: str = '',
                       use_feature_extractors: bool = False,
                       trades_dir: str = '',
                       bar_builder: str = 'time',
                       dollar_threshold: float = 1e6,
                       feature_window_bars: int = 10,
                       trades_load_mode: str = 'auto',
                       prefer_trades_feather: bool = True,
                       daily_data_template: str = '',
                       monthly_data_template: str = '',
                       active_family: str = '',
                       feature_family_include: list = None,
                       feature_family_exclude: list = None,
                       file_path: Optional[str] = None,
                       kline_file_path: Optional[str] = None):
    """
    微结构版数据准备：
    
    数据流程：
    1. 读取 tick 级别交易数据（trades）
    2. 从 trades 构建 bars（TimeBar 或 DollarBar）得到 OHLCV
    3. 从 bars 的 OHLCV 生成标签（return_f）
    4. 提取微观结构特征（基于 trades 和 bars）
    
    参数说明：
    - file_path: tick交易数据文件路径（优先级最高）
    - kline_file_path: K线数据文件路径（用于标签生成的备用方案，当无tick数据时）
    - daily_data_template/monthly_data_template: tick数据模板路径
    
    返回签名保持与 data_prepare 完全一致，便于 GPAnalyzer 直接替换使用
    """
    
    # ========== 第一步：读取 Tick 交易数据 ==========
    trades_df = None
    
    # 优先级1: 直接指定的 tick 文件路径
    if file_path:
        print(f"\n{'='*60}")
        print(f"从指定路径读取 Tick 数据: {file_path}")
        print(f"{'='*60}\n")
        trades_df = _read_tick_data_file(file_path)
    
    # 优先级2: 使用 TradingPipeline 从模板读取
    elif daily_data_template or monthly_data_template:
        try:
            tp_config = {
                'data': {
                    'load_mode': trades_load_mode or 'auto',
                    'prefer_feather': bool(prefer_trades_feather),
                }
            }
            tp = TradingPipeline(tp_config)
            
            # 规范化日期为 YYYY-MM-DD
            def _normalize_date(s: str) -> str:
                if len(str(s)) == 7 and '-' in s:  # 'YYYY-MM'
                    return f"{s}-01"
                return str(s)
            
            start_norm = _normalize_date(start_date_train)
            end_norm = _normalize_date(end_date_test)
            
            trades_df = tp.load_data(
                trades_data=None,
                date_range=(start_norm, end_norm),
                daily_data_template=daily_data_template if daily_data_template else None,
                monthly_data_template=monthly_data_template if monthly_data_template else None,
            )
            print(f"✓ 成功从模板读取 {len(trades_df):,} 条 Tick 数据")
        except Exception as e:
            print(f"⚠️  从模板读取 Tick 数据失败: {e}")
    
    # 如果没有 tick 数据，回退到 K线数据
    if trades_df is None or trades_df.empty:
        print(f"\n{'='*60}")
        print("⚠️  未找到 Tick 数据，回退到 K线数据模式")
        print(f"{'='*60}\n")
        
        # 从 K线数据读取
        z_raw = data_load_v2(sym, data_dir=data_dir, start_date=start_date_train, end_date=end_date_test,
                            timeframe=timeframe, read_frequency=read_frequency, file_path=kline_file_path)
        z_raw.index = pd.to_datetime(z_raw.index)
        z_raw = z_raw[(z_raw.index >= pd.to_datetime(start_date_train)) & (z_raw.index <= pd.to_datetime(end_date_test))]
        ohlcva_df = resample(z_raw, freq)
        bars = None  # 标记没有构建 bars
    else:
        # ========== 第二步：从 Tick 数据构建 Bars ==========
        print(f"\n{'='*60}")
        print(f"从 Tick 数据构建 Bars (builder={bar_builder}, freq={freq})")
        print(f"{'='*60}\n")
        
        if str(bar_builder).lower() == 'dollar':
            db = DollarBarBuilder(dollar_threshold=float(dollar_threshold))
            bars = db.process(trades_df)
            print(f"✓ 构建了 {len(bars)} 个 Dollar Bars (threshold={dollar_threshold:,.0f})")
        else:
            try:
                tb = TimeBarBuilder(freq=freq)
                bars = tb.process(trades_df)
            except Exception as e:
                print(f"⚠️  构建 Time Bars 失败: {e}")
            print(f"✓ 构建了 {len(bars)} 个 Time Bars (freq={freq})")
        
        # 过滤时间范围
        # bars = bars[(pd.to_datetime(bars['end_time']) >= pd.to_datetime(start_date_train)) &
        #             (pd.to_datetime(bars['end_time']) <= pd.to_datetime(end_date_test))]
        
        # 设置时间索引
        bars.index = pd.to_datetime(bars['end_time'])
        ohlcva_df = None  # 标记使用 bars
        
        print(f"✓ 构建了 {len(bars)} 个 Bars")
        print(f"时间范围: {bars.index.min()} 至 {bars.index.max()}\n")

    # ========== 第三步：统一数据接口（bars 或 ohlcva_df）==========
    # 如果有 bars，直接使用 bars；否则使用 ohlcva_df（K线回退模式）
    if bars is not None:
        data_df = bars
        close = data_df['close'].astype(float)
    else:
        data_df = ohlcva_df
        close = data_df['c'].astype(float)
    
    eps = 1e-12

    # 家族意图解析（控制“只计算启用家族”）
    def _to_list(v):
        if v is None:
            return []
        return v if isinstance(v, list) else [v]

    include_keys = _to_list(feature_family_include)
    exclude_keys = _to_list(feature_family_exclude)

    fam = str(active_family).strip().lower() if isinstance(active_family, str) else ''
    family_map = {
        'momentum': ['bar_logret', 'bar_abs_logret', 'micro_dp_short', 'micro_dp_zscore'],
        'liquidity': ['amihud', 'kyle', 'hasbrouck', 'cs_spread', 'impact_roll', 'bar_amihud'],
        'impact': ['amihud', 'kyle', 'hasbrouck', 'impact_'],
    }

    enabled_families: set = set()
    if fam in family_map:
        enabled_families.add(fam)
    elif include_keys:
        for fam_name, keys in family_map.items():
            if any(any(k in key or key in k for key in keys) for k in include_keys):
                enabled_families.add(fam_name)

    # 微结构代理特征（bar级）- 暂时为空DataFrame，后续由特征提取器填充
    f = pd.DataFrame(index=data_df.index)

    # ========== 第四步：提取微观结构特征（如果有 tick 数据和 bars）==========
    extracted_df = None
    if use_feature_extractors and trades_df is not None and bars is not None and not bars.empty:
        print(f"\n{'='*60}")
        print("开始提取微观结构特征")
        print(f"{'='*60}\n")
        
        try:
            # 构建交易上下文
            tp = TradesProcessor()
            ctx = tp.build_context(trades_df)
            
            # 以 trading_pipeline 的窗口方式提取：对每个目标bar i，用 [i-feature_window_bars, i-1] 作为特征窗口
            bars = bars.reset_index(drop=True)
            bars['end_time'] = pd.to_datetime(bars['end_time'])
            
            # 构造仅启用家族的提取器配置
            extractor_cfg: Dict[str, Any] = {}
            if enabled_families:
                # 先全部禁用，再按需启用
                for k in ['basic','volatility','momentum','orderflow','impact','tail','path_shape','bucketed_flow']:
                    extractor_cfg[k] = {'enabled': False}
                for fam_name in enabled_families:
                    if fam_name in extractor_cfg:
                        extractor_cfg[fam_name] = {'enabled': True}
            
            extractor = MicrostructureFeatureExtractor(extractor_cfg if extractor_cfg else None)
            
            feat_rows = []
            feat_index = []
            total_bars = len(bars)
            print(f"开始逐 bar 提取特征，共 {total_bars} 个 bars...")
            
            for i in range(total_bars):
                if i % 100 == 0 and i > 0:
                    print(f"  进度: {i}/{total_bars} ({i/total_bars*100:.1f}%)")
                
                start_idx = i - int(feature_window_bars)
                end_idx = i - 1
                if start_idx < 0 or end_idx < 0 or end_idx < start_idx:
                    continue
                
                st = pd.to_datetime(bars.loc[start_idx, 'start_time'])
                et = pd.to_datetime(bars.loc[end_idx, 'end_time'])
                
                try:
                    feats = extractor.extract_from_context(
                        ctx=ctx,
                        start_ts=st,
                        end_ts=et,
                        bars=bars,
                        bar_window_start_idx=start_idx,
                        bar_window_end_idx=end_idx,
                    )
                except Exception:
                    feats = {}
                
                feat_rows.append(feats)
                feat_index.append(pd.to_datetime(bars.loc[i, 'end_time']))
            
            print(f"✓ 特征提取完成，共提取 {len(feat_rows)} 个样本")
            
            extracted_df = pd.DataFrame(feat_rows, index=pd.to_datetime(feat_index))
            
            # 缺失与极值处理
            print(f"处理缺失值和极值...")
            extracted_df = extracted_df.replace([np.inf, -np.inf], np.nan)
            try:
                lower2 = extracted_df.quantile(0.01)
                upper2 = extracted_df.quantile(0.99)
                extracted_df = extracted_df.clip(lower=lower2, upper=upper2, axis=1)
            except Exception:
                pass
            extracted_df = extracted_df.fillna(0.0)
            print(f"✓ 提取了 {len(extracted_df.columns)} 个微观结构特征\n")
            
        except Exception as e:
            print(f"⚠️  基于 features/ 特征提取失败，将仅使用 bar 级代理特征：{e}\n")

    # ========== 第五步：生成标签 ==========
    z_lab = pd.DataFrame(index=data_df.index)
    z_lab['return_f'] = np.log(close).diff(y_train_ret_period).shift(-y_train_ret_period)
    z_lab['return_f'] = z_lab['return_f'].fillna(0)

    def _norm_ret(x, window=rolling_w):
        x = np.log1p(np.asarray(x))
        x = np.nan_to_num(x)
        std = pd.Series(x).rolling(window=window, min_periods=1).std().replace(0, np.nan)
        val = pd.Series(x) / std
        return np.nan_to_num(val).values

    z_lab['ret_rolling_zscore'] = _norm_ret(z_lab['return_f'])

    # ========== 第六步：切分训练集和测试集 ==========
    # 根据数据类型选择列名
    if bars is not None:
        open_col, close_col = 'open', 'close'
    else:
        open_col, close_col = 'o', 'c'
    
    open_train = data_df[open_col][(data_df[open_col].index >= pd.to_datetime(start_date_train)) & (data_df[open_col].index < pd.to_datetime(end_date_train))]
    open_test = data_df[open_col][(data_df[open_col].index >= pd.to_datetime(start_date_test)) & (data_df[open_col].index <= pd.to_datetime(end_date_test))]
    close_train = data_df[close_col][(data_df[close_col].index >= pd.to_datetime(start_date_train)) & (data_df[close_col].index < pd.to_datetime(end_date_train))]
    close_test = data_df[close_col][(data_df[close_col].index >= pd.to_datetime(start_date_test)) & (data_df[close_col].index <= pd.to_datetime(end_date_test))]

    # 切分 train/test
    z_train = pd.concat([f, z_lab], axis=1)
    z_test = z_train.copy()
    z_train = z_train[(z_train.index >= pd.to_datetime(start_date_train)) & (z_train.index < pd.to_datetime(end_date_train))]
    z_test = z_test[(z_test.index >= pd.to_datetime(start_date_test)) & (z_test.index <= pd.to_datetime(end_date_test))]

    # y/ret
    y_dataset_train = z_train['ret_rolling_zscore'].values
    y_dataset_test = z_test['ret_rolling_zscore'].values
    ret_dataset_train = z_train['return_f'].values
    ret_dataset_test = z_test['return_f'].values

    # 删除包含未来信息的列
    z_train = z_train.drop(['return_f', 'ret_rolling_zscore'], axis=1)
    z_test = z_test.drop(['return_f', 'ret_rolling_zscore'], axis=1)
    f_all = pd.concat([f], axis=1)

    # X_all/X_train/X_test
    if output_format == 'ndarry':
        X_all = np.where(np.isnan(f_all), 0, f_all)
        X_dataset_train = np.where(np.isnan(z_train), 0, z_train)
        X_dataset_test = np.where(np.isnan(z_test), 0, z_test)
    elif output_format == 'dataframe':
        X_all = f_all.fillna(0)
        X_dataset_train = z_train.fillna(0)
        X_dataset_test = z_test.fillna(0)
    else:
        print('output_format of data_prepare_micro should be "ndarry" or "dataframe"')
        exit(1)

    feature_names = z_train.columns

    # 为了兼容性，如果使用 bars，需要创建 ohlcva_df 格式的返回值
    if bars is not None:
        # 创建兼容格式的 ohlcva_df
        ohlcva_df = data_df[['open', 'high', 'low', 'close', 'volume', 'dollar_value', 'trades']].copy()
        ohlcva_df.columns = ['o', 'h', 'l', 'c', 'vol', 'vol_ccy', 'trades']
        for col in ['oi', 'oi_ccy', 'toptrader_count_lsr', 'toptrader_oi_lsr', 'count_lsr', 'taker_vol_lsr']:
            ohlcva_df[col] = 0

    return (X_all, X_dataset_train, y_dataset_train, ret_dataset_train,
            X_dataset_test, y_dataset_test, ret_dataset_test,
            feature_names, open_train, open_test, close_train, close_test,
            z_lab.index, ohlcva_df)


def _filter_features_by_type_and_keywords(feature_cols: List[str], 
                                        feature_types: Optional[List[str]] = None,
                                        feature_keywords: Optional[List[str]] = None) -> List[str]:
    """
    根据特征类型和关键词筛选特征列
    
    参数:
    - feature_cols: 所有特征列名列表
    - feature_types: 特征类型筛选，如['momentum', 'tail', 'orderflow', 'impact', 'volatility', 'basic', 'path_shape']
    - feature_keywords: 关键词筛选，如['lambda', 'large', 'signed', 'imbalance']等
    
    返回:
    - 筛选后的特征列名列表
    """
    if not feature_types and not feature_keywords:
        return feature_cols
    
    # 定义特征类型映射
    feature_type_mapping = {
        'momentum': ['mr_', 'momentum', 'reversion', 'dp_short', 'dp_zscore'],
        'tail': ['large_', 'q90', 'q95', 'q99', 'sweep', 'burstiness'],
        'orderflow': ['ofi_', 'gof_', 'signed_', 'imbalance', 'flow'],
        'impact': ['lambda', 'amihud', 'kyle', 'hasbrouck', 'impact'],
        'volatility': ['rv', 'bpv', 'jump', 'volatility', 'std'],
        'basic': ['int_trade_', 'vwap', 'volume', 'dollar', 'intensity'],
        'path_shape': ['vwap_deviation', 'corr_', 'path', 'shape', 'deviation']
    }
    
    filtered_cols = []
    
    for col in feature_cols:
        col_lower = col.lower()

        # if col_lower != 'bar_gof_by_count':
        #     continue
        
        include_col = True
        
        # 特征类型筛选
        if feature_types:
            type_match = False
            for feature_type in feature_types:
                if feature_type in feature_type_mapping:
                    keywords = feature_type_mapping[feature_type]
                    if any(keyword in col_lower for keyword in keywords):
                        type_match = True
                        break
            if not type_match:
                include_col = False
        
        # 关键词筛选
        if include_col and feature_keywords:
            keyword_match = any(keyword.lower() in col_lower for keyword in feature_keywords)
            if not keyword_match:
                include_col = False
        
        if include_col and not col.startswith('cs_'):
            filtered_cols.append(col)
    
    return filtered_cols


def data_prepare_rolling(sym: str, freq: str,
                        start_date_train: str, end_date_train: str,
                        start_date_test: str, end_date_test: str,
                        y_train_ret_period: int = 1, rolling_w: int = 2000,
                        feature_window_bars: int = 10,
                        output_format: str = 'ndarry', data_dir: str = '',
                        read_frequency: str = '', timeframe: str = '',
                        file_path: Optional[str] = None,
                        rolling_windows: Optional[List[int]] = None,
                        use_rolling_aggregator: bool = True,
                        feature_types: Optional[List[str]] = None,
                        feature_keywords: Optional[List[str]] = None):
    """
    滚动统计版数据准备：
    
    数据流程：
    1. 读取已聚合的bar级数据（包含OHLCV和微观结构因子）
    2. 从bar数据生成标签（return_f）
    3. 使用RollingAggregator对bar级特征进行滚动统计
    4. 返回与data_prepare_micro完全一致的接口
    
    参数说明：
    - file_path: 已聚合的bar数据文件路径（包含OHLCV和微观结构因子）
    - rolling_windows: 滚动窗口列表，如[5, 10, 20]表示5bar、10bar、20bar窗口
    - use_rolling_aggregator: 是否使用RollingAggregator进行滚动统计
    - feature_types: 特征类型筛选，如['momentum', 'tail', 'orderflow', 'impact', 'volatility', 'basic', 'path_shape']
    - feature_keywords: 关键词筛选，如['lambda', 'large', 'signed', 'imbalance']等
    
    返回签名保持与data_prepare_micro完全一致，便于GPAnalyzer直接替换使用
    """
    
    print(f"\n{'='*60}")
    print(f"滚动统计版数据准备: {sym}")
    print(f"时间范围: {start_date_train} 至 {end_date_test}")
    print(f"滚动窗口: {rolling_windows or [rolling_w]}")
    if feature_types:
        print(f"特征类型筛选: {feature_types}")
    if feature_keywords:
        print(f"关键词筛选: {feature_keywords}")
    print(f"{'='*60}\n")
    
    # ========== 第一步：读取已聚合的Bar数据 ==========
    if file_path:
        print(f"从指定路径读取Bar数据: {file_path}")
        try:
            if file_path.endswith('.feather'):
                bars_df = pd.read_feather(file_path)
            elif file_path.endswith('.csv'):
                bars_df = pd.read_csv(file_path)
            elif file_path.endswith('.parquet'):
                bars_df = pd.read_parquet(file_path)
            else:
                # 尝试自动检测格式
                bars_df = pd.read_feather(file_path)
        except Exception as e:
            print(f"读取文件失败: {e}")
            raise
    else:
        # 回退到传统K线数据读取
        print("未指定file_path，回退到传统K线数据读取")
        z_raw = data_load_v2(sym, data_dir=data_dir, start_date=start_date_train, end_date=end_date_test,
                            timeframe=timeframe, read_frequency=read_frequency)
        z_raw.index = pd.to_datetime(z_raw.index)
        z_raw = z_raw[(z_raw.index >= pd.to_datetime(start_date_train)) & (z_raw.index <= pd.to_datetime(end_date_test))]
        bars_df = resample(z_raw, freq)
        # 重命名列以匹配标准格式
        if 'o' in bars_df.columns:
            bars_df = bars_df.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'vol': 'volume'})
    
    # 确保时间索引正确
    if 'timestamp' in bars_df.columns:
        bars_df.index = pd.to_datetime(bars_df['timestamp'])
    elif 'end_time' in bars_df.columns:
        bars_df.index = pd.to_datetime(bars_df['end_time'])
    elif 'start_time' in bars_df.columns:
        bars_df.index = pd.to_datetime(bars_df['start_time'])
    else:
        bars_df.index = pd.to_datetime(bars_df.index)
    
    # 过滤时间范围
    bars_df = bars_df[(bars_df.index >= pd.to_datetime(start_date_train)) & 
                     (bars_df.index <= pd.to_datetime(end_date_test))]
    
    print(f"✓ 读取了 {len(bars_df)} 个Bar数据")
    print(f"时间范围: {bars_df.index.min()} 至 {bars_df.index.max()}")
    print(f"数据列: {list(bars_df.columns)}")
    
    # ========== 第二步：生成标签 ==========
    print(f"\n开始生成标签 (ret_period={y_train_ret_period})")
    
    # 确保有close列
    if 'close' not in bars_df.columns and 'c' in bars_df.columns:
        bars_df['close'] = bars_df['c']
    elif 'close' not in bars_df.columns:
        raise ValueError("数据中必须包含close或c列")
    
    close = bars_df['close'].astype(float)
    
    # 生成收益率标签
    bars_df['return_f'] = np.log(close).diff(y_train_ret_period).shift(-y_train_ret_period)
    bars_df['return_f'] = bars_df['return_f'].fillna(0)
    bars_df['r'] = np.log(close).diff()
    bars_df['r'] = bars_df['r'].fillna(0)
    
    # 应用滚动标准化（与data_prepare保持一致）
    def norm_ret(x, window=rolling_w):
        x = np.log1p(np.asarray(x))
        factors_data = pd.DataFrame(x, columns=['factor'])
        factors_data = factors_data.replace([np.inf, -np.inf, np.nan], 0.0)
        factors_std = factors_data.rolling(window=window, min_periods=1).std()
        factor_value = factors_data / factors_std
        factor_value = factor_value.replace([np.inf, -np.inf, np.nan], 0.0)
        return np.nan_to_num(factor_value).flatten()
    
    bars_df['ret_rolling_zscore'] = norm_ret(bars_df['return_f'])
    
    print(f"✓ 标签生成完成")
    print(f"return_f skew = {bars_df['return_f'].skew():.4f}")
    print(f"return_f kurtosis = {bars_df['return_f'].kurtosis():.4f}")
    print(f"ret_rolling_zscore skew = {bars_df['ret_rolling_zscore'].skew():.4f}")
    print(f"ret_rolling_zscore kurtosis = {bars_df['ret_rolling_zscore'].kurtosis():.4f}")
    
    # ========== 第三步：准备特征数据 ==========
    print(f"\n开始准备特征数据")
    
    # 识别微观结构特征列（排除OHLCV和标签列）
    exclude_cols = {'open', 'high', 'low', 'close', 'volume', 'vol', 'o', 'h', 'l', 'c', 
                   'return_f', 'r', 'ret_rolling_zscore', 'timestamp', 'start_time', 'end_time'}
    
    # 自动识别所有可能的特征列
    all_feature_cols = [col for col in bars_df.columns 
                       if col not in exclude_cols and 
                       pd.api.types.is_numeric_dtype(bars_df[col])]
    
    print(f"识别到 {len(all_feature_cols)} 个原始特征列")
    
    # ========== 特征类型筛选 ==========
    feature_cols = _filter_features_by_type_and_keywords(
        all_feature_cols, feature_types, feature_keywords
    )
    
    print(f"筛选后保留 {len(feature_cols)} 个特征列: {feature_cols[:10]}{'...' if len(feature_cols) > 10 else ''}")
    
    # 创建特征DataFrame
    feature_df = bars_df[feature_cols].copy()
    
    # ========== 第四步：滚动统计特征提取 ==========
    if use_rolling_aggregator and feature_cols:
        print(f"\n开始滚动统计特征提取")
        
        # 导入RollingAggregator
        from features.rolling_aggregator import RollingAggregator
        
        # 设置滚动窗口
        windows = [feature_window_bars]
        print(f"使用滚动窗口: {windows}")
        
        # 创建RollingAggregator实例
        rolling_agg = RollingAggregator(windows=windows)
        
        # 为每个特征添加bar_前缀（模拟bar级特征）
        bar_feature_df = feature_df.copy()
        # bar_feature_df.columns = [f'bar_{col}' for col in bar_feature_df.columns]
        
        # 筛选出带bar_前缀的列
        bar_feature_df = bar_feature_df[[col for col in bar_feature_df.columns if col.startswith('bar_')]]
        
        # 收集所有滚动统计特征
        all_rolling_features = []
        
        for window in windows:
            print(f"  处理窗口 {window}...")
            window_features = []
            
            for i in range(len(bar_feature_df)):
                if i < window:
                    # 窗口不足，跳过
                    window_features.append({})
                    continue
                
                # 提取滚动统计特征
                rolling_stats = rolling_agg.extract_rolling_statistics(
                    bar_feature_df, window=window, current_idx=i
                )
                window_features.append(rolling_stats)
            
            # 转换为DataFrame
            window_df = pd.DataFrame(window_features, index=bar_feature_df.index)
            all_rolling_features.append(window_df)
        
        # 合并所有窗口的特征
        if all_rolling_features:
            final_feature_df = pd.concat(all_rolling_features, axis=1)
            print(f"✓ 生成了 {len(final_feature_df.columns)} 个滚动统计特征")
        else:
            final_feature_df = pd.DataFrame(index=feature_df.index)
            print("⚠️  未生成任何滚动统计特征")
    else:
        # 不使用滚动统计，直接使用原始特征
        print("使用原始特征（未进行滚动统计）")
        final_feature_df = feature_df
    
    # ========== 第五步：数据分割和格式化 ==========
    print(f"\n开始数据分割和格式化")
    
    # 确保索引一致
    final_feature_df = final_feature_df.loc[bars_df.index]
    
    # 分割训练集和测试集
    train_mask = (final_feature_df.index >= pd.to_datetime(start_date_train)) & \
                 (final_feature_df.index < pd.to_datetime(end_date_train))
    test_mask = (final_feature_df.index >= pd.to_datetime(start_date_test)) & \
                (final_feature_df.index <= pd.to_datetime(end_date_test))
    
    X_train = final_feature_df[train_mask].fillna(0)
    X_test = final_feature_df[test_mask].fillna(0)
    y_train = bars_df.loc[train_mask, 'ret_rolling_zscore'].fillna(0).values
    y_test = bars_df.loc[test_mask, 'ret_rolling_zscore'].fillna(0).values
    ret_train = bars_df.loc[train_mask, 'return_f'].fillna(0).values
    ret_test = bars_df.loc[test_mask, 'return_f'].fillna(0).values
    
    # OHLC数据
    ohlc_cols = ['open', 'high', 'low', 'close']
    if all(col in bars_df.columns for col in ohlc_cols):
        open_train = bars_df.loc[train_mask, 'open'].fillna(0)
        open_test = bars_df.loc[test_mask, 'open'].fillna(0)
        close_train = bars_df.loc[train_mask, 'close'].fillna(0)
        close_test = bars_df.loc[test_mask, 'close'].fillna(0)
    else:
        # 回退到c列
        open_train = bars_df.loc[train_mask, 'close'].fillna(0)  # 用close作为open的近似
        open_test = bars_df.loc[test_mask, 'close'].fillna(0)
        close_train = bars_df.loc[train_mask, 'close'].fillna(0)
        close_test = bars_df.loc[test_mask, 'close'].fillna(0)
    
    # 特征名称
    feature_names = list(final_feature_df.columns)
    
    # 完整数据集（用于某些分析）- 处理NaN值
    X_all = final_feature_df.fillna(0)
    
    print(f"✓ 数据分割完成")
    print(f"训练集: {len(X_train)} 样本, {len(feature_names)} 特征")
    print(f"测试集: {len(X_test)} 样本, {len(feature_names)} 特征")
    
    # ========== 第六步：返回结果 ==========
    return (X_all, X_train, y_train, ret_train, X_test, y_test, ret_test,
            feature_names, open_train, open_test, close_train, close_test,
            final_feature_df.index, bars_df)


def compute_transformed_series(column):
    """
    输入的是一个dataframe的一列，series.
    计算得到如下几个ndarry-

    1. log_return: 取log return。
    2. log_log_return：对log_return再做一次log.
    3. boxcox_transformed: 使用Box-Cox变换。
    4. yeo_johnson_transformed: 使用Yeo-Johnson变换。
    5. winsorized_log_return: 对log_return进行Winsorizing。
    6. scaled_log_return: 对log_return进行RobustScaler缩放。

    plot一个直方图，上面4种颜色显示如上四类数值各自的直方分布图.
    并且在图上画出如上4个序列各自的skewness和kurtosis
    -------

    """
    log_return = (np.log(column).diff(1).fillna(0)*1).shift(-1)
    log_return = np.where(np.isnan(log_return), 0, log_return)

    # ---------尝试对log return做滚动标准化--------
    log_return = _rolling_zscore(log_return, 300)

    # 计算 log_log_return
    log_log_return = np.log(log_return + 1)
    # log_log_return_2 = np.log(np.log(column)).diff().fillna(0).shift(-1)

    # 平移数据使其为正值
    log_return_shifted = log_return - np.min(log_return) + 1
    # 应用 Box-Cox 变换
    boxcox_transformed, _ = boxcox(log_return_shifted)

    # 应用 Yeo-Johnson 变换
    yeo_johnson_transformed, _ = yeojohnson(log_return)

    # 应用 Winsorizing
    winsorized_log_return = mstats.winsorize(log_return, limits=[0.05, 0.05])

    # # 应用 RobustScaler
    # scaler = RobustScaler()
    # scaled_log_return = scaler.fit_transform(log_return).flatten()

    # 绘制直方图
    plt.figure(figsize=(12, 6))

    # 绘制 log return 的直方图
    plt.hist(log_return, bins=160, alpha=0.3, color='blue', label='Log Return')

    # 绘制 log_log_return 的直方图
    plt.hist(log_log_return, bins=160, alpha=0.3,
             color='orange', label='Log Log Return')

    # 绘制 boxcox_transformed 的直方图
    plt.hist(boxcox_transformed, bins=160, alpha=0.3,
             color='green', label='Box-Cox Transformed')

    # 绘制 yeo_johnson_transformed 的直方图
    plt.hist(yeo_johnson_transformed, bins=160, alpha=0.3,
             color='red', label='Yeo-Johnson Transformed')

    # 绘制 winsorized_log_return 的直方图
    plt.hist(winsorized_log_return, bins=160, alpha=0.3,
             color='red', label='Winsorized Transformed')

    # 计算并显示 skewness 和 kurtosis
    log_return_skewness = skew(log_return)
    log_return_kurtosis = kurtosis(log_return)
    log_log_return_skewness = skew(log_log_return)
    log_log_return_kurtosis = kurtosis(log_log_return)
    boxcox_skewness = skew(boxcox_transformed)
    boxcox_kurtosis = kurtosis(boxcox_transformed)
    yeo_johnson_skewness = skew(yeo_johnson_transformed)
    yeo_johnson_kurtosis = kurtosis(yeo_johnson_transformed)
    winsorized_skewness = skew(winsorized_log_return)
    winsorized_kurtosis = kurtosis(winsorized_log_return)

    # 在图上显示 skewness 和 kurtosis
    plt.text(0.05, 0.95,
             f'Log Return Skewness: {log_return_skewness:.2f}\nLog Return Kurtosis: {log_return_kurtosis:.2f}',
             fontsize=10, transform=plt.gca().transAxes, va='top', ha='left', bbox=dict(facecolor='blue', alpha=0.5))
    plt.text(0.05, 0.85,
             f'Log Log Return Skewness: {log_log_return_skewness:.2f}\nLog Log Return Kurtosis: {log_log_return_kurtosis:.2f}',
             fontsize=10, transform=plt.gca().transAxes, va='top', ha='left', bbox=dict(facecolor='orange', alpha=0.5))
    plt.text(0.05, 0.75,
             f'Box-Cox Transformed Skewness: {boxcox_skewness:.2f}\nBox-Cox Transformed Kurtosis: {boxcox_kurtosis:.2f}',
             fontsize=10, transform=plt.gca().transAxes, va='top', ha='left', bbox=dict(facecolor='green', alpha=0.5))
    plt.text(0.05, 0.65,
             f'Yeo-Johnson Transformed Skewness: {yeo_johnson_skewness:.2f}\nYeo-Johnson Transformed Kurtosis: {yeo_johnson_kurtosis:.2f}',
             fontsize=10, transform=plt.gca().transAxes, va='top', ha='left', bbox=dict(facecolor='red', alpha=0.5))
    plt.text(0.05, 0.55,
             f'Winsorized Skewness: {winsorized_skewness:.2f}\nWinsorized Kurtosis: {winsorized_kurtosis:.2f}',
             fontsize=10, transform=plt.gca().transAxes, va='top', ha='left', bbox=dict(facecolor='purple', alpha=0.5))

    # 添加图例和标签
    plt.legend()
    plt.title('Histogram of Transformed Series')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # 显示图形
    plt.show()


def check_zscore_window_series(df, column, n_values=[50, 100, 200, 250, 300, 450, 600, 1200, 2400, 4800, 9600]):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    if column == 'c':
        # 计算原始对数收益率，此时 column 应为 NumPy ndarray
        log_return = (np.log(df[column]).diff(1).fillna(0)*1).shift(-1)
        # 将 NaN 值替换为0
        log_return = np.where(np.isnan(log_return), 0, log_return)
    else:
        log_return = df[column].values

    # 绘制原始对数收益率的直方图
    ax1.hist(log_return, bins=100, alpha=0.5,
             color='blue', label='Original Log Return')

    # 计算偏度和峰度
    skewness_orig = skew(log_return)
    kurtosis_orig = kurtosis(log_return)
    ax1.text(0.01, 0.9, f'Original Skew: {skewness_orig:.2f}, Kurtosis: {kurtosis_orig:.2f}',
             transform=ax1.transAxes, fontsize=10, color='blue')

    # 颜色生成器
    color_cycle = plt.cm.viridis(np.linspace(0, 1, len(n_values)))

    # 计算并绘制每个n值的滚动标准化对数收益率
    for n, color in zip(n_values, color_cycle):
        # rolling_mean = np.convolve(log_return, np.ones(n) / n, mode='valid')
        # # 填充使长度一致
        # rolling_mean = np.concatenate(
        #     (np.full(n - 1, np.nan), rolling_mean, np.full(len(log_return) - len(rolling_mean) - (n - 1), np.nan)))
        # rolling_std = np.sqrt(np.convolve((log_return - rolling_mean) ** 2, np.ones(n) / n, mode='valid'))
        # rolling_std = np.concatenate(
        #     (np.full(n - 1, np.nan), rolling_std, np.full(len(log_return) - len(rolling_std) - (n - 1), np.nan)))
        # norm_log_return = (log_return - rolling_mean) / rolling_std

        norm_log_return = _rolling_zscore_np(log_return, n)

        # 绘制直方图
        ax1.hist(norm_log_return, bins=100, alpha=0.5,
                 color=color, label=f'Norm Log Return n={n}')

        # 计算偏度和峰度
        skewness = skew(norm_log_return[~np.isnan(norm_log_return)])
        kurtosis_val = kurtosis(norm_log_return[~np.isnan(norm_log_return)])
        ax1.text(0.01, 0.8 - 0.07 * n_values.index(n),
                 f'n={n} Skew: {skewness:.2f}, Kurtosis: {kurtosis_val:.2f}',
                 transform=ax1.transAxes, fontsize=10, color=color)

    # 在第二个子图上设置两个y轴
    ax2_2 = ax2.twinx()
    for n, color in zip(n_values, color_cycle):
        norm_log_return = _rolling_zscore_np(log_return, n)
        ax2.plot(np.arange(len(df[column])), norm_log_return.cumsum(
        ), color=color, label=f'Cumulative Norm Log Ret n={n}')

    # 绘制原始数值
    ax2_2.plot(np.arange(len(df[column])), df[column], color='black',
               linewidth=2, label='Original Values', alpha=0.7)
    ax2.set_ylabel('Cumulative Norm Log Ret')
    ax2_2.set_ylabel('Original Values')

    ax1.set_title('Histogram of Log Returns and Normalized Log Returns')
    ax1.set_xlabel('Log Return Value')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax2.legend(loc='upper left')
    ax2_2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


def _rolling_zscore(x1, n=300):  # 标准差标准化
    x1 = x1.flatten().astype(np.double)
    x1 = np.nan_to_num(x1)
    x1_rolling_avg = ta.SMA(x1, n)  # 使用TA-Lib中的简单移动平均函数SMA
    x_value = _DIVP(x1, ta.STDDEV(x1, n))
    # x_value = np.clip(x_value, -6, 6)
    return np.nan_to_num(x_value)


def _rolling_zscore_np(x1, n=300):  # 标准差标准化
    x = np.asarray(x1, dtype=np.float64)
    x1 = np.nan_to_num(x1)
    x1_rolling_avg = ta.SMA(x1, n)  # 使用TA-Lib中的简单移动平均函数SMA
    x_value = _DIVP(x1, ta.STDDEV(x1, n))
    # x_value = np.clip(x_value, -6, 6)
    return np.nan_to_num(x_value)


def _DIVP(x1, x2):  # 零分母保护的除法
    x1 = x1.flatten().astype(np.double)
    x2 = x2.flatten().astype(np.double)
    x = np.nan_to_num(np.where(x2 != 0, np.divide(x1, x2), 0))

    return x


def cal_ret(sym: str, freq: str, n: int) -> pd.Series:
    '''计算未来n个周期的收益率
    params
    sym:品种
    freq:降频周期
    n:第几个周期后的收益率'''
    z = data_load(sym)
    z = resample(z, freq)

    ret = (np.log(z.c).diff(n)*1).shift(-n)  # 计算对数收益率
    ret = np.where(np.isnan(ret), 0, ret)

    # 关键 - 对label进行了rolling_zscore处理！！
    ret_ = _rolling_zscore_np(ret, n)
    return ret_
