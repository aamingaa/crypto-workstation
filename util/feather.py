import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


def _generate_date_range(start_date: str, end_date: str) -> List[str]:
    """生成日期范围（每日）
    
    Args:
        start_date: 开始日期 格式: YYYY-MM-DD
        end_date: 结束日期 格式: YYYY-MM-DD
        
    Returns:
        日期列表
    """
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    date_list = []
    current = start
    while current <= end:
        date_list.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    return date_list


def _generate_month_range(start_month: str, end_month: str) -> List[str]:
    """生成月份范围
    
    Args:
        start_month: 开始月份 格式: YYYY-MM
        end_month: 结束月份 格式: YYYY-MM
        
    Returns:
        月份列表
    """
    start = datetime.strptime(start_month, '%Y-%m')
    end = datetime.strptime(end_month, '%Y-%m')
    
    month_list = []
    current = start
    while current <= end:
        month_list.append(current.strftime('%Y-%m'))
        # 移动到下一个月
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    return month_list


def convert_zip_to_feather_daily(
    start_date: str,
    end_date: str,
    symbol: str = 'ETHUSDT',
    base_path: str = '/Volumes/Ext-Disk/data/futures/um',
    data_type: str = 'trades',
    verbose: bool = True
) -> None:
    """将每日的 zip 格式数据转换为 feather 格式
    
    Args:
        start_date: 开始日期 格式: YYYY-MM-DD
        end_date: 结束日期 格式: YYYY-MM-DD
        symbol: 交易对符号
        base_path: 数据基础路径
        data_type: 数据类型（如 'trades', 'klines' 等）
        verbose: 是否打印详细信息
    """
    date_list = _generate_date_range(start_date, end_date)
    
    data_path_template = f'{base_path}/daily/{data_type}/{symbol}/{symbol}-{data_type}-{{date}}.{{ext}}'
    
    for date in date_list:
        zip_file_path = data_path_template.format(date=date, ext='zip')
        feather_file_path = data_path_template.format(date=date, ext='feather')
        
        # 检查源文件是否存在
        if not Path(zip_file_path).exists():
            print(f'⚠️  文件不存在，跳过: {zip_file_path}')
            continue
        
        # 检查目标文件是否已存在
        if Path(feather_file_path).exists():
            print(f'✓  已存在，跳过: {feather_file_path}')
            continue
        
        try:
            if verbose:
                print(f'\n处理日期: {date}')
                print(f'读取 CSV 开始 {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            
            df = pd.read_csv(zip_file_path)
            
            if verbose:
                print(f'读取 CSV 完成 {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
                print(f'数据行数: {len(df):,}')
                print(df.head())
            
            df.to_feather(feather_file_path)
            
            if verbose:
                print(f'✓ 保存 Feather 完成: {feather_file_path}')
                
        except Exception as e:
            print(f'❌ 处理失败 {date}: {str(e)}')


def convert_zip_to_feather_monthly(
    start_month: str,
    end_month: str,
    symbol: str = 'ETHUSDT',
    base_path: str = '/Volumes/Ext-Disk/data/futures/um',
    data_type: str = 'trades',
    verbose: bool = True
) -> None:
    """将每月的 zip 格式数据转换为 feather 格式
    
    Args:
        start_month: 开始月份 格式: YYYY-MM
        end_month: 结束月份 格式: YYYY-MM
        symbol: 交易对符号
        base_path: 数据基础路径
        data_type: 数据类型（如 'trades', 'klines' 等）
        verbose: 是否打印详细信息
    """
    month_list = _generate_month_range(start_month, end_month)
    
    data_path_template = f'{base_path}/monthly/{data_type}/{symbol}/{symbol}-{data_type}-{{month}}.{{ext}}'
    
    for month in month_list:
        zip_file_path = data_path_template.format(month=month, ext='zip')
        feather_file_path = data_path_template.format(month=month, ext='feather')
        
        # 检查源文件是否存在
        if not Path(zip_file_path).exists():
            print(f'⚠️  文件不存在，跳过: {zip_file_path}')
            continue
        
        # 检查目标文件是否已存在
        if Path(feather_file_path).exists():
            print(f'✓  已存在，跳过: {feather_file_path}')
            continue
        
        try:
            if verbose:
                print(f'\n处理月份: {month}')
                print(f'读取 CSV 开始 {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            
            df = pd.read_csv(zip_file_path)
            
            if verbose:
                print(f'读取 CSV 完成 {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
                print(f'数据行数: {len(df):,}')
                print(df.head())
            
            df.to_feather(feather_file_path)
            
            if verbose:
                print(f'✓ 保存 Feather 完成: {feather_file_path}')
                
        except Exception as e:
            print(f'❌ 处理失败 {month}: {str(e)}')


def convert_zip_to_feather(
    start: str,
    end: str,
    mode: str = 'daily',
    symbol: str = 'ETHUSDT',
    base_path: str = '/Volumes/Ext-Disk/data/futures/um',
    data_type: str = 'trades',
    verbose: bool = True
) -> None:
    """将 zip 格式数据转换为 feather 格式（统一接口）
    
    Args:
        start: 开始时间 (daily模式: YYYY-MM-DD, monthly模式: YYYY-MM)
        end: 结束时间 (daily模式: YYYY-MM-DD, monthly模式: YYYY-MM)
        mode: 'daily' 或 'monthly'
        symbol: 交易对符号
        base_path: 数据基础路径
        data_type: 数据类型（如 'trades', 'klines' 等）
        verbose: 是否打印详细信息
    """
    if mode == 'daily':
        convert_zip_to_feather_daily(start, end, symbol, base_path, data_type, verbose)
    elif mode == 'monthly':
        convert_zip_to_feather_monthly(start, end, symbol, base_path, data_type, verbose)
    else:
        raise ValueError(f"mode 必须是 'daily' 或 'monthly'，当前值: {mode}")


def merge_daily_to_feather(
    start_date: str,
    end_date: str,
    output_path: str,
    symbol: str = 'ETHUSDT',
    base_path: str = '/Volumes/Ext-Disk/data/futures/um',
    data_type: str = 'trades',
    sort_by: Optional[str] = None,
    verbose: bool = True
) -> None:
    """将多个每日的 zip 文件合并为一个 feather 文件
    
    Args:
        start_date: 开始日期 格式: YYYY-MM-DD
        end_date: 结束日期 格式: YYYY-MM-DD
        output_path: 输出文件路径（完整路径，包含文件名）
        symbol: 交易对符号
        base_path: 数据基础路径
        data_type: 数据类型（如 'trades', 'klines' 等）
        sort_by: 排序字段（如 'time', 'timestamp' 等），None则不排序
        verbose: 是否打印详细信息
    """
    date_list = _generate_date_range(start_date, end_date)
    
    data_path_template = f'{base_path}/daily/{data_type}/{symbol}/{symbol}-{data_type}-{{date}}.{{ext}}'
    
    dfs = []
    total_rows = 0
    
    if verbose:
        print(f'\n开始合并 {len(date_list)} 天的数据...')
        print(f'日期范围: {start_date} 到 {end_date}')
    
    for idx, date in enumerate(date_list, 1):
        zip_file_path = data_path_template.format(date=date, ext='zip')
        
        if not Path(zip_file_path).exists():
            if verbose:
                print(f'⚠️  [{idx}/{len(date_list)}] 文件不存在，跳过: {date}')
            continue
        
        try:
            if verbose:
                print(f'📖 [{idx}/{len(date_list)}] 读取: {date}', end=' ')
            
            df = pd.read_csv(zip_file_path)
            dfs.append(df)
            total_rows += len(df)
            
            if verbose:
                print(f'✓ ({len(df):,} 行)')
                
        except Exception as e:
            print(f'\n❌ 读取失败 {date}: {str(e)}')
    
    if not dfs:
        print('❌ 没有成功读取任何数据文件！')
        return
    
    if verbose:
        print(f'\n合并数据中... 总计 {total_rows:,} 行')
    
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # 如果指定了排序字段，进行排序
    if sort_by and sort_by in merged_df.columns:
        if verbose:
            print(f'按 {sort_by} 字段排序中...')
        merged_df = merged_df.sort_values(by=sort_by).reset_index(drop=True)
    
    if verbose:
        print(f'保存到: {output_path}')
    
    # 确保输出目录存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    merged_df.to_feather(output_path)
    
    if verbose:
        print(f'\n✅ 合并完成！')
        print(f'总行数: {len(merged_df):,}')
        print(f'文件大小: {Path(output_path).stat().st_size / (1024**2):.2f} MB')
        print(f'\n数据预览:')
        print(merged_df.head())


def merge_monthly_to_feather(
    start_month: str,
    end_month: str,
    output_path: str,
    symbol: str = 'ETHUSDT',
    base_path: str = '/Volumes/Ext-Disk/data/futures/um',
    data_type: str = 'trades',
    sort_by: Optional[str] = None,
    verbose: bool = True
) -> None:
    """将多个每月的 zip 文件合并为一个 feather 文件
    
    Args:
        start_month: 开始月份 格式: YYYY-MM
        end_month: 结束月份 格式: YYYY-MM
        output_path: 输出文件路径（完整路径，包含文件名）
        symbol: 交易对符号
        base_path: 数据基础路径
        data_type: 数据类型（如 'trades', 'klines' 等）
        sort_by: 排序字段（如 'time', 'timestamp' 等），None则不排序
        verbose: 是否打印详细信息
    """
    month_list = _generate_month_range(start_month, end_month)
    
    data_path_template = f'{base_path}/monthly/{data_type}/{symbol}/{symbol}-{data_type}-{{month}}.{{ext}}'
    
    dfs = []
    total_rows = 0
    
    if verbose:
        print(f'\n开始合并 {len(month_list)} 个月的数据...')
        print(f'月份范围: {start_month} 到 {end_month}')
    
    for idx, month in enumerate(month_list, 1):
        zip_file_path = data_path_template.format(month=month, ext='zip')
        
        if not Path(zip_file_path).exists():
            if verbose:
                print(f'⚠️  [{idx}/{len(month_list)}] 文件不存在，跳过: {month}')
            continue
        
        try:
            if verbose:
                print(f'📖 [{idx}/{len(month_list)}] 读取: {month}', end=' ')
            
            df = pd.read_csv(zip_file_path)
            dfs.append(df)
            total_rows += len(df)
            
            if verbose:
                print(f'✓ ({len(df):,} 行)')
                
        except Exception as e:
            print(f'\n❌ 读取失败 {month}: {str(e)}')
    
    if not dfs:
        print('❌ 没有成功读取任何数据文件！')
        return
    
    if verbose:
        print(f'\n合并数据中... 总计 {total_rows:,} 行')
    
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # 如果指定了排序字段，进行排序
    if sort_by and sort_by in merged_df.columns:
        if verbose:
            print(f'按 {sort_by} 字段排序中...')
        merged_df = merged_df.sort_values(by=sort_by).reset_index(drop=True)
    
    if verbose:
        print(f'保存到: {output_path}')
    
    # 确保输出目录存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    merged_df.to_feather(output_path)
    
    if verbose:
        print(f'\n✅ 合并完成！')
        print(f'总行数: {len(merged_df):,}')
        print(f'文件大小: {Path(output_path).stat().st_size / (1024**2):.2f} MB')
        print(f'\n数据预览:')
        print(merged_df.head())


def merge_to_feather(
    start: str,
    end: str,
    output_path: str,
    mode: str = 'daily',
    symbol: str = 'ETHUSDT',
    base_path: str = '/Volumes/Ext-Disk/data/futures/um',
    data_type: str = 'trades',
    sort_by: Optional[str] = None,
    verbose: bool = True
) -> None:
    """将多个 zip 文件合并为一个 feather 文件（统一接口）
    
    Args:
        start: 开始时间 (daily模式: YYYY-MM-DD, monthly模式: YYYY-MM)
        end: 结束时间 (daily模式: YYYY-MM-DD, monthly模式: YYYY-MM)
        output_path: 输出文件路径（完整路径，包含文件名）
        mode: 'daily' 或 'monthly'
        symbol: 交易对符号
        base_path: 数据基础路径
        data_type: 数据类型（如 'trades', 'klines' 等）
        sort_by: 排序字段（如 'time', 'timestamp' 等），None则不排序
        verbose: 是否打印详细信息
    """
    if mode == 'daily':
        merge_daily_to_feather(start, end, output_path, symbol, base_path, data_type, sort_by, verbose)
    elif mode == 'monthly':
        merge_monthly_to_feather(start, end, output_path, symbol, base_path, data_type, sort_by, verbose)
    else:
        raise ValueError(f"mode 必须是 'daily' 或 'monthly'，当前值: {mode}")


# 使用示例
if __name__ == '__main__':
    # ==================== 单独转换模式 ====================
    # Daily 模式示例 - 将每天的zip单独转换为feather
    # convert_zip_to_feather(
    #     start='2022-01-01',
    #     end='2025-01-01',
    #     mode='daily',
    #     symbol='ETHUSDT',
    #     data_type='trades'
    # )
    
    # Monthly 模式示例 - 将每月的zip单独转换为feather
    convert_zip_to_feather(
        start='2022-01',
        end='2023-12',
        mode='monthly',
        symbol='ETHUSDT',
        data_type='trades'
    )
    
    # ==================== 合并模式 ====================
    # 合并多天数据到一个feather文件
    # merge_to_feather(
    #     start='2025-01-01',
    #     end='2025-01-31',
    #     output_path='./output/ETHUSDT-trades-2025-01-merged.feather',
    #     mode='daily',
    #     symbol='ETHUSDT',
    #     data_type='trades',
    #     sort_by='time'  # 可选：按时间字段排序
    # )
    
    # 合并多个月数据到一个feather文件
    # merge_to_feather(
    #     start='2025-01',
    #     end='2025-03',
    #     output_path='/Volumes/Ext-Disk/data/futures/um/monthly/trades/ETHUSDT/ETHUSDT-trades-2025-01-03-merged.feather',
    #     mode='monthly',
    #     symbol='ETHUSDT',
    #     data_type='trades',
    #     # sort_by='time'  # 可选：按时间字段排序
    # )
    
    pass
    