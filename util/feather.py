import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any


def _generate_date_range(start_date: str, end_date: str) -> List[str]:
    """生成日期范围"""
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    date_list = []
    current = start
    while current <= end:
        date_list.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    return date_list

start_date = "2025-01-01"
end_date = "2025-01-01"
date_list = _generate_date_range(start_date, end_date)
data_path_template = '/Volumes/Ext-Disk/data/futures/um/daily/trades/ETHUSDT/ETHUSDT-trades-{date}.{ext}'

monthly_data_path_template = '/Volumes/Ext-Disk/data/futures/um/monthly/trades/ETHUSDT/ETHUSDT-trades-2025-03.{ext}'

for date in date_list:
    # zip_file_path = data_path_template.format(date=date, ext='zip')
    # feather_file_path = data_path_template.format(date=date, ext='feather')

    monthly_zip_file_path = monthly_data_path_template.format(ext='zip')
    monthly_feather_file_path = monthly_data_path_template.format(ext='feather')

    print(f'read csv start {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    df = pd.read_csv(monthly_zip_file_path)
    print(f'read csv end {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(df.head())

    df.to_feather(monthly_feather_file_path)
    # print(f'read feather start {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    # new_df = pd.read_feather(feather_file_path)
    # print(f'read feather end {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')


# monthly_feather_file_path = monthly_data_path_template.format(ext='feather')
# new_df = pd.read_feather(monthly_feather_file_path)
# print(new_df.head())