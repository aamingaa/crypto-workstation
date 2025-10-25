from datetime import datetime
import pandas as pd

crypto_list = ["BTCUSDT","ETHUSDT","BCHUSDT","XRPUSDT","LTCUSDT","TRXUSDT","ETCUSDT","LINKUSDT","XLMUSDT","ADAUSDT","XMRUSDT","DASHUSDT","XTZUSDT","BNBUSDT","ATOMUSDT","ONTUSDT","IOTAUSDT","BATUSDT","VETUSDT","DOGEUSDT","SOLUSDT"]


def _generate_date_range(start_date, end_date):
    start = datetime.strptime(start_date, '%Y-%m')
    end = datetime.strptime(end_date, '%Y-%m')
    date_list = []
    current_dt = start
    while current_dt <= end:
        date_list.append(current_dt.strftime('%Y-%m'))
        # current += timedelta(days=1)
        if current_dt.month == 12:
                current_dt = current_dt.replace(year=current_dt.year + 1, month=1)
        else:
                current_dt = current_dt.replace(month=current_dt.month + 1)
    return date_list


start_date = '2025-01'
end_date = '2025-01'
df_list = []
range_start = 5
range_end = 15

date_list = _generate_date_range(start_date, end_date)
crypto_data_list = {}
for crypto in crypto_list:
    crypto_data_list[crypto] = {}
    raw_data = []
    df_list = []
    for date in date_list:
        file_path = f'/Volumes/Ext-Disk/data/futures/um/monthly/klines/{crypto}/1h/2025/{crypto}-1h-{date}.zip'
        df = pd.read_csv(file_path)
        df_list.append(df)
    raw_data = pd.concat(df_list)
    raw_data['open_time'] = pd.to_datetime(raw_data['open_time'], unit='ms')
    raw_data['close_time'] = pd.to_datetime(raw_data['close_time'], unit='ms')
    raw_data = raw_data.sort_values(by='close_time', ascending=True)
    raw_data = raw_data.drop_duplicates('close_time').reset_index(drop=True)
    raw_data.set_index('close_time', inplace=True)
    for i in range(range_start, range_end, 2):
        raw_data[f'ret_{i}'] = (raw_data['close'].shift(-i) / raw_data['close'] - 1).fillna(0)
    
    crypto_data_list[crypto]['raw_data'] = raw_data

# 创建平均后的数据字典
averaged_crypto_data = {}

# 获取第一个crypto作为基准（保留其他列数据）
first_crypto = list(crypto_data_list.keys())[0]
base_data = crypto_data_list[first_crypto]['raw_data'].copy()

# 找出所有ret列
ret_cols = [col for col in base_data.columns if col.startswith('ret_')]

print(f"开始计算 {len(ret_cols)} 个ret列的平均值...")

crypto_data_list_copy = crypto_data_list.copy()

# 对每个ret列计算所有加密货币的平均
for ret_col in ret_cols:
    print(f"正在处理 {ret_col}...")
    # 收集所有crypto的该ret列（按照index自动对齐）
    ret_series_list = []
    for crypto in crypto_data_list:
        ret_series_list.append(crypto_data_list[crypto]['raw_data'][ret_col])
    
    # 合并成DataFrame并按行计算平均值
    ret_df = pd.concat(ret_series_list, axis=1)
    base_data[ret_col] = ret_df.mean(axis=1)





# 保存到新的字典
averaged_crypto_data['merged_crypto'] = {'raw_data': base_data}

print(f"\n✓ 合并完成！")
print(f"合并后的数据形状: {base_data.shape}")
print(f"\n合并后的数据前5行:")
print(base_data.head())
print(f"\nret列: {ret_cols}")


# crypto_data_list
# 



# for date in date_list:
#     file_path = f'/Volumes/Ext-Disk/data/futures/um/monthly/klines/ETHUSDT/1h/2025/ETHUSDT-1h-{date}.zip'
#     df = pd.read_csv(file_path)
#     df_list.append(df)

# raw_data = pd.concat(df_list)
# raw_data['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
# raw_data['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
# raw_data = raw_data.sort_values(by='close_time', ascending=True)
# raw_data = raw_data.drop_duplicates('close_time').reset_index(drop=True)


# for i in range(range_start, range_end, 2):
#     raw_data[f'ret_{i}'] = (raw_data['close'].shift(-i) / raw_data['close'] - 1).fillna(0)


# raw_data['ret'] = (raw_data['close'] / raw_data['close'].shift(1) - 1).fillna(0)
range_start = 5
range_end = 15
