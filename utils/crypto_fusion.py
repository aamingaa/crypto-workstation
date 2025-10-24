from datetime import datetime
import pandas as pd

crypto_lsit = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "SOLUSDT", "XPLUSDT", "AVAXUSDT", "SUIUSDT"]


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
raw_data = []
range_start = 5
range_end = 15

date_list = _generate_date_range(start_date, end_date)


for date in date_list:
    file_path = f'/Volumes/Ext-Disk/data/futures/um/monthly/klines/DOGEUSDT/1h/2025/DOGEUSDT-1h-{date}.zip'
    df = pd.read_csv(file_path)
    # df['date'] = date
    df_list.append(df)

raw_data = pd.concat(df_list)
raw_data['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
raw_data['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
raw_data = raw_data.sort_values(by='close_time', ascending=True)
raw_data = raw_data.drop_duplicates('close_time').reset_index(drop=True)

for i in range(range_start, range_end, 2):
    raw_data[f'ret_{i}'] = (raw_data['close'].shift(-i) / raw_data['close'] - 1).fillna(0)
    # print(f'mean of term {i} with norm window of {window} absolute return values is {raw_data[f"ret_{i}"].abs().mean()}')
    # print(raw_data[f"ret_{i}"].describe())
    # raw_data[f'ret_{i}_norm'] = norm(raw_data[f'ret_{i}'].values, window=window, clip=6)



# raw_data['ret'] = (raw_data['close'] / raw_data['close'].shift(1) - 1).fillna(0)
range_start = 5
range_end = 15
