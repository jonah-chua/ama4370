import pandas as pd
import os, datetime
p = r"c:\Users\ejdch\Downloads\algo trading\data\ETHUSDT_1m_futures\2024-09-26_2024-10-25.parquet"
print('path:', p)
try:
    df = pd.read_parquet(p)
except Exception as e:
    print('Failed to read parquet:', e)
    raise
print('rows,cols:', df.shape)
if 'timestamp' in df.columns:
    ts = df['timestamp']
    print('timestamp dtype:', ts.dtype)
    mn = int(ts.min())
    mx = int(ts.max())
    print('min raw:', mn)
    print('max raw:', mx)
    def _to_dt(v):
        if v > 1e12:
            return pd.to_datetime(v, unit='ms')
        elif v > 1e9:
            return pd.to_datetime(v, unit='s')
        else:
            return pd.to_datetime(v, unit='s')
    print('min ->', _to_dt(mn))
    print('max ->', _to_dt(mx))
    print('\nhead timestamps:')
    for i,r in df.head(3).iterrows():
        v = int(r['timestamp'])
        print(i, v, _to_dt(v))
    print('\ntail timestamps:')
    for i,r in df.tail(3).iterrows():
        v = int(r['timestamp'])
        print(i, v, _to_dt(v))
else:
    print('No timestamp column')

st = os.path.getmtime(p)
print('file mtime:', datetime.datetime.fromtimestamp(st))
print('done')
