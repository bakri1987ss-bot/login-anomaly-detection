#!/usr/bin/env python3
# generate_features.py
import sys, pandas as pd

if len(sys.argv) != 3:
    print("Usage: python3 generate_features.py parsed.csv features.csv")
    sys.exit(1)

infile = sys.argv[1]; outfile = sys.argv[2]
df = pd.read_csv(infile, parse_dates=['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)
df = df[df['ip'].notnull()].copy()

features = []
window_minutes = [1,5,15]

for ip, group in df.groupby('ip'):
    g = group.copy().set_index('timestamp')
    times = g.index
    for ts in times:
        row = {'ip': ip, 'timestamp': ts}
        for w in window_minutes:
            start = ts - pd.Timedelta(minutes=w)
            cnt = g.loc[start:ts].shape[0]
            row[f'cnt_last_{w}m'] = cnt
        past = g.loc[:ts]
        succ = past[past['result']=='accepted'].shape[0]
        fail = past[past['result']=='failed'].shape[0]
        total = past.shape[0]
        row['succ_count'] = succ
        row['fail_count'] = fail
        row['total_count'] = total
        row['fail_rate'] = fail / total if total>0 else 0.0
        orig = g.loc[ts]
        if isinstance(orig, pd.DataFrame):
            orig = orig.iloc[0]
        row['event_user'] = orig.get('user', None)
        row['event_result'] = orig.get('result', None)
        features.append(row)

feat_df = pd.DataFrame(features)
feat_df.to_csv(outfile, index=False)
print(f"Wrote features to {outfile} with {len(feat_df)} rows")
PY
