import pandas as pd
from collections import deque

# ------------------------------
# 1️⃣ Compute Failed Streak
# ------------------------------
def compute_failed_streak(x):
    """
    Compute consecutive failed attempts per user.
    Input: Series of 0/1 (failed_flag)
    Output: Series with streak counts
    """
    return x.groupby((x==0).cumsum()).cumsum()

# ------------------------------
# 2️⃣ Unique Users Last 5 Attempts
# ------------------------------
def unique_users_last_5(series):
    """
    Count unique users per IP in the last 5 login attempts.
    Input: Series of users for one IP
    Output: Series of counts
    """
    arr = series.tolist()
    counts = []
    dq = deque(maxlen=5)
    for user in arr:
        dq.append(user)
        counts.append(len(set(dq)))
    return pd.Series(counts, index=series.index)

# ------------------------------
# 3️⃣ Add Time Features
# ------------------------------
def add_time_features(df, timestamp_col='timestamp'):
    """
    Add hour_of_day and is_night features
    """
    df['hour'] = df[timestamp_col].dt.hour
    df['is_night'] = df['hour'].isin([0,1,2,3,4,5,23]).astype(int)
    return df

# ------------------------------
# 4️⃣ Compute Average Interarrival Per IP
# ------------------------------
def avg_interarrival(df, timestamp_col='timestamp', ip_col='ip'):
    """
    Compute average time difference between consecutive logins per IP
    """
    df_sorted = df.sort_values([ip_col, timestamp_col])
    df_sorted['time_diff'] = df_sorted.groupby(ip_col)[timestamp_col].diff().dt.total_seconds()
    df['avg_interarrival'] = df_sorted.groupby(ip_col)['time_diff'].transform('mean').fillna(0)
    return df
