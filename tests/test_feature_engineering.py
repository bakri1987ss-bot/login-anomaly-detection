import pandas as pd
from scripts.feature_engineering import compute_failed_streak

def test_failed_streak():
    df = pd.DataFrame({'user': ['a','a','a','b'], 'failed_flag':[1,1,0,1]})
    df['streak'] = df.groupby('user')['failed_flag'].transform(compute_failed_streak)
    assert df['streak'].tolist() == [1,2,0,1]
