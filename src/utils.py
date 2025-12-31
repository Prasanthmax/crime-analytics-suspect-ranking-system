import pandas as pd

def read_csv(path):
    return pd.read_csv(path)

def to_datetime(df, col):
    df[col] = pd.to_datetime(df[col])
    return df
