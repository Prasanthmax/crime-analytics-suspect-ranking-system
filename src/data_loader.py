import pandas as pd

def load_processed(path="../data/processed/clean_cases.csv"):
    df = pd.read_csv(path)
    return df
