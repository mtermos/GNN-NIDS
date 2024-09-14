import numpy as np


def clean_dataset(df, timestamp_col, flow_id_col):
    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows with any NaN values
    df.dropna(axis=0, how='any', inplace=True)

    # Drop duplicate rows except for the first occurrence, based on all columns except timestamp and flow_id
    df.drop_duplicates(subset=list(set(
        df.columns) - set([timestamp_col, flow_id_col])), keep="first", inplace=True)

    return df
