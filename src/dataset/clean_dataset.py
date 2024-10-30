import numpy as np


def clean_dataset(df, timestamp_col=None, flow_id_col=None):
    print(f"==>> original df.shape[0]: {df.shape[0]}")
    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows with any NaN values
    df.dropna(axis=0, how='any', inplace=True)
    # df.dropna(axis=0, how='any', inplace=True, subset=list(set(
    #     df.columns) - set(id_columns)))
    print(f"==>> after drop na df.shape[0]: {df.shape[0]}")

    # Drop duplicate rows except for the first occurrence, based on all columns except timestamp and flow_id
    id_columns = []
    if timestamp_col:
        id_columns.append(timestamp_col)
    if flow_id_col:
        id_columns.append(flow_id_col)

    if len(id_columns) == 0:
        df.drop_duplicates(keep="first", inplace=True)
    else:
        df.drop_duplicates(subset=list(set(
            df.columns) - set(id_columns)), keep="first", inplace=True)
    print(f"==>> after drop_duplicates df.shape[0]: {df.shape[0]}")

    return df
