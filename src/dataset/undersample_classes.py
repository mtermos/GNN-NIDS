import pandas as pd


def undersample_classes(df, class_col, n_undersample, fraction=0.5):
    """
    Undersamples the classes with the highest number of records.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        class_col (str): The name of the class column.
        n_undersample (int): The number of classes to undersample.
        fraction (float): The fraction of samples to keep from the undersampled classes.

    Returns:
        pd.DataFrame: The undersampled DataFrame.
    """
    # Group by the class column and get the count of records in each class
    class_counts = df.groupby(class_col).size()

    # Sort the counts in descending order
    class_counts_sorted = class_counts.sort_values(ascending=False)

    # Get the classes with the highest number of records to undersample
    classes_to_undersample = class_counts_sorted.index[:n_undersample]

    # Undersample the classes with the highest number of records
    dfs = []
    for class_label in class_counts_sorted.index:
        print(f"==>> class_label: {class_label}")
        class_df = df[df[class_col] == class_label]
        if class_label in classes_to_undersample:
            # Specify the fraction of samples to keep
            undersampled_df = class_df.sample(frac=fraction)
            dfs.append(undersampled_df)
        else:
            dfs.append(class_df)

    # Concatenate all DataFrames and shuffle the undersampled DataFrame
    df_undersampled = pd.concat(dfs).sample(frac=1).reset_index(drop=True)

    return df_undersampled


if __name__ == "__main__":
    # Example usage
    data = {
        'class': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'C', 'C'],
        'value': [1, 2, 3, 4, 5, 6, 7, 8, 9]
    }
    df = pd.DataFrame(data)
    df_undersampled = undersample_classes(
        df, class_col='class', n_undersample=1, fraction=0.5)
    print(df_undersampled)
