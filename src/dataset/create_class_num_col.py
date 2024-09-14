from sklearn.preprocessing import LabelEncoder
import numpy as np


def one_dataset_class_num_col(df, class_num_col, class_col):
    classes = df[class_col].unique()
    label_encoder = LabelEncoder()

    label_encoder.fit(list(classes))
    df[class_num_col] = label_encoder.transform(df[class_col])

    labels_names = dict(zip(label_encoder.transform(
        label_encoder.classes_), label_encoder.classes_))

    print(f"==>> labels_names: {labels_names}")

    return df, labels_names


def two_dataset_class_num_col(df1, df2, class_num_col, class_col, class_num_col2=None, class_col2=None):
    if class_num_col2 == None:
        class_num_col2 = class_num_col
    if class_col2 == None:
        class_col2 = class_col

    classes1 = df1[class_col].unique()
    classes2 = df2[class_col2].unique()

    classes = set(np.concatenate([classes2, classes1]))
    label_encoder = LabelEncoder()
    label_encoder.fit(list(classes))

    df1[class_num_col] = label_encoder.transform(
        df1[class_col])
    df2[class_num_col2] = label_encoder.transform(
        df2[class_col2])
    labels_names = dict(zip(label_encoder.transform(
        label_encoder.classes_), label_encoder.classes_))

    print(f"==>> labels_names: {labels_names}")

    return df1, df2, labels_names
