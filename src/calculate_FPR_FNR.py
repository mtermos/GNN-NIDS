def calculate_FPR_FNR(cm):

    TN = cm[0][0]
    FN = cm[1][0]
    TP = cm[1][1]
    FP = cm[0][1]

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN) if (TP + FN) != 0 else 0
    # Specificity or true negative rate
    TNR = TN / (TN + FP) if (TN + FP) != 0 else 0
    # Precision or positive predictive value
    PPV = TP / (TP + FP) if (TP + FP) != 0 else 0
    # Negative predictive value
    NPV = TN / (TN + FN) if (TN + FN) != 0 else 0
    # Fall out or false positive rate
    FPR = FP / (FP + TN) if (FP + TN) != 0 else 0
    # False negative rate
    FNR = FN / (TP + FN) if (TP + FN) != 0 else 0
    # False discovery rate
    FDR = FP / (TP + FP) if (TP + FP) != 0 else 0

    return FPR, FNR
