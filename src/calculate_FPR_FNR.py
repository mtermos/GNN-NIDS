import numpy as np
# def calculate_FPR_FNR(cm):

#     TN = cm[0][0]
#     FN = cm[1][0]
#     TP = cm[1][1]
#     FP = cm[0][1]

#     # Sensitivity, hit rate, recall, or true positive rate
#     TPR = TP / (TP + FN) if (TP + FN) != 0 else 0
#     # Specificity or true negative rate
#     TNR = TN / (TN + FP) if (TN + FP) != 0 else 0
#     # Precision or positive predictive value
#     PPV = TP / (TP + FP) if (TP + FP) != 0 else 0
#     # Negative predictive value
#     NPV = TN / (TN + FN) if (TN + FN) != 0 else 0
#     # Fall out or false positive rate
#     FPR = FP / (FP + TN) if (FP + TN) != 0 else 0
#     # False negative rate
#     FNR = FN / (TP + FN) if (TP + FN) != 0 else 0
#     # False discovery rate
#     FDR = FP / (TP + FP) if (TP + FP) != 0 else 0

#     return FPR, FNR


def calculate_FPR_FNR_with_global(cm):
    """
    Calculate FPR and FNR for each class and globally for a multi-class confusion matrix.

    Parameters:
        cm (numpy.ndarray): Confusion matrix of shape (num_classes, num_classes).

    Returns:
        dict: A dictionary containing per-class and global FPR and FNR.
    """
    num_classes = cm.shape[0]
    results = {"per_class": {}, "global": {}}

    # Initialize variables for global calculation
    total_TP = 0
    total_FP = 0
    total_FN = 0
    total_TN = 0

    # Per-class calculation
    for class_idx in range(num_classes):
        TP = cm[class_idx, class_idx]
        FN = np.sum(cm[class_idx, :]) - TP
        FP = np.sum(cm[:, class_idx]) - TP
        TN = np.sum(cm) - (TP + FP + FN)

        # Calculate FPR and FNR for this class
        FPR = FP / (FP + TN) if (FP + TN) != 0 else None
        FNR = FN / (TP + FN) if (TP + FN) != 0 else None

        # Store per-class results
        results["per_class"][class_idx] = {"FPR": FPR, "FNR": FNR}

        # Update global counts
        total_TP += TP
        total_FP += FP
        total_FN += FN
        total_TN += TN

    # Global calculation
    global_FPR = total_FP / \
        (total_FP + total_TN) if (total_FP + total_TN) != 0 else None
    global_FNR = total_FN / \
        (total_FN + total_TP) if (total_FN + total_TP) != 0 else None

    results["global"]["FPR"] = global_FPR
    results["global"]["FNR"] = global_FNR

    return results
