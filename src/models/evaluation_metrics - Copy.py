import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, classification_report
 
# from keras import backend as K
 
 
# def recall_m(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall = true_positives / (possible_positives + K.epsilon())
#     return recall
 
 
# def precision_m(y_true, y_pred):
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     return precision
 
 
# def f1_m(y_true, y_pred):
#     precision = precision_m(y_true, y_pred)
#     recall = recall_m(y_true, y_pred)
#     return 2*((precision*recall)/(precision+recall+K.epsilon()))
 
 
def custom_acc_mc(y_true, y_pred):
    average = "weighted"
 
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average=average)
    precision = precision_score(y_true, y_pred, average=average)
    f1s = f1_score(y_true, y_pred, average=average)
    cm = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred)
 
    FP = cm.sum(axis=0) - np.diag(cm)
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)
 
    FP = FP.astype(float).sum()
    FN = FN.astype(float).sum()
    TP = TP.astype(float).sum()
    TN = TN.astype(float).sum()
 
    records_count = {}
    class_accuracy = {}
    class_precision = {}
    class_recall = {}
    class_f1 = {}
    class_fnr = {}
    class_fpr = {}
    for i in range(cm.shape[0]):
        # True positive, true negative, false positive, false negative
        tp = cm[i, i]
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
 
        # Num of records in this class
        records_count[i] = tp + fn
 
        # Accuracy for the class
        class_accuracy[i] = (tp + tn) / (tp + tn + fp + fn)
 
        class_precision[i] = tp / (tp + fp)
 
        class_recall[i] = tp / (tp + fn)
 
        class_f1[i] = 2 * (precision * recall) / (precision + recall)
 
        class_fnr[i] = fn / (fn + tp)
 
        class_fpr[i] = fp / (fp + tn)
 
    class_report = {
        "records_count": records_count,
        "class_accuracy": class_accuracy,
        "class_precision": class_precision,
        "class_recall": class_recall,
        "class_f1": class_f1,
        "class_fnr": class_fnr,
        "class_fpr": class_fpr,
        "class_report": class_report
    }
 
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP)
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
 
    scores = {
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "f1s": f1s,
        "FPR": FPR,
        "FNR": FNR,
        "class_report": class_report
    }
    return scores
 
 
def custom_acc_binary(y_true, y_pred):
    average = "weighted"
 
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average=average)
    precision = precision_score(y_true, y_pred, average=average)
    f1s = f1_score(y_true, y_pred, average=average)
    cm = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred)
 
    if cm.shape[0] == 1 and cm.shape[1] == 1:
        return {
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision,
            "f1s": f1s,
            "FPR": 0,
            "FNR": 0,
            "class_report": class_report
        }
 
    TN = cm[0][0]
    FN = cm[1][0]
    TP = cm[1][1]
    FP = cm[0][1]
 
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP)
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
 
    scores = {
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "f1s": f1s,
        "FPR": FPR,
        "FNR": FNR,
        "class_report": class_report
    }
    return scores