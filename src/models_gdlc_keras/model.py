from abc import ABCMeta, abstractmethod
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, classification_report

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src.calculate_FPR_FNR import calculate_FPR_FNR_with_global


class Model(metaclass=ABCMeta):

    def __init__(self, sequential=False):
        self.sequential = sequential

    @abstractmethod
    def model_name(self): raise NotImplementedError

    @abstractmethod
    def build(self): raise NotImplementedError

    @abstractmethod
    def train(self): raise NotImplementedError

    @abstractmethod
    def predict(self): raise NotImplementedError

    def evaluate(self, predictions, labels, time, labels_names, labels_dict, verbose=0):
        model_name = self.model_name()
        if verbose == 1:
            print("model: ", model_name)

        # if self.options.multi_class:
        #     average = "weighted"
        # else:
        #     average = "binary"
        average = "weighted"

        accuracy = accuracy_score(labels, predictions)
        recall = recall_score(labels, predictions, average=average)
        precision = precision_score(labels, predictions, average=average)
        f1s = f1_score(labels, predictions, average=average)
        cm = confusion_matrix(labels, predictions)
        print("confusion_matrix:")
        print(cm)
        # plt.figure(figsize=(8, 6))
        # sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", linewidths=.5)
        # plt.title('Confusion Matrix')
        # plt.xlabel('Predicted')
        # plt.ylabel('True')
        # plt.show()

        print("End of confusion_matrix:")
        class_report = classification_report(labels, predictions)
        print("Classification Report:")
        print(class_report)
        print("End of Classification Report:")

        if self.multi_class:
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
                "class_fpr": class_fpr
            }
        else:
            print(cm)
            if cm.shape[0] == 1 and cm.shape[1] == 1:
                return (model_name, {
                    "accuracy": accuracy,
                    "recall": recall,
                    "precision": precision,
                    "f1s": f1s,
                    "FPR": 0,
                    "FNR": 0,
                    "time": time
                })
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

        # if verbose == 1:
        #     print("Accuracy: " + "{:.3%}".format(accuracy))
        #     print("Recall: " + "{:.3%}".format(recall))
        #     print("Precision: " + "{:.3%}".format(precision))
        #     print("F1-Score: " + "{:.3%}".format(f1s))
        #     print("True positive: " + "{}".format(TP))
        #     print("True negative: " + "{}".format(TN))
        #     print("False positive: " + "{}".format(FP))
        #     print("False negative: " + "{}".format(FN))
        #     print("False positive rate: " + "{:.3%}".format(FPR))
        #     print("False negative rate: " + "{:.3%}".format(FNR))

        #     print("Prediction time: " + "{:.3%}".format(time))
        #     print()
        #     print("======================================")
        #     print()

        scores = {
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision,
            "f1s": f1s,
            "FPR": FPR,
            "FNR": FNR,
            "time": time
        }

        if self.multi_class:
            actual = np.vectorize(labels_dict.get)(labels)
            test_pred = np.vectorize(labels_dict.get)(predictions)
        else:
            actual = ["Normal" if i == 0 else "Attack" for i in labels]
            test_pred = ["Normal" if i == 0 else "Attack" for i in predictions]

        cm_new = confusion_matrix(actual, test_pred, labels=labels_names)
        cm_normalized = confusion_matrix(
            actual, test_pred, labels=labels_names, normalize="true")
        results_fpr_fnr = calculate_FPR_FNR_with_global(cm)

        return (model_name, scores, class_report, actual, test_pred, cm_new, cm_normalized, results_fpr_fnr)
