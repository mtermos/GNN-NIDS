
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix


class ModelEvaluation():
    def __init__(self,
                 model,
                 predictions,
                 labels,
                 time):

        self.model = model
        self.predictions = predictions
        self.labels = labels
        self.time = time

    def evaluate(self, verbose=0):
        model_name = self.model.model_name()
        if verbose == 1:
            print("model: ", model_name)

        accuracy = accuracy_score(self.labels, self.predictions)
        recall = recall_score(self.labels, self.predictions)
        precision = precision_score(self.labels, self.predictions)
        f1s = f1_score(self.labels, self.predictions)
        cm = confusion_matrix(self.labels, self.predictions)

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

        if verbose == 1:
            print("Accuracy: " + "{:.3%}".format(accuracy))
            print("Recall: " + "{:.3%}".format(recall))
            print("Precision: " + "{:.3%}".format(precision))
            print("F1-Score: " + "{:.3%}".format(f1s))
            print("True positive: " + "{}".format(TP))
            print("True negative: " + "{}".format(TN))
            print("False positive: " + "{}".format(FP))
            print("False negative: " + "{}".format(FN))
            print("False positive rate: " + "{:.3%}".format(FPR))
            print("False negative rate: " + "{:.3%}".format(FNR))

            print("Prediction time: " + "{:.3%}".format(self.time))
            print()
            print("======================================")
            print()

        return (model_name, [accuracy, recall, precision, f1s, FPR, FNR, self.time])
