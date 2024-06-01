
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, matthews_corrcoef
import numpy as np
import pandas as pd

def compute_confusion_matrix(y, yhat):
    """
    Compute the confusion matrix
    Inputs:
    - y: Array of true labels
    - yhat: Array of predicted labels
    Output:
    - Confusion matrix
    """
    return confusion_matrix(y, yhat)

def compute_accuracy(conf_matrix):
    """
    Calculate accuracy
    Inputs:
    - conf_matrix: Confusion matrix
    Output:
    - Accuracy
    """

    total = np.sum(conf_matrix)
    correct = np.trace(conf_matrix)
    return correct / total

def compute_mcc(y, yhat):
    """
    Calculate Matthews correlation coefficient
    Inputs:
    - conf_matrix: Confusion matrix
    Output:
    - Matthews correlation coefficient
    """
    return matthews_corrcoef(y, yhat)


def compute_weighted_average_metrics(y, yhat):
    """
    Calculate Matthews correlation coefficient
    Inputs:
    - conf_matrix: Confusion matrix
    Output:
    - Matthews correlation coefficient
    """
    precision, recall, f1, _ = precision_recall_fscore_support(y, yhat, average='weighted')

    return {
        'weighted_precision': precision,
        'weighted_recall': recall,
        'weighted_f1_score': f1
    }

def compute_macro_average_metrics(y, yhat):
    """
    Calculate macro-averaged recall, precision, and F1 score
    Inputs:
    - y: Array of true labels
    - yhat: Array of predicted labels
    Output:
    - Macro-averaged recall, precision, and F1 score
    """
    precision, recall, f1, _ = precision_recall_fscore_support(y, yhat, average='macro')

    return {
        'macro_precision': precision,
        'macro_recall': recall,
        'macro_f1_score': f1
    }

def cal_all_metrics(y, yhat):
    """
    Calculate all metrics
    Inputs:
    - y: Array of true labels
    - yhat: Array of predicted labels
    Output:
    - accuracy, recall, precision, F1 score, MCC
    """
    conf_matrix = compute_confusion_matrix(y, yhat)
    acc = compute_accuracy(conf_matrix=conf_matrix)
    metrics = compute_macro_average_metrics(y, yhat)
    mcc = compute_mcc(y=y, yhat=yhat)
    recall = metrics['macro_recall']
    precision = metrics['macro_precision']
    f1 = metrics['macro_f1_score']
    return [acc, recall, precision, f1, mcc]


def cal_metrics_from_table(df):
    y = df['y']
    yhat = df['yhat']
    results = cal_all_metrics(y, yhat)
    return results
