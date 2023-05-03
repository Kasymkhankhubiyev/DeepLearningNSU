import numpy as np


def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    tp, fp, tn, fn = 0, 0, 0, 0

    for p, gt in zip(prediction, ground_truth):
        if p == True and gt == True:
            tp += 1
        elif p == False and gt == True:
            fn += 1
        elif p == False and gt == False:
            tn += 1
        elif p == True and gt == False:
            fp += 1

    precision = tp / (tp + fp)

    recall = tp / (tp + fn)

    f1 = 2 * precision * recall / (precision + recall)

    accuracy = (tp + tn) / len(ground_truth)

    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    accuracy = np.sum(np.where(prediction == ground_truth, 1, 0))

    return accuracy / len(ground_truth)