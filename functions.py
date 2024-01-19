import numpy as np
from sklearn.metrics import confusion_matrix
from linear_assignment import linear_assignment

def calculate_accuracy(true_row_labels, predicted_row_labels):
    """Get the best accuracy.

    Parameters
    ----------
    true_row_labels: array-like
        The true row labels, given as external information
    predicted_row_labels: array-like
        The row labels predicted by the model

    Returns
    -------
    float
        Best value of accuracy
    """

    cm = confusion_matrix(true_row_labels, predicted_row_labels)
    indexes = linear_assignment(_make_cost_m(cm))
    total = 0
    for row, column in indexes:
        value = cm[row][column]
        total += value
    matrix = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #print(matrix)
    return (total * 1. / np.sum(cm))

def _make_cost_m(cm):
    s = np.max(cm)
    return (- cm + s)