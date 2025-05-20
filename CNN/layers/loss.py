import math
import typing
from typing import Union, List

def binary_cross_entropy(y_true: List[Union[int, float]], y_pred: List[Union[int, float]]) -> float:
    """Compute the binary cross-entropy loss.

    Args:
        y_true (list): Ground truth labels (0 or 1).
        y_pred (list): Predicted probabilities (between 0 and 1).

    Returns:
        float: The computed binary cross-entropy loss.
    """
    assert len(y_true) == len(y_pred), "Input arrays must have the same length"
    loss_sum = 0.0
    
    for i in range(len(y_true)):
        # Clip predictions to avoid log(0) or log(1) issues
        y_p = max(min(y_pred[i], 1.0 - 1e-15), 1e-15)
        loss_sum += -y_true[i] * math.log(y_p) - (1 - y_true[i]) * math.log(1 - y_p)
    return loss_sum / len(y_true)