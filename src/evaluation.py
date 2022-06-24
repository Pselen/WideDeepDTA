"""Module for evaluation metric calculations."""
from itertools import combinations

from sklearn.metrics import mean_squared_error, r2_score

def ci(gold_truths, predictions):
    """
    Convordance Index calculation.
    
    For continuous values, CI is a ranking metric. The CI is used to determine
    whether two random drug-target combinations’ projected binding affinity
    values were predicted in the same order as their actual values or not.

    Parameters
    ----------
    gold_truths : list
        Prediction values.
    predictions : list
        Predicted values.

    Returns
    -------
    float
        The CI spans from 0 to 1.0, with 1.0 indicating perfect prediction 
        accuracy and 0.5 indicating a random predictor.

    """
    gold_combs, pred_combs = combinations(gold_truths, 2), combinations(predictions, 2)
    nominator, denominator = 0, 0
    for (g1, g2), (p1, p2) in zip(gold_combs, pred_combs):
        if g2 > g1:
            nominator = nominator + 1 * (p2 > p1) + 0.5 * (p2 == p1)
            denominator = denominator + 1

    return float(nominator / denominator)


def mse(gold_truths, predictions):
    """
    Mean Square Error calculation.
    
    Mean Square Error is a widely used statistic for continuous prediction
    error. It is used in regression tasks to see how near the fitted line is to
    the actual data points, which is shown by connecting the estimated values.

    Parameters
    ----------
    gold_truths : list
        Prediction values.
    predictions : list
        Predicted values.

    Returns
    -------
    float
        MSE value.

    """
    return float(mean_squared_error(gold_truths, predictions, squared=True))


def rmse(gold_truths, predictions):
    """
    Root Mean Square Error calculation.
    
    Root Mean Square Error is the average distance between data points and the
    fitted line, calculated as the square root of MSE.

    Parameters
    ----------
    gold_truths : list
        Prediction values.
    predictions : list
        Predicted values.

    Returns
    -------
    float
        RMSE value.

    """
    return float(mean_squared_error(gold_truths, predictions, squared=False))


def r2(gold_truths, predictions):
    """
    R-squared calculation.
    
    R-squared is a statistical measure that quantifies the proportion of
    variation explained by an independent variable or variables in a regression
    model for a dependent variable. R-squared reveals how much the variation of
    one variable explains the variance of the second variable, whereas
    correlation explains the strength of the relationship between an independent
    and dependent variable. 

    Parameters
    ----------
    gold_truths : list
        Prediction values.
    predictions : list
        Predicted values.

    Returns
    -------
    float
        R-squared value.

    """
    return float(r2_score(gold_truths, predictions))


def evaluate_predictions(y_true, y_preds, metrics):
    """
    General method for metrics.
    
    Gets the metric with actual and predicted y values, and returns the result of the corresponsing 
    function call.

    Parameters
    ----------
    y_true : list
        Prediction values.
    y_preds : list
        Predicted values..
    metrics : string
        CI, R2, RMSE, or MSE.

    Returns
    -------
    dict
        Contains the corresponding score of evaluation metric.

    """
    metrics = [metric.lower() for metric in metrics]
    name_to_fn = {'ci': ci, 'r2': r2, 'rmse': rmse, 'mse': mse}
    return {metric: name_to_fn[metric](y_true, y_preds) for metric in metrics}
