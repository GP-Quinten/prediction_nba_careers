"""functions for the training module"""
import logging
from ast import literal_eval
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


from qtracker.utils import module_from_string

logger = logging.getLogger(__name__)


def get_covariates(data: pd.DataFrame, covariates: Union[List, str], outcome: str, weight: str) -> pd.DataFrame:
    """
    Get covariates of interest in a dataset

    Args:
        data: Dataset of interest
        covariates: The list of covariates to used; if set to "all", all the columns are used except the outcome column
        outcome: Name of the column with the label
        weight: Name of the column with the sample weights

    Returns:
        The covariates of interest

    """
    if covariates == "all" or covariates is None:
        return data[[column for column in data.columns if column not in [outcome, weight]]]
    if isinstance(covariates, List):
        return data[[column for column in covariates if column not in [outcome, weight]]]
    raise ValueError("covariates wrongly defined")


def get_outcome(data: pd.DataFrame, outcome_field: str) -> pd.Series:
    """
    Get outcome of interest in a dataset

    Args:
        data: Dataset of interest
        outcome_field: Name of the column with the label

    Returns:
        The outcome of interest

    """
    return (data[outcome_field]).astype(int)


def get_sample_weights(data: pd.DataFrame, sample_weight_field: Union[str, None]) -> pd.Series:
    """
    Get sample weights of the dataset

    Args:
        data: Dataset of interest
        sample_weight_field: Name of the column with the sample weights

    Returns:
        The sample weights of the dataset

    """
    if sample_weight_field is not None:
        return (data[sample_weight_field]).astype(float)
    return pd.Series(np.ones(len(data)), name="sample_weight")


def train_model(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    weights_train: pd.DataFrame,
    classifier_name: str,
    hyperparameters: Optional[Dict[str, Any]] = None,
    random_state: Optional[int] = None,
) -> BaseEstimator:
    """
    Train the model

    Args:
        x_train: The features of the train set
        y_train: The labels of the train set
        weights_train: The sample weights of the train set
        classifier_name: Name of the model to be trained
        hyperparameters: The set of hyperparameters
        random_state: A random state for reproductibility purpose

    Returns:
        The trained model

    """
    hyperparameters = hyperparameters or {}
    hyperparameters["random_state"] = random_state
    if classifier_name == "sklearn.neural_network.MLPClassifier":
        if "hidden_layer_sizes" not in hyperparameters.keys():
            n_hidden_layer = hyperparameters["n_hidden_layer"]
            hidden_layer_sizes = tuple()
            for i in range(n_hidden_layer):
                hidden_layer_size = hyperparameters[f"hidden_layer_size_{i}"]
                hidden_layer_sizes += (hidden_layer_size,)
                hyperparameters.pop(f"hidden_layer_size_{i}", None)
            hyperparameters["hidden_layer_sizes"] = hidden_layer_sizes
            hyperparameters.pop("n_hidden_layer", None)
        else:
            hyperparameters["hidden_layer_sizes"] = literal_eval(hyperparameters["hidden_layer_sizes"])
    logger.info("Training model with the following hyperparameters: %s", hyperparameters)
    trained_model = module_from_string(classifier_name)(**hyperparameters)
    if classifier_name == "catboost.CatBoostClassifier":
        trained_model.fit(x_train, y_train, sample_weight=weights_train, silent=True, plot=False)
    elif classifier_name == "sklearn.neural_network.MLPClassifier":
        pipe = Pipeline([("scaler", RobustScaler()), ("model", trained_model)])
        pipe.fit(x_train, y_train)
        logger.info("Model training done")
        return pipe
    else:
        trained_model.fit(x_train, y_train, sample_weight=weights_train)
    logger.info("Model training done")

    return trained_model
