"""functions for the inference module"""
import logging

import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def sample_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns the head of the given data

    Args:
        data: Dataframe of interest

    Results:
        The head of the dataframe

    """
    return data.head()


def infer_predictions(model: Pipeline, data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate predictions based on the features and trained model

    Args:
        model: Trained model used for generating predictions
        data: Features used for generating predictions

    Results:
        The predictions of the model

    """
    y_proba = model.predict_proba(data)
    result = pd.DataFrame(dict(score=y_proba[:, 1]))
    return result
