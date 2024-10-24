"""functions for the evaluation module"""
import logging
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from qtracker.evaluation import DistPlotDisplay
from qtracker.evaluation.custom_metrics import auprc_score

logger = logging.getLogger(__name__)


def compute_metrics(
    y_train_true: pd.DataFrame,
    y_train_score: pd.DataFrame,
    y_test_true: pd.DataFrame,
    y_test_score: pd.DataFrame,
) -> Dict[str, float]:
    """
    Compute several metrics of interest for train and test sets

    Args:
        y_train_true: Real labels of the train set
        y_train_score: Predicted probabilities for the positive target of the train set
        y_test_true: Real labels of the test set
        y_test_score: Predicted probabilities for the positive target of the test set

    Returns:
        All the metrics of interest for the train and test sets

    """
    y_train_score = y_train_score.values
    y_test_score = y_test_score.values
    y_train_true = y_train_true.astype(int).values
    y_test_true = y_test_true.astype(int).values
    threshold = 0.5
    y_train_pred = y_train_score >= threshold
    y_test_pred = y_test_score >= threshold

    metrics = {
        "roc_auc.train": roc_auc_score(y_true=y_train_true, y_score=y_train_score),
        "roc_auc.test": roc_auc_score(y_true=y_test_true, y_score=y_test_score),
        "auprc.train": auprc_score(y_true=y_train_true, y_score=y_train_score),
        "auprc.test": auprc_score(y_true=y_test_true, y_score=y_test_score),
        "f1_weighted.train": f1_score(y_true=y_train_true, y_pred=y_train_pred, average="weighted"),
        "f1_weighted.test": f1_score(y_true=y_test_true, y_pred=y_test_pred, average="weighted"),
        "f1.train": f1_score(y_true=y_train_true, y_pred=y_train_pred),
        "f1.test": f1_score(y_true=y_test_true, y_pred=y_test_pred),
        "recall.train": recall_score(y_true=y_train_true, y_pred=y_train_pred),
        "recall.test": recall_score(y_true=y_test_true, y_pred=y_test_pred),
        "precision.train": precision_score(y_true=y_train_true, y_pred=y_train_pred),
        "precision.test": precision_score(y_true=y_test_true, y_pred=y_test_pred),
        "brier_score.train": brier_score_loss(y_true=y_train_true, y_prob=y_train_pred),
        "brier_score.test": brier_score_loss(y_true=y_test_true, y_prob=y_test_pred),
    }
    return metrics


def visualize_metrics_plots(  # pylint: disable-msg=too-many-locals
    y_train_true: pd.DataFrame,
    y_train_score: pd.DataFrame,
    y_test_true: pd.DataFrame,
    y_test_score: pd.DataFrame,
    outcome_field: str,
    classifier_name: str,
) -> Dict[str, plt.figure]:
    """
    Visualize several metrics plots of interest for train and test sets

    Args:
        y_train_true: Real labels of the train set
        y_train_score: Predicted probabilities for the positive target of the train set
        y_test_true: Real labels of the test set
        y_test_score: Predicted probabilities for the positive target of the test set
        outcome_field: Name of the column where the real label is stored
        classifier_name: Name of the classifier

    Returns:
        All the plots on interest

    """
    classifier_name = classifier_name.split(".")[-1]

    y_train_score = y_train_score.values.ravel()
    y_test_score = y_test_score.values.ravel()

    y_train_true = y_train_true[outcome_field].astype(int).values
    y_test_true = y_test_true[outcome_field].astype(int).values

    logger.info("Visualize ROC curve (train & test)")
    roc_curve_train_display = RocCurveDisplay.from_predictions(
        y_true=y_train_true, y_pred=y_train_score, name=classifier_name
    )
    roc_curve_test_display = RocCurveDisplay.from_predictions(
        y_true=y_test_true, y_pred=y_test_score, name=classifier_name
    )
    roc_curve_display, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    roc_curve_train_display.plot(ax=ax1)
    ax1.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    ax1.set_title("ROC curve for the train set")
    roc_curve_test_display.plot(ax=ax2)
    ax2.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    ax2.set_title("ROC curve for the test set")
    roc_curve_display.tight_layout()

    logger.info("Visualize Precision-Recall curve (train & test)")
    train_precision, train_recall, _ = precision_recall_curve(y_train_true, y_train_score)
    test_precision, test_recall, _ = precision_recall_curve(y_test_true, y_test_score)
    # Use AUC function to calculate the area under the curve of precision recall curve
    train_auc_precision_recall = auc(train_recall, train_precision)
    test_auc_precision_recall = auc(test_recall, test_precision)
    pr_curve_train_display = PrecisionRecallDisplay(
        precision=train_precision,
        recall=train_recall,
        average_precision=None,
        estimator_name=f"{classifier_name} (AUPRC = {round(train_auc_precision_recall, 3)})",
    )
    pr_curve_test_display = PrecisionRecallDisplay(
        precision=test_precision,
        recall=test_recall,
        average_precision=None,
        estimator_name=f"{classifier_name} (AUPRC = {round(test_auc_precision_recall, 3)})",
    )
    pr_curve_display, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    pr_curve_train_display.plot(ax=ax1)
    ax1.set_title("Precision-Recall curve for the train set")
    pr_curve_test_display.plot(ax=ax2)
    ax2.set_title("Precision-Recall curve for the test set")
    pr_curve_display.tight_layout()

    logger.info("Visualize confusion matrix (train & test)")
    y_train_pred = y_train_score >= 0.5
    y_test_pred = y_test_score >= 0.5
    cm_train = confusion_matrix(y_train_true, y_train_pred)
    cm_train_display = ConfusionMatrixDisplay(cm_train)
    cm_test = confusion_matrix(y_test_true, y_test_pred)
    cm_test_display = ConfusionMatrixDisplay(cm_test)

    cm_display, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    cm_train_display.plot(ax=ax1, cmap="Blues")
    ax1.set_title(f"Confusion matrix for the train set ({classifier_name})")
    cm_test_display.plot(ax=ax2, cmap="Blues")
    ax2.set_title(f"Confusion matrix for the test set ({classifier_name})")
    cm_display.tight_layout()

    logger.info("Visualize classes distribution (train & test)")
    classes_distplot_display = DistPlotDisplay(
        y_train_true=y_train_true,
        y_train_score=y_train_score,
        y_test_true=y_test_true,
        y_test_score=y_test_score,
    )
    classes_distplot_display.plot(neg_label=f"no_{outcome_field}", pos_label=outcome_field)

    plots = {
        "roc_curve.png": roc_curve_display,
        "precision_recall_curve.png": pr_curve_display,
        "classes_distribution.png": classes_distplot_display.figure_,
        "confusion_matrix.png": cm_display,
    }

    logger.info("Model evaluation done")

    return plots


def get_feature_names(pipeline: Pipeline) -> List[str]:
    """
    Retrieve name of the features

    Args
        pipeline: Pipeline to extract the feature names from

    Returns:
        List of the feature names

    """
    feature_names = []
    transformer_list_features = pipeline.named_steps["feature_extraction"].transformer_list
    for _, transformer_feature in enumerate(transformer_list_features):
        feature_names += list(transformer_feature[1].named_steps["DictVectorizer"].get_feature_names_out())
    return feature_names
