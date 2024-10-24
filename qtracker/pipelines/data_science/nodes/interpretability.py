"""functions for the evaluation module"""
import logging
from typing import Dict, Union

import matplotlib.pyplot as plt
import pandas as pd
import shap
from matplotlib.figure import Figure
from shapash import SmartExplainer
from sklearn import tree
from sklearn.pipeline import Pipeline

from qtracker.evaluation.custom_metrics import auprc
from qtracker.interpretability.feature_importance import FeatureImportanceDisplay
from qtracker.interpretability.shap_importance import ShapValuesDisplay

logger = logging.getLogger(__name__)


def get_explainer(model: Pipeline, x: pd.DataFrame, y: pd.DataFrame) -> SmartExplainer:
    """
    Generates the Shapash explainer object for interpretability

    Args:
        model: The trained model
        x: The features
        y: The labels

    Returns:
        The Shapash explainer object
    """
    if isinstance(model, Pipeline):
        model = model.named_steps["model"]
    if model.__class__.__name__ != "MLPClassifier":
        explainer = SmartExplainer(model=model)
        explainer.compile(x=x, y_target=y)
        return explainer
    func = lambda data: model.predict_proba(pd.DataFrame(data, columns=x.columns))
    explainer = shap.KernelExplainer(func, data=x, link="logit")
    kept_data = x.iloc[0:1, :]
    return explainer.shap_values(kept_data, nsamples=50)


def run_shapash_app(explainer: SmartExplainer):
    """
    Run the interactive Shapash application to interpret the model

    Args:
        explainer: The Shapash explainer object

    """
    return explainer.run_app(title_story="Shapash Explainer", host="localhost", port=8020)


def visualize_feature_importance_plots(
    model: Pipeline,
    x: pd.DataFrame,
    y: pd.DataFrame,
    random_state: Union[int, None],
) -> Dict[str, Figure]:
    """
    Visualize several features importance plots of interest for train and test sets

    Args:
        model: The trained model
        x: The features
        y: The labels
        random_state: A random state for reproductibility purpose

    Returns:
        All the plots on interest

    """
    y = y.astype(int).values

    if isinstance(model, Pipeline):
        model = model.named_steps["model"]

    if model.__class__.__name__ == "DecisionTreeClassifier":
        figure, ax = plt.subplots(1, 1, figsize=(12, 8))
        tree.plot_tree(
            model,
            max_depth=None,
            feature_names=x.columns,
            class_names=None,
            label="all",
            filled=True,
            impurity=True,
            node_ids=False,
            proportion=False,
            rounded=True,
            precision=3,
            ax=ax,
            fontsize=None,
        )
        return {
            "decision_tree.png": figure,
        }

    final_plots = {}

    if model.__class__.__name__ in [
        "XGBClassifier",
        "LGBMClassifier",
        "CatBoostClassifier",
        "GradientBoostingClassifier",
        "RandomForestClassifier",
    ]:
        logger.info("Visualize feature importances")
        feature_importances_display = FeatureImportanceDisplay.from_estimator(
            estimator=model,
            built_in_importance=True,
            feature_names=x.columns,
        )
        feature_importances_display.plot(max_features=20, title=f"Feature importances for {model.__class__.__name__}")
        final_plots["feature_importances.png"] = feature_importances_display.figure_

        logger.info("Visualize shap values")
        shap_values_display = ShapValuesDisplay(estimator=model, x=x)
        shap_values_display.plot(
            max_features=20, title=f"Feature importances via SHAP values for {model.__class__.__name__}"
        )
        final_plots["SHAP_importances.png"] = shap_values_display.figure_

    logger.info("Visualize permutation importances")
    permutation_display = FeatureImportanceDisplay.from_estimator(
        estimator=model,
        built_in_importance=False,
        permutation_scoring={"roc_auc": "roc_auc", "auprc": auprc, "f1_weighted": "f1_weighted"},
        permutation_x=x,
        permutation_y=y,
        permutation_random_state=random_state,
        feature_names=x.columns,
    )
    permutation_display.plot(
        max_features=20, title=f"Feature importances via permutations for {model.__class__.__name__}"
    )
    final_plots["permutation_importances.png"] = permutation_display.figure_

    return final_plots
