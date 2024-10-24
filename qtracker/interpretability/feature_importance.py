"""
Plots for feature importance
"""

from typing import Any, Callable, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.utils import Bunch
from xgboost import XGBModel


class FeatureImportanceDisplay:
    """
    Feature importances visualizations:
        - built-in feature importances when available
        - permutation importances
    """

    def __init__(
        self,
        importance_result: Dict[str, Bunch],
        feature_names: np.ndarray,
    ):
        self.importance_result = importance_result
        self.feature_names = feature_names
        self.ax_ = None
        self.figure_ = None

    @staticmethod
    def xgboost_feature_importance(
        estimator: XGBModel,
        feature_names: np.ndarray,
        importance_type: Union[str, List[str]],
    ) -> Dict[str, Bunch]:
        """
        Extract the feature importances of specified type from the xgboost model and format
        it in a structure compatible with FeatureImportanceDisplay

        Args:
            estimator: XGBoost model to extract the feature importances from
            importance_type: Type of importance passed to the get_score method of xgboost
            feature_names: Name of the features
        Returns:
            A dictionnary of feature importances for each type of interest

        """
        if isinstance(importance_type, str):
            importance_type = [importance_type]

        assert isinstance(estimator, XGBModel)

        if estimator.booster == "gblinear":
            importance_type = ["weight"]

        booster = estimator.get_booster().copy()
        booster.feature_names = list(feature_names)

        result = {}
        for _type in importance_type:
            feature_importance_map = booster.get_score(importance_type=_type)
            result[_type] = Bunch(
                importances=np.array([feature_importance_map.get(column, 0) for column in feature_names])
            )
        return result

    @staticmethod
    def built_in_feature_importance(estimator: Pipeline, estimator_name: str) -> Dict[str, Bunch]:
        """
        Extract the feature importances of a model with built-in feature iportances and format
        it in a structure compatible with FeatureImportanceDisplay

        Args:
            estimator: Model to extract the feature importances from
            estimator_name: Name of the estimator
        Returns:
            A dictionnary of feature importances for each type of interest

        """
        if estimator_name != "LGBMClassifier":
            return {"built-in": Bunch(importances=np.array(estimator.feature_importances_))}
        result = {}
        for _type in ["gain", "split"]:
            result[_type] = Bunch(importances=np.array(estimator.booster_.feature_importance(importance_type=_type)))
        return result

    @classmethod
    def from_estimator(
        cls,
        estimator: Pipeline,
        feature_names: np.ndarray,
        built_in_importance: bool,
        permutation_scoring: Optional[Union[str, Callable, List[str], Dict, None]] = None,
        permutation_x: Optional[Any] = None,
        permutation_y: Optional[Any] = None,
        permutation_random_state: Optional[int] = None,
    ):
        """
        Generate the visualization object from the estimator

        Args:
            estimator: Estimator to generate the importance visualizations from
            feature_names: Name of the features
            built_in_importance: whether or not to plot built-in feature importances
            permutation_scoring: Name of sklearn metrics or callable to use for permuration, defaults to None
            permutation_x: Input features to use for permutation, defaults to None
            permutation_y: Outcome to use for permutation, defaults to None
            permutation_random_state: Random state to use for permutation, defaults to None
        Returns:
            The class object

        """
        importance_result = {}
        estimator_name = estimator.__class__.__name__
        if built_in_importance:
            if estimator_name == "XGBClassifier":
                importance_result.update(
                    cls.xgboost_feature_importance(
                        estimator=estimator, feature_names=feature_names, importance_type=["weight", "gain", "cover"]
                    )
                )
            if estimator_name in [
                "RandomForestClassifier",
                "GradientBoostingClassifier",
                "AdaBoostClassifier",
                "LGBMClassifier",
                "CatBoostClassifier",
            ]:
                importance_result.update(
                    cls.built_in_feature_importance(estimator=estimator, estimator_name=estimator_name)
                )
        if permutation_scoring is not None:
            permutation_importance_result = permutation_importance(
                estimator=estimator,
                X=permutation_x,
                y=permutation_y,
                n_repeats=10,
                random_state=permutation_random_state,
                scoring=permutation_scoring,
            )
            permutation_importance_result = {
                f"permutation / {key}": value for key, value in permutation_importance_result.items()
            }
            importance_result.update(permutation_importance_result)
        return cls(
            importance_result=importance_result,
            feature_names=feature_names,
        )

    def plot(self, max_features: Optional[int] = None, title: Optional[str] = None) -> "FeatureImportanceDisplay":
        """
        Plot visualization

        Args:
            max_features: Display only the max_features features with the highest importance, defaults to None
            title: Title of the plot
        Returns:
            The FeatureImportanceDisplay object

        """
        title = title or "Feature importances"

        n_metrics = len(self.importance_result)
        nrows = 1
        ncols = n_metrics
        fig, ax = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(7 * ncols, 5 * nrows),
        )
        if n_metrics == 1:
            ax = [ax]

        for axis, (importance_type, importance_values) in zip(ax, self.importance_result.items()):

            if importance_values.importances.ndim == 1:
                sorted_idx = importance_values.importances.argsort()
                if max_features is not None:
                    max_features: int
                    sorted_idx = sorted_idx[-max_features:]
                indices = np.arange(0, len(sorted_idx)) + 0.5
                axis.barh(indices, importance_values.importances[sorted_idx], height=0.7)
                axis.set_yticks(indices)
                axis.set_yticklabels(self.feature_names[sorted_idx])
                axis.set_ylim((0, len(sorted_idx)))
            else:
                sorted_idx = importance_values.importances_mean.argsort()
                if max_features is not None:
                    max_features: int
                    sorted_idx = sorted_idx[-max_features:]
                if max_features is not None:
                    max_features: int
                    sorted_idx = sorted_idx[-max_features:]
                axis.boxplot(
                    importance_values.importances[sorted_idx].T,
                    vert=False,
                    labels=None if self.feature_names is None else self.feature_names[sorted_idx],
                )
                axis.axvline(0, ls="--")

            axis.set_title(f"Feature importances ({importance_type})")

        fig.suptitle(title, fontweight="bold", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        self.ax_ = ax
        self.figure_ = fig
        return self
