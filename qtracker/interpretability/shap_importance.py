"""
Plots for SHAP importance
"""

from typing import Any, Optional

import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


class ShapValuesDisplay:  # pylint: disable=too-few-public-methods
    """
    Feature importances visualizations based on SHAP
    """

    def __init__(
        self,
        estimator: Pipeline,
        x: Any,
    ):
        self.x = x
        self.ax_ = None
        self.figure_ = None
        self.estimator = estimator

    def plot(self, max_features: Optional[int] = 20, title: Optional[str] = None) -> "ShapValuesDisplay":
        """
        Plot visualization

        Args:
            max_features: Display only the max_features features with the highest importance, defaults to None
            title: Title of the plot
        Returns:
            The ShapValuesDisplay object

        """
        fig = plt.figure(figsize=(28, 10))
        title = title or "Feature importances via SHAP values"
        fig.suptitle(title, fontweight="bold", fontsize=16)
        try:
            explainer = shap.TreeExplainer(model=self.estimator, link="logit")
            kept_data = self.x
            shap_values = explainer.shap_values(kept_data)
        except:  # pylint: disable-msg=bare-except
            # hack to avoid error when sklearn predictor trained with DataFrame with column names
            func = lambda data: self.estimator.predict_proba(pd.DataFrame(data, columns=self.x.columns))
            explainer = shap.KernelExplainer(func, data=self.x, link="logit")
            kept_data = self.x.iloc[0:50, :]
            shap_values = explainer.shap_values(kept_data, nsamples=50)

        if len(shap_values) == 2:
            shap_values = shap_values[1]
        plt.subplot(1, 2, 1)
        shap.summary_plot(
            shap_values=shap_values,
            features=kept_data,
            feature_names=self.x.columns,
            plot_type="bar",
            plot_size=None,
            max_display=max_features,
            show=False,
        )
        plt.subplot(1, 2, 2)
        shap.summary_plot(
            shap_values=shap_values,
            features=kept_data,
            feature_names=self.x.columns,
            plot_type="dot",
            plot_size=None,
            max_display=max_features,
            show=False,
        )
        ax = fig.gca()
        plt.tight_layout()

        self.ax_ = ax
        self.figure_ = fig
        return self
