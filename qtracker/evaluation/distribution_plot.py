"""Class to plot the positive and negative distribution predicted by a ML model"""
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.linalg import LinAlgError


class DistPlotDisplay:  # pylint: disable=too-few-public-methods
    """
    Custom distribution plot visualization
    """

    def __init__(
        self,
        y_train_true: pd.Series,
        y_train_score: pd.Series,
        y_test_true: pd.Series,
        y_test_score: pd.Series,
    ):
        self.y_train_true = y_train_true
        self.y_train_score = y_train_score
        self.y_test_true = y_test_true
        self.y_test_score = y_test_score
        self.ax_ = None
        self.figure_ = None

    @staticmethod
    def binary_distplot(  # pylint: disable-msg=too-many-arguments, too-many-locals
        y_true: np.array,
        y_score: np.array,
        neg_label: str,
        pos_label: str,
        kde: bool = True,
        bins: Union[int, None] = 50,
        axis: plt.axis = None,
        style: str = "whitegrid",
        font_scale: float = 1.3,
        figsize: tuple = (10, 6),
        **kwargs,
    ) -> plt.axis:
        """
        This function computes the distribution plot of a binary classifier.

        Args:
            y_true: true values we want to predict
            y_score: probabilities predicted by the classifier
            neg_label: name of the negative target
            pos_label: name of the positive target
            kde: whether to plot a gaussian kernel density estimate.
            bins: specification of hist bins, or None to use Freedman-Diaconis rule
            axis: Axes objects to plot
            style: style of the plot
            font_scale: font scale of the plot
            figsize: size of the plot

        Returns:
            the Axes objects with the distribution plot drawn onto it.

        """
        sns.set(style=style, font_scale=font_scale)

        df_proba = pd.DataFrame()
        y_true = np.array(y_true).astype(int)
        df_proba["target"] = y_true
        df_proba["proba"] = np.array(y_score)[list(range(len(y_true))), y_true]
        if axis is None:
            _, axis = plt.subplots(figsize=figsize)

        neg_color = kwargs.get("neg_color", "darkblue")
        pos_color = kwargs.get("pos_color", "forestgreen")
        kde_kws = kwargs.get("kde_kws", {"bw": 0.1})

        for target, label, color in zip(np.unique(y_true), [neg_label, pos_label], [neg_color, pos_color]):
            distribution = (1 - target) * (1 - df_proba[df_proba["target"] == target].proba) + target * df_proba[
                df_proba["target"] == target
            ].proba
            try:
                sns.distplot(distribution, kde=kde, color=color, bins=bins, label=label, ax=axis, kde_kws=kde_kws)
            except LinAlgError:
                sns.distplot(distribution, kde=False, color=color, bins=bins, label=label, ax=axis, kde_kws=kde_kws)
        axis.set_xlim((0, 1))
        axis.set_xlabel("Probability")
        axis.set_ylabel("Density")
        axis.set_title("Distribution plot of the classifier")
        axis.legend()
        return axis

    def plot(
        self,
        figsize: tuple = (15, 5),
        title: str = "Distribution plot",
        neg_label: str = "negative",
        pos_label: str = "positive",
    ) -> "DistPlotDisplay":
        """
        This function plots the distribution plot of a binary classifier for the train and test sets.

        Args:
            figsize: size of the plot
            title: title of the plot
            neg_label: name of the negative target
            pos_label: name of the positive target

        Returns:
            the DistPlotDisplay object.

        """
        y_train_true = np.array(self.y_train_true)
        y_train_score = np.array(self.y_train_score)
        y_train_score = np.hstack(((1 - y_train_score)[:, np.newaxis], y_train_score[:, np.newaxis]))

        y_test_true = np.array(self.y_test_true)
        y_test_score = np.array(self.y_test_score)
        y_test_score = np.hstack(((1 - y_test_score)[:, np.newaxis], y_test_score[:, np.newaxis]))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        ax1 = self.binary_distplot(
            y_train_true,
            y_train_score,
            neg_label=neg_label,
            pos_label=pos_label,
            bins=[i * 0.02 for i in range(51)],
            axis=ax1,
        )
        ax1.set_title(f"{title} for the train set")

        ax2 = self.binary_distplot(
            y_test_true,
            y_test_score,
            neg_label=neg_label,
            pos_label=pos_label,
            bins=[i * 0.02 for i in range(51)],
            axis=ax2,
        )
        ax2.set_title(f"{title} for the test set")

        ylim = int(max(ax1.get_ylim()[1], ax2.get_ylim()[1]) + 1)
        ax1.set_ylim((0, ylim))
        ax2.set_ylim((0, ylim))

        self.ax_ = (ax1, ax2)
        self.figure_ = fig
        return self
