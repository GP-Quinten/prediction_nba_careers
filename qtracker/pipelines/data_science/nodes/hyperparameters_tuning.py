"""functions for the hyperparameters tuning module"""
import logging
from datetime import datetime
from functools import partial
from typing import Any, Dict, Tuple, Union

import mlflow

import optuna
import pandas as pd
import sklearn.metrics
from matplotlib.figure import Figure
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


from qtracker.evaluation.custom_metrics import CUSTOM_SCORERS, auprc
from qtracker.utils import custom_round, get_search_space, module_from_string, optuna_hyperparams_input

logger = logging.getLogger(__name__)


def objective_optuna_func(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    weights_train: pd.DataFrame,
    classifier_name: str,
    hyperparameters_tuning_conf: Dict,
    random_state: Union[int, None],
    trial: optuna.trial,
):
    """The function performs an iteration of the optimization process:
     - hyperparameters of interest are chosen
     - the model is trained and tested in cross validation mode
     - for each fold several metrics of interest are computed
     - hyperparameters and metrics are stored into MLFlow

    Args:
        x_train: The features of the train set
        y_train: The labels of the train set
        weights_train: The sample weights of the train set
        hyperparameters_tuning_conf: The hyperparameters configuration
        random_state: A random state for reproductibility purpose
        trial: The Optuna trial - process of evaluating an objective function

    Raises:
        ValueError: The metric to be optimized or so-called objective function is not known

    Returns:
        The value of the objective function

    """
    parameters = {"random_state": random_state}
    current_params = optuna_hyperparams_input(trial, classifier_name, hyperparameters_tuning_conf)
    if hyperparameters_tuning_conf["sampler"] != "optuna.samplers.GridSampler":
        parameters.update(current_params)
    else:
        parameters.update(trial.params)
    logging.info(parameters)
    model = module_from_string(classifier_name)(**parameters)
    if classifier_name == "catboost.CatBoostClassifier":
        fit_params = {"sample_weight": weights_train, "silent": True, "plot": False}
    elif classifier_name == "sklearn.neural_network.MLPClassifier":
        model = Pipeline([("scaler", RobustScaler()), ("model", model)])
        fit_params = {}
    else:
        fit_params = {"sample_weight": weights_train}
    skf = StratifiedKFold(n_splits=hyperparameters_tuning_conf["n_folds"], shuffle=True, random_state=random_state)
    if hyperparameters_tuning_conf["goalMetric"] in sklearn.metrics.get_scorer_names():
        default_scoring = hyperparameters_tuning_conf["goalMetric"]
        default_scoring_dict = {f"{default_scoring}": default_scoring}
    elif hyperparameters_tuning_conf["goalMetric"] in CUSTOM_SCORERS:
        default_scoring = hyperparameters_tuning_conf["goalMetric"]
        default_scoring_dict = {f"{default_scoring}": CUSTOM_SCORERS[default_scoring]}
    else:
        raise ValueError(f"goalMetric wrongly defined: {hyperparameters_tuning_conf['goalMetric']}")
    cv_results_clf = cross_validate(
        model,
        x_train,
        y_train.values.ravel(),
        cv=skf.split(x_train, y_train),
        scoring={
            "roc_auc": "roc_auc",
            "auprc": auprc,
            "f1_weighted": "f1_weighted",
            "f1": "f1",
            "recall": "recall",
            "precision": "precision",
            "neg_brier_score": "neg_brier_score",
            **default_scoring_dict,
        },
        error_score="raise",
        fit_params=fit_params,
    )
    logging.info(cv_results_clf)
    with mlflow.start_run(nested=True):
        mlflow.log_params(parameters)
        mlflow.log_metrics({f"{key}_mean": custom_round(value.mean()) for key, value in cv_results_clf.items()})
        mlflow.log_metrics({f"{key}_std": custom_round(value.std()) for key, value in cv_results_clf.items()})
    test_score = cv_results_clf[f"test_{default_scoring}"].mean()
    return test_score


def visualize_hyperparameters_plots(study: optuna.Study) -> Dict[str, Figure]:
    """
    Visualize several plots of interest of the hyperparameters tuning process

    Args:
        study: The Optuna study - all the trials

    Returns:
        All the plots on interest

    """
    optimization_history = optuna.visualization.matplotlib.plot_optimization_history(study)
    optimization_history.figure.tight_layout()
    optimization_history.figure.set_size_inches(8, 5)
    param_importances = optuna.visualization.matplotlib.plot_param_importances(study)
    param_importances.figure.tight_layout()
    param_importances.figure.set_size_inches(8, 5)
    parallel_coordinate = optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    parallel_coordinate.figure.tight_layout()
    parallel_coordinate.figure.set_size_inches(9, 5)
    return {
        "optimization_history.png": optimization_history.figure,
        "param_importances.png": param_importances.figure,
        "parallel_coordinate.png": parallel_coordinate.figure,
    }


def hyperparameters_tuning(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    weights_train: pd.DataFrame,
    classifier_name: str,
    hyperparameters_tuning_conf: Union[Dict[str, Any], None],
    optimization_storage_uri: str,
    random_state: Union[int, None],
) -> Tuple[Dict[str, float], Dict[str, Figure]]:
    """
    The global hyperparameters tuning process via Optuna

    Args:
        x_train: The features of the train set
        y_train: The labels of the train set
        weights_train: The sample weights of the train set
        classifier_name: The name of the classifier to be optimized
        hyperparameters_tuning_conf: The hyperparameters configuration
        optimization_storage_uri: Storage URI of the Optuna study
        random_state: A random state for reproductibility purpose

    Returns:
        Best hyperparameters, All the plots on interest

    """
    datetime_now = datetime.now().strftime("%Y/%m/%d@%H:%M:%S")
    if hyperparameters_tuning_conf["sampler"] != "optuna.samplers.GridSampler":
        study = optuna.create_study(
            study_name=f"""{classifier_name}_{datetime_now}""",
            storage=optimization_storage_uri,
            direction=hyperparameters_tuning_conf["goalDirection"],
            sampler=module_from_string(hyperparameters_tuning_conf.get("sampler", None))(),
            pruner=module_from_string(hyperparameters_tuning_conf.get("pruner", None))(),
            load_if_exists=False,
        )
        objective_function_callable = partial(
            objective_optuna_func,
            x_train,
            y_train,
            weights_train,
            classifier_name,
            hyperparameters_tuning_conf,
            random_state,
        )
    else:
        search_space = get_search_space(classifier_name, hyperparameters_tuning_conf)
        study = optuna.create_study(
            study_name=f"""{classifier_name}_{datetime_now}""",
            storage=optimization_storage_uri,
            direction=hyperparameters_tuning_conf["goalDirection"],
            sampler=module_from_string(hyperparameters_tuning_conf.get("sampler", None))(search_space),
            pruner=module_from_string(hyperparameters_tuning_conf.get("pruner", None))(),
            load_if_exists=False,
        )
        objective_function_callable = partial(
            objective_optuna_func,
            x_train,
            y_train,
            weights_train,
            classifier_name,
            hyperparameters_tuning_conf,
            random_state,
        )
    study.optimize(
        func=objective_function_callable,
        n_trials=hyperparameters_tuning_conf["maxTrials"],
        timeout=hyperparameters_tuning_conf["timeout"],
        n_jobs=hyperparameters_tuning_conf["ParallelJobs"],
        show_progress_bar=True,
    )
    best_params = study.best_params
    hp_tuning_plots = visualize_hyperparameters_plots(study)

    return best_params, hp_tuning_plots
