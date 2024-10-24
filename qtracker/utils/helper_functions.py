"""script for the helper functions"""
from importlib import import_module
from math import exp, floor, log
import random
from typing import Dict, List, Optional, Union

import numpy as np
import optuna


def module_from_string(module_name: str):
    """
    This function loads a module from a string

    Args:
        module_name: name of the module to be loaded e.g. xgboost.XGBClassifier
    Returns:
        The loaded module of interest

    """
    if module_name:
        return getattr(
            import_module((".").join(module_name.split(".")[:-1])),
            module_name.rsplit(".")[-1],
        )
    return module_name


def custom_round(x: Union[float, str], precision: Optional[int] = 2) -> float:
    """
    This function rounds a float in order to keep some decimals after the first non zero decimal

    Args:
        x: The float to be rounded
        precision: The number of decimals after the first non zero decimal
    Returns:
        The rounded float

    """
    x = float(x)
    if x >= 1:
        return round(x, precision)
    if x > 0:
        power_of_10 = floor(log(x, 10))
        return round(x, abs(power_of_10) + precision)
    if x == 0:
        return 0
    power_of_10 = floor(log(-x, 10))
    return round(x, abs(power_of_10) + precision)


def optuna_hyperparams_input(trial: optuna.trial, classifier_name: str, hyperparameters_tuning_conf: Dict) -> Dict:
    """
    This function returns the hyperparameters of interest based on the hyperparameters tuning configuration using optuna

    Args:
        trial: The optuna trial
        classifier_name: The name of the classifier to be optimized
        hyperparameters_tuning_conf: The hyperparameters tuning configuration
    Returns:
        The hyperparameters of interest

    """
    parameters = {}
    for params in hyperparameters_tuning_conf["params"][classifier_name]:
        if params["type"] == "CATEGORICAL":
            parameters[params["parameterName"]] = trial.suggest_categorical(
                name=params["parameterName"], choices=params["categoricalValues"]
            )
        elif params["type"] == "INTEGER" and params["parameterName"] != "hidden_layer_size":
            parameters[params["parameterName"]] = trial.suggest_int(
                name=params["parameterName"],
                low=int(params["minValue"]),
                high=int(params["maxValue"]),
                step=int(params.get("step", 1)),
                log=bool(params.get("log", False)),
            )
        elif params["type"] == "FLOAT":
            step = params.get("step", None)
            step = float(step) if step else step
            parameters[params["parameterName"]] = custom_round(
                trial.suggest_float(
                    name=params["parameterName"],
                    low=float(params["minValue"]),
                    high=float(params["maxValue"]),
                    step=step,
                    log=bool(params.get("log", False)),
                )
            )
        else:
            if params["parameterName"] != "hidden_layer_size":
                raise ValueError(f"parameterName not handled: {params['parameterName']}")
    if classifier_name == "sklearn.neural_network.MLPClassifier":
        n_hidden_layer = parameters["n_hidden_layer"]
        hidden_layer_sizes = tuple()
        for i in range(n_hidden_layer):
            hidden_layer_size_conf = [
                params
                for params in hyperparameters_tuning_conf["params"][classifier_name]
                if params["parameterName"] == "hidden_layer_size"
            ][0]
            hidden_layer_size = trial.suggest_int(
                name=f"""{hidden_layer_size_conf["parameterName"]}_{i}""",
                low=int(hidden_layer_size_conf["minValue"]),
                high=int(hidden_layer_size_conf["maxValue"]),
                step=int(hidden_layer_size_conf.get("step", 1)),
                log=bool(hidden_layer_size_conf.get("log", False)),
            )
            hidden_layer_sizes += (hidden_layer_size,)
        parameters["hidden_layer_sizes"] = hidden_layer_sizes
        parameters.pop("n_hidden_layer", None)
    return parameters


def generate_parameter(config: Dict, used_values: Dict, n_choices_by_param: int):
    """_summary_

    Args:
        config (Dict): _description_
        used_values (Dict): _description_
        n_choices_by_param (int): _description_

    Returns:
        _type_: _description_
    """
    if config["type"] == "INTEGER":
        if "step" in config:
            possible_values = list(range(int(config["minValue"]), int(config["maxValue"]) + 1, int(config["step"])))
        else:
            possible_values = list(range(int(config["minValue"]), int(config["maxValue"]) + 1))
        possible_values = possible_values[::n_choices_by_param]
        possible_values = [value for value in possible_values if value not in used_values[config["parameterName"]]]

    elif config["type"] == "FLOAT":
        if "log" in config and config["log"] is True:
            possible_values = np.logspace(
                np.log10(float(config["minValue"])), np.log10(float(config["maxValue"])), num=100, base=10.0
            )
        elif "step" in config:
            possible_values = np.arange(
                float(config["minValue"]), float(config["maxValue"]) + 1e-10, float(config["step"])
            )
        else:
            possible_values = np.linspace(float(config["minValue"]), float(config["maxValue"]), num=100)
        possible_values = possible_values[::n_choices_by_param]
        possible_values = [value for value in possible_values if value not in used_values[config["parameterName"]]]
        return custom_round(random.choice(possible_values)) if possible_values else None
    elif config["type"] == "CATEGORICAL":
        possible_values = [
            value for value in config["categoricalValues"] if value not in used_values[config["parameterName"]]
        ]

    return random.choice(possible_values) if possible_values else None


def generate_parameters(configs: List[Dict], n_choices_by_param: int):
    """_summary_

    Args:
        configs (List[Dict]): _description_
        n_choices_by_param (int): _description_

    Returns:
        _type_: _description_
    """
    parameters = {config["parameterName"]: [] for config in configs}
    used_values = {config["parameterName"]: [] for config in configs}
    for _ in range(n_choices_by_param):
        for config in configs:
            param_value = generate_parameter(config, used_values, n_choices_by_param)
            if param_value:
                parameters[config["parameterName"]].append(param_value)
                used_values[config["parameterName"]].append(param_value)
    return parameters


def get_search_space(classifier_name: str, hyperparameters_tuning_conf: Dict) -> Dict:
    """_summary_

    Args:
        classifier_name (str): _description_
        hyperparameters_tuning_conf (Dict): _description_

    Returns:
        Dict: _description_
    """
    n_trials = hyperparameters_tuning_conf["maxTrials"]
    n_params = len(hyperparameters_tuning_conf["params"][classifier_name])
    n_choice_by_param = int(exp(log(n_trials) / n_params))
    configs = hyperparameters_tuning_conf["params"][classifier_name]
    return generate_parameters(configs, n_choice_by_param)
