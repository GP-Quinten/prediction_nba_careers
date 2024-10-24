# QTracker

## Overview

QTracker is a library to track all metrics and artifacts in both hyperparameters tuning process and training process.

The library is designed for binary classification with tabular data.

Below are the supported classifiers:
- xgboost.XGBClassifier
- lightgbm.LGBMClassifier
- catboost.CatBoostClassifier
- sklearn.ensemble.GradientBoostingClassifier
- sklearn.ensemble.RandomForestClassifier
- sklearn.tree.DecisionTreeClassifier
- sklearn.neural_network.MLPClassifier

The package provides demo data to be run in local mode.

## Prerequisites

### Python version

You will need Python 3.8+. You can install [pyenv](https://github.com/pyenv/pyenv) to easily switch between multiple versions of Python.

This library has been developped using Python 3.10.

### Poetry

Poetry is a tool for dependency management and packaging in Python. It allows you to declare the libraries your project depends on and it will manage (install/update) them for you. Poetry offers a lockfile to ensure repeatable installs, and can build your project for distribution.

```bash
curl -sSL https://install.python-poetry.org | python3 - --version 1.4.1
```

```bash
poetry config installer.modern-installation false
```

If poetry command is not found, add the following command to your `.bashrc` or `.zshrc` file:  
`export PATH="$HOME/.local/bin:$PATH"`

See https://python-poetry.org/docs/ for more details.

NB: Poetry requires Python 3.7+.

### Environment around the application

```bash
> poetry shell # setup virtualenv
> poetry install # install dependencies
> (optional) poetry run python -m ipykernel install --user --name <kernel-name> # create associated kernel
```

### Data format

The use of the library implies that you already have datasets ready for the training and serving steps (e.g., cleaning, scaling, imputation already performed). You only have to **provide such train and test sets under csv format** where the columns are you covariates and you outcome of interest.

NB: The preprocessing step will take care of the selection of the needed covariates and outcome.

## Kedro pipelines

Kedro is an open-source Python framework for creating reproducible, maintainable, and modular data science code. This framework helps to accelerate data pipelining, enhance data science prototyping, and promote pipeline reproducibility.

See https://kedro.org/ for more details.

### Configuration

Before runnnig any pipelines you must define the needed configuration in:
- `qtracker/conf/base/globals.yml`
- `qtracker/conf/base/parameters/processing.yml`
- `qtracker/conf/base/parameters/hyperparameters_tuning.yml`
- `qtracker/conf/base/parameters/training.yml`

### Data catalog

Kedro avoids the scattered IO logic problem by collecting all the IO and data-related logic in a single place, which they call the data catalog. In this package it is located at `qtracker/conf/base/catalog.yml` or `qtracker/conf/local/catalog.yml`. 

As a result, you only need to write the IO logic once, and you can reuse it throughout your project. Most of the time, you don’t need to write this IO logic yourself because Kedro has implementations for most common data types and storage systems.

See https://docs.kedro.org/en/stable/data/data_catalog.html for more details.

To run the different pipelines just **be sure to have a train and test sets** either in local (local data catalog) 

```yml
train.final_dataset:
  type: pandas.CSVDataSet
  layer: model_input
  filepath: data/01_model_input/train.final_dataset.csv
```

or on a cloud bucket (base data catalog).

```yml
<train or test>.final_dataset:
  type: pandas.CSVDataSet
  layer: model_input
  filepath: ${cloud_storage}://${bucket_name}/${folder_name}/data/01_model_input/<train or test>.final_dataset.csv
```

NB: If you don't have a test set for your use case you will need to copy your train set as a test set. The test results will obviously be biased but it would allow the pipelines to runned properly.

### Preprocessing

The preprocessing step retrieve the covariates and outcome of interest.

You can update the configuration in `qtracker/conf/base/parameters/processing.yml`.

The preprocessing step is included in each of the following pipeline:
- hyperparameters_tuning
- training
- inference

It can be run using:

```
kedro run --pipeline=<pipeline_name> --tag=preprocessing
```

### Hyperparameters tuning

Hyperparameters tuning is performed in cross validation mode (StratifiedKFold). 

The library uses Optuna, a hyperparameter optimization framework, particularly designed for machine learning. Optuna has modern functionalities as follows:
- Lightweight, versatile, and platform agnostic architecture
- Pythonic search spaces
- Efficient optimization algorithms
- Easy parallelization
- Quick visualization

See https://optuna.readthedocs.io/en/stable/ for more details.

To perform the HP tuning you can define:
- the number of folds in the configuration via `n_folds`.
- the `goalMetric` and the `goalDirection` for the optimization. All metrics defined in scikit-learn can be used. You can also defined specific metrics of interest in `qtracker/evaluation/custom_metrics.py`.
- the number of combinations to test `maxTrials`
- the maximum time to allow for HP tuning via `timeout`
- the sampler to use via `sampler`. It allows to either use Bayesian optimization, Random Search or Grid Search - see the documentation 
[here](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html#which-sampler-and-pruner-should-be-used) to have more insights about which sampler should be used

You can update the configuration in `qtracker/conf/base/parameters/hyperparameters_tuning.yml`. Be sure to have properly define the `classifier_name` in  `qtracker/conf/base/globals.yml`.

Then run the pipeline:

```
kedro run --pipeline=hyperparameters_tuning
```

### Training with evaluation

`qtracker/conf/base/parameters/training.yml`

Be sure to have properly define the `classifier_name` in  `qtracker/conf/base/globals.yml`.

NB: The `best_hyperparameters` generated after the hyperparameters tuning step at not automatically used for the training. Indeed you might want to use other hyperparameters that give slightly similar results for the metric of interest but that bring other advantages. That is why you also need to be sure to have properly define the `hyperparameters` in  `qtracker/conf/base/parameters/training.yml`. It is up to you to copy paste or not the best ones from hyperparameters tuning.

Then run the pipeline:

```
kedro run --pipeline=training
```

### Interpretability app via Shapash

Then run the pipeline:

```
kedro run --pipeline=shapash_app
```

### Inference

In case you already have a trained model and you want to validate it on new data you can locate:
- the model like defined in the data catalog under `trained_model` (or even change the location)
- the dataset like defined in the data catalog under `inference.final_dataset` (or even change the location)

then run the inference pipeline

```
kedro run --pipeline=inference
```

## MLflow tracking

To open the MLFlow UI and compare your experiment, run the following command:

```
mlflow ui --backend-store-uri=sqlite:///mlruns/mlflow.db
```

## Optuna Dashboard

Optuna also provides a dashboard to compare all your experiments and trials. You can run the following command to open it.

```
optuna-dashboard sqlite:///mlruns/optuna.db
```

## Development

### Lint and Tests

* Lint
```
> poetry shell
> black .
> pylint qtracker
# OR
> make style
```
* Test
```
> poetry shell
> pytest
# OR
> make test
```