"""Project pipelines to be created"""
from kedro.pipeline import Pipeline, node, pipeline

from qtracker.pipelines.data_science.nodes.evaluation import (
    compute_metrics,
    visualize_metrics_plots,
)
from qtracker.pipelines.data_science.nodes.hyperparameters_tuning import (
    hyperparameters_tuning,
)
from qtracker.pipelines.data_science.nodes.inference import infer_predictions
from qtracker.pipelines.data_science.nodes.interpretability import (
    get_explainer,
    run_shapash_app,
    visualize_feature_importance_plots,
)
from qtracker.pipelines.data_science.nodes.training import (
    get_covariates,
    get_outcome,
    get_sample_weights,
    train_model,
)


def create_training_pipeline() -> Pipeline:
    """Create the data science pipeline related to the training part

    Returns:
        Pipeline: The kedro training pipeline
    """
    training_pipeline = pipeline(
        pipe=[
            # Preprocessing
            node(
                name="get_train_covariates",
                func=get_covariates,
                inputs=dict(
                    data="train.final_dataset",
                    covariates="params:fields.covariates",
                    outcome="params:fields.outcome",
                    weight="params:fields.sample_weight",
                ),
                outputs="train.x",
                tags=["preprocessing"],
            ),
            node(
                name="get_test_covariates",
                func=get_covariates,
                inputs=dict(
                    data="test.final_dataset",
                    covariates="params:fields.covariates",
                    outcome="params:fields.outcome",
                    weight="params:fields.sample_weight",
                ),
                outputs="test.x",
                tags=["preprocessing"],
            ),
            node(
                name="get_train_outcome",
                func=get_outcome,
                inputs=dict(data="train.final_dataset", outcome_field="params:fields.outcome"),
                outputs="train.y_true",
                tags=["preprocessing"],
            ),
            node(
                name="get_test_outcome",
                func=get_outcome,
                inputs=dict(data="test.final_dataset", outcome_field="params:fields.outcome"),
                outputs="test.y_true",
                tags=["preprocessing"],
            ),
            node(
                name="get_train_weights",
                func=get_sample_weights,
                inputs=dict(data="train.final_dataset", sample_weight_field="params:fields.sample_weight"),
                outputs="train.sample_weights",
                tags=["preprocessing"],
            ),
            # Training
            node(
                name="train_model",
                func=train_model,
                inputs=dict(
                    x_train="train.x",
                    y_train="train.y_true",
                    weights_train="train.sample_weights",
                    classifier_name="params:model_info.classifier_name",
                    hyperparameters="params:model_info.hyperparameters",
                    random_state="params:random_state",
                ),
                outputs="trained_model",
                 tags=["model_training"],
            ),
            # Evaluation
            node(
                name="predict_train",
                func=infer_predictions,
                inputs=dict(model="trained_model", data="train.x"),
                outputs="train.y_score",
                tags=["evaluation"],
            ),
            node(
                name="predict_test",
                func=infer_predictions,
                inputs=dict(model="trained_model", data="test.x"),
                outputs="test.y_score",
                tags=["evaluation"],
            ),
            node(
                name="compute_metrics",
                func=compute_metrics,
                inputs=["train.y_true", "train.y_score", "test.y_true", "test.y_score"],
                outputs="metrics",
                tags=["evaluation"],
            ),
            node(
                name="visualize_metrics_plots",
                func=visualize_metrics_plots,
                inputs=[
                    "train.y_true",
                    "train.y_score",
                    "test.y_true",
                    "test.y_score",
                    "params:fields.outcome",
                    "params:model_info.classifier_name",
                ],
                outputs="metrics_plots",
                tags=["evaluation"],
            ),
            # Interpretability
            node(
                name="visualize_feature_importance_plots",
                func=visualize_feature_importance_plots,
                inputs=[
                    "trained_model",
                    "train.x",
                    "train.y_true",
                    "params:random_state",
                ],
                outputs="feature_importance_plots",
                tags=["interpretability"],
            ),
            node(
                name="get_explainer",
                func=get_explainer,
                inputs=["trained_model", "train.x", "train.y_true"],
                outputs="explainer",
                tags=["interpretability"],
            ),
            node(
                name="run_shapash_app",
                func=run_shapash_app,
                inputs="explainer",
                outputs=None,
                tags=["interpretability", "shapash_app"],
            ),
        ],
        tags=["training"],
    )
    return training_pipeline


def create_hyperparameters_tuning_pipeline() -> Pipeline:
    """Create the data science pipeline related to the hyperparameters tuning part

    Returns:
        Pipeline: The kedro hyperparameters tuning pipeline
    """
    hyperparameters_tuning_pipeline = pipeline(
        pipe=[
            # Preprocessing
            node(
                name="get_train_covariates",
                func=get_covariates,
                inputs=dict(
                    data="train.final_dataset",
                    covariates="params:fields.covariates",
                    outcome="params:fields.outcome",
                    weight="params:fields.sample_weight",
                ),
                outputs="train.x",
                tags=["preprocessing"],
            ),
            node(
                name="get_train_outcome",
                func=get_outcome,
                inputs=dict(data="train.final_dataset", outcome_field="params:fields.outcome"),
                outputs="train.y_true",
                tags=["preprocessing"],
            ),
            node(
                name="get_train_weights",
                func=get_sample_weights,
                inputs=dict(data="train.final_dataset", sample_weight_field="params:fields.sample_weight"),
                outputs="train.sample_weights",
                tags=["preprocessing"],
            ),
            # Optimization
            node(
                name="hyperparameters_tuning",
                func=hyperparameters_tuning,
                inputs=dict(
                    x_train="train.x",
                    y_train="train.y_true",
                    weights_train="train.sample_weights",
                    classifier_name="params:model_info.classifier_name",
                    hyperparameters_tuning_conf="params:hyperparameters_tuning_conf",
                    optimization_storage_uri="params:optimization_storage_uri",
                    random_state="params:random_state",
                ),
                outputs=[
                    "best_hyperparameters",
                    "hp_tuning_plots",
                ],
                tags=["model_tuning"],
            ),
        ],
        tags=["hyperparameters_tuning"],
    )
    return hyperparameters_tuning_pipeline


def create_inference_pipeline() -> Pipeline:
    """Create the data science pipeline related to the inference part

    Returns:
        Pipeline: The kedro inference pipeline
    """
    inference_pipeline = pipeline(
        pipe=[
            node(
                name="get_inference_covariates",
                func=get_covariates,
                inputs=dict(
                    data="inference.final_dataset",
                    covariates="params:fields.covariates",
                    outcome="params:fields.outcome",
                    weight="params:fields.sample_weight",
                ),
                outputs="inference.x",
                tags=["preprocessing"],
            ),
            node(
                name="inference.predict",
                func=infer_predictions,
                inputs=["trained_model", "inference.x"],
                outputs="inference.predictions",
            ),
        ],
        tags=["inference"],
    )
    return inference_pipeline
