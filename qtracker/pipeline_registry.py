"""Project pipelines to be registered"""
from typing import Dict

from kedro.pipeline import Pipeline

from qtracker.pipelines import data_science


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a kedro `Pipeline` object.
    """

    hp_tuning_pipeline = data_science.create_hyperparameters_tuning_pipeline()
    training_pipeline = data_science.create_training_pipeline()
    training_pipeline = Pipeline([node for node in training_pipeline.nodes if "shapash_app" not in node.tags])
    shapash_pipeline = data_science.create_training_pipeline().only_nodes_with_tags("shapash_app")
    inference_pipeline = data_science.create_inference_pipeline()

    return {
        "hyperparameters_tuning": hp_tuning_pipeline,
        "training": training_pipeline,
        "shapash_app": shapash_pipeline,
        "inference": inference_pipeline,
    }
