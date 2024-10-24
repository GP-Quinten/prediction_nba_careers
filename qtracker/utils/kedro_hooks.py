"""script for the kedro hooks"""
import logging
import os
import pdb
import shutil
import sys
import traceback

import fsspec
import mlflow
from kedro.framework.hooks import hook_impl
from mlflow.utils.git_utils import get_git_branch, get_git_commit, get_git_repo_url

logger = logging.getLogger(__name__)


class MlflowTrackingHook:  # pylint: disable-msg=too-many-instance-attributes
    """
    These hooks ensure the git commit is correctly tracked is needed.

    This is necessary because by running the project with the `kedro run` command,
    mlflow is not able to retrieve the local git folder (its default behaviour is
    to look which file has been run with the python command)
    """

    def __init__(self, model_version, git_tracking=False):
        self.model_version = model_version
        self.git_tracking = git_tracking
        self.storage_env = None
        self.bucket_name_ = None
        self.folder_name_ = None
        self.mlflow_tracking_uri = None
        self.optimization_storage_uri = None
        self.active_run_id_ = None
        self.active_experiment_id_ = None
        self.hp_tuning_plots_path = None
        self.metrics_plots_path = None
        self.feature_importance_plots_path = None
        self.explainer_path = None

    @hook_impl
    def after_context_created(self, context):
        """
        Actions to take after the kedro context has been created

        Args:
            context: The kedro context

        """
        self.storage_env = context.config_loader.get("globals.yml")["storage_env"]
        self.mlflow_tracking_uri = context.config_loader.get("globals.yml")["mlflow_tracking_uri"]
        self.optimization_storage_uri = context.config_loader.get("parameters/hyperparameters_tuning.yml")[
            "optimization_storage_uri"
        ]
        self.hp_tuning_plots_path = context.config_loader.get("catalog.yml")["hp_tuning_plots"]["data_set"]["filepath"]
        self.metrics_plots_path = context.config_loader.get("catalog.yml")["metrics_plots"]["data_set"]["filepath"]
        self.feature_importance_plots_path = context.config_loader.get("catalog.yml")["feature_importance_plots"][
            "data_set"
        ]["filepath"]
        self.explainer_path = context.config_loader.get("catalog.yml")["explainer"]["filepath"]
        if self.storage_env == "cloud":
            self.bucket_name_ = context.config_loader.get("globals.yml")["bucket_name"]
            self.folder_name_ = context.config_loader.get("globals.yml")["folder_name"]

    # pylint: disable=no-self-use,unused-argument,too-few-public-methods
    @hook_impl
    def before_pipeline_run(self, run_params, pipeline, catalog):
        """
        Actions to take before each run of pipeline

        Args:
            run_params: The run parameters
            pipeline: The kedro pipeline to be run
            catalog: The kedro catalog

        """
        if run_params["pipeline_name"] == "training":
            shutil.rmtree(self.metrics_plots_path, onerror=FileNotFoundError)
            shutil.rmtree(self.feature_importance_plots_path, onerror=FileNotFoundError)
            if os.path.exists(self.explainer_path):
                os.remove(self.explainer_path)
        if mlflow.active_run():
            active_run_info = mlflow.active_run().info
            self.active_experiment_id_ = active_run_info.experiment_id
            self.active_run_id_ = active_run_info.run_id
            if self.git_tracking:
                path = run_params.get("project_path", ".")
                git_commit = get_git_commit(path)
                git_branch = get_git_branch(path)
                git_repo_url = get_git_repo_url(path)
                mlflow.set_tags(
                    {
                        # Set system tags
                        # See: https://www.mlflow.org/docs/latest/tracking.html#system-tags
                        "mlflow.source.git.commit": git_commit,
                        "mlflow.source.git.branch": git_branch,
                        "mlflow.source.git.repoURL": git_repo_url,
                        "mlflow.runName": git_commit,
                        # Repeat some tags to make them visible in the UI
                        "source.git.branch": git_branch,
                        "source.git.repoURL": git_repo_url,
                    }
                )
        if self.storage_env == "cloud" and self.active_experiment_id_ and self.active_run_id_:
            filesystem = fsspec.filesystem("s3")
            path = run_params.get("project_path", ".")
            # mlflow DB
            local_mlflow_db_path = os.path.join(path, self.mlflow_tracking_uri.split(":///")[-1])
            remote_mlflow_db_path = (
                f"""{self.bucket_name_}/{self.folder_name_}/{self.mlflow_tracking_uri.split(":///")[-1]}"""
            )
            filesystem.get(rpath=remote_mlflow_db_path, lpath=local_mlflow_db_path, recursive=False)
            # optimization DB
            local_optimization_db_path = os.path.join(path, self.optimization_storage_uri.split(":///")[-1])
            remote_optimization_db_path = (
                f"""{self.bucket_name_}/{self.folder_name_}/{self.optimization_storage_uri.split(":///")[-1]}"""
            )
            filesystem.get(rpath=remote_optimization_db_path, lpath=local_optimization_db_path, recursive=False)

    @hook_impl
    def after_pipeline_run(self, run_params, pipeline, catalog):
        """
        Actions to take after each run of pipeline

        Args:
            run_params: The run parameters
            pipeline: The kedro pipeline to be run
            catalog: The kedro catalog

        """
        shutil.rmtree(self.hp_tuning_plots_path, onerror=FileNotFoundError)
        if self.storage_env == "cloud" and self.active_experiment_id_ and self.active_run_id_:
            filesystem = fsspec.filesystem("s3")
            path = run_params.get("project_path", ".")
            logger.info("Saving the run %s of experiment %s", self.active_run_id_, self.active_experiment_id_)
            mlruns_folders = "mlruns"
            # local artifacts
            local_experiment_path = os.path.join(path, mlruns_folders, self.active_experiment_id_, self.active_run_id_)
            remote_experiment_path = os.path.join(
                self.bucket_name_, self.folder_name_, mlruns_folders, self.active_experiment_id_, self.active_run_id_
            )
            filesystem.put(lpath=local_experiment_path, rpath=remote_experiment_path, recursive=True)
            shutil.rmtree(local_experiment_path, onerror=FileNotFoundError)
            # mlflow DB
            local_mlflow_db_path = os.path.join(path, self.mlflow_tracking_uri.split(":///")[-1])
            remote_mlflow_db_path = (
                f"""{self.bucket_name_}/{self.folder_name_}/{self.mlflow_tracking_uri.split(":///")[-1]}"""
            )
            filesystem.put(lpath=local_mlflow_db_path, rpath=remote_mlflow_db_path, recursive=False)
            os.remove(local_mlflow_db_path)
            # optimization DB
            local_optimization_db_path = os.path.join(path, self.optimization_storage_uri.split(":///")[-1])
            remote_optimization_db_path = (
                f"""{self.bucket_name_}/{self.folder_name_}/{self.optimization_storage_uri.split(":///")[-1]}"""
            )
            filesystem.put(lpath=local_optimization_db_path, rpath=remote_optimization_db_path, recursive=False)
            os.remove(local_optimization_db_path)
            logger.info("Run saved to %s", remote_experiment_path)


class PDBNodeDebugHook:  # pylint:disable=too-few-public-methods
    """A hook class for creating a post mortem debugging with the PDB debugger
    whenever an error is triggered within a node. The local scope from when the
    exception occured is available within this debugging session.
    """

    @hook_impl
    def on_node_error(self) -> "PDBNodeDebugHook":  # pylint:disable=no-self-use
        """
        Action to take on node error to allow debugging

        """
        _, _, traceback_object = sys.exc_info()

        #  Print the traceback information for debugging ease
        traceback.print_tb(traceback_object)

        # Drop you into a post mortem debugging session
        pdb.post_mortem(traceback_object)  # pylint:disable=no-member


class PDBPipelineDebugHook:  # pylint:disable=too-few-public-methods
    """A hook class for creating a post mortem debugging with the PDB debugger
    whenever an error is triggered within a pipeline. The local scope from when the
    exception occured is available within this debugging session.
    """

    @hook_impl
    def on_pipeline_error(self) -> "PDBPipelineDebugHook":  # pylint:disable=no-self-use
        """
        Action to take on pipeline error to allow debugging

        """
        # We don't need the actual exception since it is within this stack frame
        _, _, traceback_object = sys.exc_info()

        #  Print the traceback information for debugging ease
        traceback.print_tb(traceback_object)

        # Drop you into a post mortem debugging session
        pdb.post_mortem(traceback_object)  # pylint:disable=no-member
