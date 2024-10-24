"""Project settings. There is no need to edit this file unless you want to change values
from the Kedro defaults. For further information, including these default values, see
https://kedro.readthedocs.io/en/stable/kedro_project_setup/settings.html."""

from kedro.config import TemplatedConfigLoader

from qtracker import __version__ as VERSION
from qtracker.utils.kedro_hooks import (
    MlflowTrackingHook,
    PDBNodeDebugHook,
    PDBPipelineDebugHook,
)

# Instantiated project hooks.
DEBUG_MODE = False

if DEBUG_MODE:
    HOOKS = (
        MlflowTrackingHook(model_version=VERSION, git_tracking=False),
        PDBNodeDebugHook(),
        PDBPipelineDebugHook(),
    )
else:
    HOOKS = (MlflowTrackingHook(model_version=VERSION, git_tracking=False),)

# Class that manages how configuration is loaded.
# pylint:disable=invalid-name
CONFIG_LOADER_CLASS = TemplatedConfigLoader
# Keyword arguments to pass to the `CONFIG_LOADER_CLASS` constructor.
CONFIG_LOADER_ARGS = {
    "globals_pattern": "*globals.yml",
    "globals_dict": {},
}
