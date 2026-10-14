"""Notebook helpers for the triplane (TPV) lifting notebooks - one class per file.

Not part of the ``lifting`` package: these only draw figures and run small experiments on top
of the public ``lifting.triplane`` API. Notebooks import them with::

    import sys; sys.path.insert(0, ".")        # run from notebooks/triplane/
    from tpv_kit import TriplaneScene, PlanePlotter, OccupancyExperiment, ...
"""

from tpv_kit.contract_checker import TriplaneContractChecker
from tpv_kit.experiment_result import ExperimentResult
from tpv_kit.occupancy_experiment import OccupancyExperiment
from tpv_kit.pipeline_diagram import PipelineDiagram
from tpv_kit.plane_plotter import PlanePlotter
from tpv_kit.storyboard_viewer import StoryboardViewer
from tpv_kit.triplane_scene import TriplaneScene

__all__ = [
    "ExperimentResult",
    "OccupancyExperiment",
    "PipelineDiagram",
    "PlanePlotter",
    "StoryboardViewer",
    "TriplaneContractChecker",
    "TriplaneScene",
]
