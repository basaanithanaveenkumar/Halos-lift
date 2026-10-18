"""ENCODERS / TASKS registries and name-based encoder specs in the model factories."""

from __future__ import annotations

import pytest

from lifting.encoders import ENCODERS, SimpleConvEncoder
from lifting.exceptions import RegistryError
from lifting.geometry import GridSpec
from lifting.models import build_bev_model, resolve_encoder
from lifting.training import TASKS, BEVSegmentationTask, OccupancyTask


def test_builtin_names() -> None:
    assert {"simple_conv", "resnet", "hale_backbone"} <= set(ENCODERS.names())
    assert set(TASKS.names()) == {"bev_segmentation", "occupancy"}


def test_tasks_build_from_name_and_mapping() -> None:
    assert isinstance(TASKS.build("bev_segmentation", num_classes=4), BEVSegmentationTask)
    task = TASKS.build({"type": "occupancy", "num_classes": 3, "aux_weight": 0.2})
    assert isinstance(task, OccupancyTask) and task.aux_weight == 0.2


def test_unknown_names_suggest_close_matches() -> None:
    with pytest.raises(RegistryError, match="simple_conv"):
        ENCODERS.get("simple_cnv")


def test_resolve_encoder_accepts_instance_name_mapping_and_none() -> None:
    inst = SimpleConvEncoder(out_channels=8)
    assert resolve_encoder(inst, 16) is inst
    assert resolve_encoder(None, 16).out_channels == 16
    assert resolve_encoder("simple_conv", 12).out_channels == 12
    assert resolve_encoder({"type": "simple_conv", "num_levels": 1}, 8).num_levels == 1


def test_build_bev_model_with_encoder_name() -> None:
    model = build_bev_model(
        "simple_bev",
        GridSpec(resolution=(8, 8, 2)),
        encoder="simple_conv",
        channels=8,
        image_size=(32, 64),
        num_cameras=2,
    )
    assert model.encoder.out_channels == 8
