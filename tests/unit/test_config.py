"""ConfigLoader inheritance / overrides and the typed TrainConfig boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from lifting.config import ConfigLoader, TrainConfig
from lifting.exceptions import InvalidConfigError

pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[2]


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_base_merge_and_override(tmp_path: Path) -> None:
    _write(tmp_path / "base.yaml", "a: {x: 1, y: 2}\nb: 3\n")
    child = _write(tmp_path / "child.yaml", "_base_: base.yaml\na: {y: 20}\n")
    assert ConfigLoader.load(child) == {"a": {"x": 1, "y": 20}, "b": 3}


def test_legacy_inherits_key_and_base_list(tmp_path: Path) -> None:
    _write(tmp_path / "p.yaml", "a: 1\nb: 1\n")
    _write(tmp_path / "q.yaml", "b: 2\nc: 2\n")
    child = _write(tmp_path / "c.yaml", "inherits: [p.yaml, q.yaml]\nc: 3\n")
    assert ConfigLoader.load(child) == {"a": 1, "b": 2, "c": 3}


def test_delete_replaces_instead_of_merging(tmp_path: Path) -> None:
    _write(tmp_path / "base.yaml", "a: {x: 1, y: 2}\n")
    child = _write(tmp_path / "child.yaml", "_base_: base.yaml\na: {_delete_: true, z: 9}\n")
    assert ConfigLoader.load(child) == {"a": {"z": 9}}


def test_base_cycle_is_detected(tmp_path: Path) -> None:
    _write(tmp_path / "a.yaml", "_base_: b.yaml\n")
    _write(tmp_path / "b.yaml", "_base_: a.yaml\n")
    with pytest.raises(InvalidConfigError, match="cycle"):
        ConfigLoader.load(tmp_path / "a.yaml")


@pytest.mark.parametrize(
    ("override", "expected"),
    [("model.depth=12", 12), ("model.name=abc", "abc"), ("model.dims=[1, 2]", [1, 2])],
)
def test_dotted_overrides_parse_values(override: str, expected: object) -> None:
    result = ConfigLoader.apply_overrides({"model": {"depth": 1}}, [override])
    assert result["model"][override.split("=")[0].split(".")[1]] == expected


@pytest.mark.parametrize("bad", ["novalue", "=3"])
def test_malformed_override_raises(bad: str) -> None:
    with pytest.raises(InvalidConfigError):
        ConfigLoader.apply_overrides({}, [bad])


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(InvalidConfigError, match="cannot read"):
        ConfigLoader.load(tmp_path / "nope.yaml")


def test_shipped_train_configs_validate() -> None:
    plain = TrainConfig.from_mapping(ConfigLoader.load(REPO / "configs" / "train_synthetic.yaml"))
    tri = TrainConfig.from_mapping(ConfigLoader.load(REPO / "configs" / "train_triplane.yaml"))
    assert plain.model == "bev"
    assert (tri.model, tri.lifter, tri.steps, tri.channels) == ("triplane", "tpvformer", 400, 32)


def test_train_config_rejects_unknown_key_and_bad_values() -> None:
    with pytest.raises(InvalidConfigError, match="unknown key"):
        TrainConfig.from_mapping({"train": {"lrr": 1.0}})
    with pytest.raises(InvalidConfigError):
        TrainConfig.from_mapping({"train": {"lr": -1.0}})
    with pytest.raises(InvalidConfigError):
        TrainConfig.from_mapping({"model": {"kind": "voxel"}})


def test_cli_config_sets_defaults_and_flags_override() -> None:
    from lifting.train_cli import parse_args

    cfg = str(REPO / "configs" / "train_synthetic.yaml")
    args = parse_args(["--config", cfg, "--cfg-options", "train.steps=7", "--lr", "0.5"])
    assert (args.steps, args.lr, args.resolution) == (7, 0.5, (32, 32, 4))
    with pytest.raises(InvalidConfigError):
        parse_args(["--cfg-options", "train.steps=7"])
