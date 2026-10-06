"""Tests for creditrisk.config."""

import numpy as np
import pytest

from creditrisk.config import PROJECT_ROOT, get_path, load_config, set_seed, setup_mlflow

RAW_DIR = PROJECT_ROOT / "data" / "raw"


def test_root_has_pyproject():
    assert (PROJECT_ROOT / "pyproject.toml").exists()


def test_config_has_sections():
    cfg = load_config()
    for section in ["project", "paths", "data", "splits", "mlflow"]:
        assert section in cfg, f"missing section: {section}"


def test_get_path_is_absolute_and_created():
    path = get_path("interim")
    assert path.is_absolute()
    assert path.exists()
    assert path == PROJECT_ROOT / "data" / "interim"


def test_get_path_unknown_key_raises():
    with pytest.raises(KeyError):
        get_path("does_not_exist")


def test_set_seed_is_repeatable():
    set_seed(123)
    a = np.random.rand(3)
    set_seed(123)
    b = np.random.rand(3)
    assert np.array_equal(a, b)


def test_set_seed_uses_config_by_default():
    assert set_seed() == load_config()["project"]["seed"]


def test_mlflow_points_to_project_root():
    assert setup_mlflow() == f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"


@pytest.mark.skipif(not RAW_DIR.exists(), reason="data not downloaded (e.g. on GitHub CI)")
def test_raw_csv_files_exist():
    tables = load_config()["data"]["tables"]
    missing = [name for name in tables.values() if not (RAW_DIR / name).exists()]
    assert not missing, f"missing CSV files: {missing}"