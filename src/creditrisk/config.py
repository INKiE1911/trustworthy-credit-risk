"""root folder, config file, paths, MLflow, seeds"""

import random
from pathlib import Path

import numpy as np
import yaml


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    for folder in here.parents:
        if(folder/"pyproject.toml").exists():
            return folder
    raise FileNotFoundError(f"Could Not Found pyproject.toml In Any Folder Above {here}")


PROJECT_ROOT = _find_project_root()
CONFIG_FILE = PROJECT_ROOT / "configs" / "config.yaml" 


def load_config() -> dict:
    with open(CONFIG_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_path(key : str) -> Path:
    paths = load_config()["paths"]
    if key not in paths:
        raise KeyError(f"Unknown Path Key {key!r}. Valid Keys : {sorted(paths)}")

    path = PROJECT_ROOT / paths[key]
    path.mkdir(parents = True, exist_ok = True)
    return path


def setup_mlflow() -> str:
    import mlflow

    cfg = load_config()["mlflow"]
    uri = f"sqlite:///{PROJECT_ROOT / cfg['db_file']}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg["experiment"])
    return uri


def set_seed(seed: int | None = None) -> int:
    if seed is None:
        seed = load_config()["project"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    return seed 


