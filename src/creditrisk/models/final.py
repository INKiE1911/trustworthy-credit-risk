"""Save and load the final model that every later step (and the app) uses."""

import json
from datetime import date
from pathlib import Path

import joblib

from creditrisk.config import PROJECT_ROOT

MODEL_DIR = PROJECT_ROOT / "models"


def save_final_model(model, feature_columns, info: dict, model_dir=None) -> Path:
    """Write the model (joblib), LightGBM's own text format, and a JSON with the details.

    The JSON records the exact feature order; predictions must use the same columns.
    """
    folder = Path(model_dir) if model_dir else MODEL_DIR
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, folder / "lightgbm_final.joblib")
    if hasattr(model, "booster_"):
        model.booster_.save_model(str(folder / "lightgbm_final.txt"))
    details = {**info, "saved_on": date.today().isoformat(), "features": list(feature_columns)}
    (folder / "lightgbm_final.json").write_text(json.dumps(details, indent=2))
    return folder


def load_final_model(model_dir=None):
    """Return (model, details). details["features"] is the column order the model expects."""
    folder = Path(model_dir) if model_dir else MODEL_DIR
    model = joblib.load(folder / "lightgbm_final.joblib")
    details = json.loads((folder / "lightgbm_final.json").read_text())
    return model, details
