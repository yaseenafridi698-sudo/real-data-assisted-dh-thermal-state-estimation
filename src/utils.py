from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import PROJECT_ROOT


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_json(obj: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def synthetic_realistic_operating_data(
    config: dict[str, Any],
    dataset_name: str = "fallback_synthetic_realistic",
    n_steps: int = 192,
    freq: str | None = None,
) -> pd.DataFrame:
    """Generate fallback operating profiles for software testing only.

    The output imitates broad district-heating operating patterns but is not a
    substitute for real dataset results.
    """
    sys = config["system"]
    real = config["real_data"]
    freq = freq or real.get("resample_rule", "15min")
    rng = np.random.default_rng(config["dataset"].get("seed", 42))
    index = pd.date_range("2019-01-01", periods=n_steps, freq=freq, tz=real.get("timezone", "UTC"))
    t = np.arange(n_steps)
    daily = np.sin(2 * np.pi * t / max(1, int(pd.Timedelta("1D") / pd.Timedelta(freq))))
    weather = sys["ambient_base_C"] + 6 * np.sin(2 * np.pi * t / (4 * 96) - 1.2)
    heat_load_kw = 9500 + 1800 * (1 - daily) + 850 * np.maximum(0, 8 - weather) + rng.normal(0, 250, n_steps)
    heat_load_kw = np.clip(heat_load_kw, 2500, None)
    supply = sys["source_temp_base_C"] + 4.0 * np.maximum(0, 8 - weather) / 15 + rng.normal(0, 0.4, n_steps)
    return_temp = 48 + 0.00075 * heat_load_kw + 1.5 * np.sin(2 * np.pi * t / 96 + 0.8)
    return_temp += rng.normal(0, 0.35, n_steps)
    return pd.DataFrame(
        {
            "timestamp": index,
            "heat_load_kw": heat_load_kw,
            "supply_temp_C": supply,
            "return_temp_C": return_temp,
            "ambient_temp_C": weather,
            "source_dataset": dataset_name,
            "is_fallback_synthetic": True,
        }
    )


def manual_download_message(dataset_name: str, url: str) -> str:
    return (
        "Automatic download failed. Please manually download the dataset from "
        f"{url} and place the files in data/raw/{dataset_name}/."
    )
