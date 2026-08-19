from __future__ import annotations

from typing import Any

import pandas as pd
import torch

from .config import PROJECT_ROOT
from .data_loaders import load_dataset_by_name
from .data_preprocessing import preprocess_dataset
from .data_registry import check_dataset_available, processed_file_exists
from .study_workflow import run_xai4heat_validation


def load_xai4heat_if_available(config: dict[str, Any]) -> pd.DataFrame | None:
    processed_path = PROJECT_ROOT / "data" / "processed" / "xai4heat_processed.csv"
    if processed_file_exists("xai4heat") and processed_path.exists():
        return pd.read_csv(processed_path)
    if not check_dataset_available("xai4heat"):
        (PROJECT_ROOT / "results" / "XAI4HEAT_NOT_RUN.txt").write_text(
            "XAI4HEAT sparse-substation validation was not run because no local XAI4HEAT files were found in data/raw/xai4heat/.\n",
            encoding="utf-8",
        )
        return None
    return preprocess_dataset(load_dataset_by_name("xai4heat"), "xai4heat", config)


def validate_xai4heat_sparse_substations(
    sim: dict[str, Any],
    config: dict[str, Any],
    model: torch.nn.Module | None,
    stats: dict[str, Any],
) -> pd.DataFrame | None:
    """Evaluate measured-node consistency on XAI4HEAT when local files exist."""
    df = load_xai4heat_if_available(config)
    return run_xai4heat_validation(df, sim, config, model, stats)
