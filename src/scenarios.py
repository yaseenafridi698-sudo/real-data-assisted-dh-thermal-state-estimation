from __future__ import annotations

from typing import Any

import pandas as pd

from .data_loaders import load_dataset_by_name
from .data_preprocessing import preprocess_dataset
from .data_registry import check_dataset_available
from .utils import synthetic_realistic_operating_data


def load_operating_dataframe(dataset_name: str, config: dict[str, Any], n_steps: int | None = None) -> tuple[pd.DataFrame, bool]:
    if check_dataset_available(dataset_name):
        df = preprocess_dataset(load_dataset_by_name(dataset_name), dataset_name, config)
        return (df.head(n_steps).copy() if n_steps else df, False)
    df = synthetic_realistic_operating_data(config, dataset_name="fallback_synthetic_realistic", n_steps=n_steps or 192)
    return preprocess_dataset(df, "fallback_synthetic_realistic", config, save=False), True
