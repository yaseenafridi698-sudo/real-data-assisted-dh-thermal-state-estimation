from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .graph_utils import node_positions


STATE_NAMES = ["Ts", "Tr", "H", "q"]


def build_state_arrays(sim: dict[str, Any], sensors: dict[str, Any], config: dict[str, Any]) -> dict[str, np.ndarray]:
    target = np.stack([sim["Ts"], sim["Tr"], sim["H"], sim["q"]], axis=-1).astype(np.float32)
    sensor_values = sensors["measurements"].astype(np.float32)
    sensor_masks = sensors["masks"].astype(np.float32)
    T, N, _ = target.shape
    x = node_positions(N).astype(np.float32)
    features = np.zeros((T, N, 14), dtype=np.float32)
    features[:, :, 0:4] = sensor_values
    features[:, :, 4:8] = sensor_masks
    features[:, :, 8] = x[None, :]
    features[:, :, 9] = np.asarray(sim["Ta"], dtype=np.float32)[:, None]
    features[:, :, 10] = np.asarray(sim["T_source"], dtype=np.float32)[:, None]
    features[:, :, 11] = np.asarray(sim["alpha"], dtype=np.float32)[:, None]
    features[:, :, 12] = (np.asarray(sim["Q_load"], dtype=np.float32) / 1000.0)[:, None]
    features[:, :, 13] = 0.0
    return {
        "features": features,
        "target": target,
        "sensor_values": sensor_values,
        "sensor_masks": sensor_masks,
        "ambient": np.asarray(sim["Ta"], dtype=np.float32),
        "source_temp": np.asarray(sim["T_source"], dtype=np.float32),
        "alpha": np.asarray(sim["alpha"], dtype=np.float32),
        "heat_load_kw": (np.asarray(sim["Q_load"], dtype=np.float32) / 1000.0),
        "time_s": np.asarray(sim["time_s"], dtype=np.float32),
        "x_m": np.asarray(sim["x_m"], dtype=np.float32),
        "trajectory_start": np.asarray(sim.get("trajectory_start", np.r_[True, np.zeros(max(T - 1, 0), dtype=bool)]), dtype=bool),
        "trajectory_id": np.asarray(sim.get("trajectory_id", np.cumsum(np.asarray(sim.get("trajectory_start", np.r_[True, np.zeros(max(T - 1, 0), dtype=bool)]), dtype=bool)) - 1), dtype=np.int32),
    }


def split_window_indices(
    n_steps: int,
    window_steps: int,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    embargo_steps: int = 0,
    valid_window_starts: list[int] | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Chronological window split with optional gap-safe starts and embargoes.

    The paper protocol first removes windows that cross a trajectory restart,
    assigns 70%/15% of the remaining starts to train/validation, and places an
    embargo of ``window_steps - 1`` starts between adjacent partitions.  This
    prevents overlapping samples from leaking across chronological partitions.
    """
    n_windows = max(0, int(n_steps) - int(window_steps) + 1)
    starts = list(range(n_windows)) if valid_window_starts is None else sorted({int(i) for i in valid_window_starts if 0 <= int(i) < n_windows})
    if not starts:
        return [], [], []
    n_train = int(float(train_fraction) * len(starts))
    n_val = int(float(val_fraction) * len(starts))
    n_train = max(1, min(n_train, len(starts)))
    n_val = max(0, min(n_val, max(0, len(starts) - n_train)))
    embargo = max(0, int(embargo_steps))
    train = starts[:n_train]
    val_begin = min(len(starts), n_train + embargo)
    val_end = min(len(starts), val_begin + n_val)
    val = starts[val_begin:val_end]
    test_begin = min(len(starts), val_end + embargo)
    test = starts[test_begin:]
    return train, val, test



def contiguous_window_starts(trajectory_start: np.ndarray, window_steps: int) -> list[int]:
    """Return window starts that do not cross a trajectory break.

    ``trajectory_start[t]`` marks the first sample of a contiguous trajectory.
    A window may begin on such a marker, but it is invalid if another marker
    occurs inside the remaining samples of the window.
    """
    starts = np.asarray(trajectory_start, dtype=bool).reshape(-1)
    window_steps = int(window_steps)
    if window_steps < 1:
        raise ValueError("window_steps must be >= 1")
    if starts.size < window_steps:
        return []
    valid: list[int] = []
    for i in range(starts.size - window_steps + 1):
        if not starts[i + 1 : i + window_steps].any():
            valid.append(i)
    return valid


def _compute_stats(arrays: dict[str, np.ndarray], indices: list[int], window_steps: int) -> dict[str, np.ndarray]:
    windows = np.stack([arrays["target"][i : i + window_steps] for i in indices], axis=0)
    target_mean = windows.mean(axis=(0, 1, 2))
    target_std = windows.std(axis=(0, 1, 2)) + 1e-6
    feature_windows = np.stack([arrays["features"][i : i + window_steps] for i in indices], axis=0)
    feature_mean = np.zeros(feature_windows.shape[-1], dtype=np.float32)
    feature_std = np.ones(feature_windows.shape[-1], dtype=np.float32)
    continuous = [0, 1, 2, 3, 9, 10, 11, 12]
    feature_mean[continuous] = feature_windows[..., continuous].mean(axis=(0, 1, 2))
    feature_std[continuous] = feature_windows[..., continuous].std(axis=(0, 1, 2)) + 1e-6
    return {
        "target_mean": target_mean.astype(np.float32),
        "target_std": target_std.astype(np.float32),
        "feature_mean": feature_mean.astype(np.float32),
        "feature_std": feature_std.astype(np.float32),
    }


class StateWindowDataset(Dataset):
    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        indices: list[int],
        config: dict[str, Any],
        stats: dict[str, np.ndarray] | None = None,
        fit_stats: bool = False,
    ) -> None:
        self.arrays = arrays
        self.indices = indices
        self.window_steps = int(config["model"]["window_steps"])
        if fit_stats:
            self.stats = _compute_stats(arrays, indices, self.window_steps)
        elif stats is not None:
            self.stats = stats
        else:
            raise ValueError("Provide stats or set fit_stats=True.")

    def __len__(self) -> int:
        return len(self.indices)

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        return (features - self.stats["feature_mean"]) / self.stats["feature_std"]

    def _normalize_target(self, target: np.ndarray) -> np.ndarray:
        return (target - self.stats["target_mean"]) / self.stats["target_std"]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = self.indices[idx]
        end = start + self.window_steps
        x = self._normalize_features(self.arrays["features"][start:end])
        y = self._normalize_target(self.arrays["target"][start:end])
        item = {
            "x": torch.tensor(x, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32),
            "target_physical": torch.tensor(self.arrays["target"][start:end], dtype=torch.float32),
            "sensor_values": torch.tensor(self.arrays["sensor_values"][start:end], dtype=torch.float32),
            "sensor_masks": torch.tensor(self.arrays["sensor_masks"][start:end], dtype=torch.float32),
            "ambient": torch.tensor(self.arrays["ambient"][start:end], dtype=torch.float32),
            "source_temp": torch.tensor(self.arrays["source_temp"][start:end], dtype=torch.float32),
            "alpha": torch.tensor(self.arrays["alpha"][start:end], dtype=torch.float32),
            "heat_load_kw": torch.tensor(self.arrays["heat_load_kw"][start:end], dtype=torch.float32),
            "time_s": torch.tensor(self.arrays["time_s"][start:end], dtype=torch.float32),
        }
        return item


def denormalize_state(tensor: torch.Tensor, stats: dict[str, np.ndarray | torch.Tensor], device: torch.device | None = None) -> torch.Tensor:
    device = device or tensor.device
    mean = torch.as_tensor(stats["target_mean"], dtype=tensor.dtype, device=device)
    std = torch.as_tensor(stats["target_std"], dtype=tensor.dtype, device=device)
    return tensor * std + mean
