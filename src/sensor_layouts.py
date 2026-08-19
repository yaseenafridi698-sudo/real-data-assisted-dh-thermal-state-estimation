from __future__ import annotations

from typing import Any

import numpy as np


def sensor_nodes_for_layout(layout_name: str, n_nodes: int, rng: np.random.Generator | None = None) -> list[int]:
    rng = rng or np.random.default_rng(42)
    if layout_name == "S1_inlet_only":
        return [0]
    if layout_name in {"S10_noisy_inlet_only", "S11_noisy_inlet_only", "S13_noisy_inlet_only"}:
        return [0]
    if layout_name in {"S11_outlet_only", "S12_outlet_only", "S14_outlet_only"}:
        return [n_nodes - 1]
    if layout_name in {"S2_inlet_outlet", "S5_noisy_inlet_outlet"}:
        return [0, n_nodes - 1]
    if layout_name in {"S12_noisy_inlet_outlet_5pct", "S13_noisy_inlet_outlet_5pct", "S15_noisy_inlet_outlet_5pct"}:
        return [0, n_nodes - 1]
    if layout_name in {"S11_middle_only", "S13_middle_only", "S14_middle_only"}:
        return [n_nodes // 2]
    if layout_name == "S12_inlet_two_middle_outlet":
        return sorted(set([0, n_nodes // 3, 2 * n_nodes // 3, n_nodes - 1]))
    if layout_name == "S3_inlet_middle_outlet":
        return [0, n_nodes // 2, n_nodes - 1]
    if layout_name in {"S4_five_sensors", "S6_dropout_five_sensors", "S14_peak_dropout_five_sensors", "S15_peak_dropout_five_sensors", "S16_peak_dropout_five_sensors"}:
        return sorted(set([0, n_nodes // 4, n_nodes // 2, 3 * n_nodes // 4, n_nodes - 1]))
    if layout_name in {"S7_xai4heat_substations", "S7_xai4heat_style_substations"}:
        return sorted(set(np.linspace(1, n_nodes - 2, min(5, max(1, n_nodes - 2))).round().astype(int).tolist()))
    if layout_name == "S8_random_three_sensors":
        nodes = [0, n_nodes - 1]
        interior = rng.choice(np.arange(1, n_nodes - 1), size=1, replace=False).tolist()
        return sorted(nodes + interior)
    if layout_name == "S9_optimized_three_sensors":
        return [0, n_nodes // 2, n_nodes - 1]
    if layout_name in {"S15_optimized_two_sensors", "S16_optimized_two_sensors", "S17_optimized_two_sensors"}:
        return [0, n_nodes - 1]
    if layout_name in {"S10_optimized_five_sensors", "S16_optimized_five_sensors"}:
        return sorted(set([0, n_nodes // 4, n_nodes // 2, 3 * n_nodes // 4, n_nodes - 1]))
    raise KeyError(f"Unknown sensor layout: {layout_name}")


def apply_sensor_layout(
    sim: dict[str, Any],
    layout_name: str,
    config: dict[str, Any],
    noise_std_fraction: float = 0.0,
    sparse_real_measurements: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(config["dataset"].get("seed", 42))
    states = np.stack([sim["Ts"], sim["Tr"], sim["H"], sim["q"]], axis=-1)
    measurements = np.zeros_like(states)
    masks = np.zeros_like(states)
    n_steps, n_nodes, n_vars = states.shape
    nodes = sensor_nodes_for_layout(layout_name, n_nodes, rng)

    if layout_name in {"S7_xai4heat_substations", "S7_xai4heat_style_substations"} and sparse_real_measurements and sparse_real_measurements.get("available"):
        for node in sparse_real_measurements["node_map"].values():
            if 0 <= node < n_nodes:
                nodes.append(int(node))
        nodes = sorted(set(nodes))
    if "optimized_sensor_nodes" in sim and layout_name == "S9_optimized_three_sensors":
        nodes = [int(n) for n in sim["optimized_sensor_nodes"]]
    if "optimized_sensor_nodes_2" in sim and layout_name in {"S15_optimized_two_sensors", "S16_optimized_two_sensors", "S17_optimized_two_sensors"}:
        nodes = [int(n) for n in sim["optimized_sensor_nodes_2"]]
    if "optimized_sensor_nodes_5" in sim and layout_name in {"S10_optimized_five_sensors", "S16_optimized_five_sensors"}:
        nodes = [int(n) for n in sim["optimized_sensor_nodes_5"]]

    masks[:, nodes, :] = 1.0
    measurements[:, nodes, :] = states[:, nodes, :]

    if layout_name in {"S5_noisy_inlet_outlet", "S10_noisy_inlet_only", "S11_noisy_inlet_only", "S13_noisy_inlet_only"} or noise_std_fraction > 0:
        std = np.nanstd(states, axis=(0, 1))
        noise = rng.normal(0.0, noise_std_fraction * np.maximum(std, 1e-6), size=states.shape)
        measurements = measurements + noise * masks
    if layout_name in {"S12_noisy_inlet_outlet_5pct", "S13_noisy_inlet_outlet_5pct", "S15_noisy_inlet_outlet_5pct"}:
        std = np.nanstd(states, axis=(0, 1))
        noise = rng.normal(0.0, 0.05 * np.maximum(std, 1e-6), size=states.shape)
        measurements = measurements + noise * masks

    if layout_name == "S6_dropout_five_sensors":
        dropout = rng.random(size=masks.shape[:2]) < 0.18
        masks[dropout, :] = 0.0
        measurements = measurements * masks
    if layout_name in {"S14_peak_dropout_five_sensors", "S15_peak_dropout_five_sensors", "S16_peak_dropout_five_sensors"}:
        peak_threshold = np.nanpercentile(sim["Q_load"], 75) if "Q_load" in sim else np.inf
        peak_steps = np.asarray(sim.get("Q_load", np.zeros(n_steps))) >= peak_threshold
        dropout = (rng.random(size=masks.shape[:2]) < 0.35) & peak_steps[:, None]
        masks[dropout, :] = 0.0
        measurements = measurements * masks

    return {
        "layout_name": layout_name,
        "sensor_nodes": nodes,
        "measurements": measurements,
        "masks": masks,
        "variables": ["Ts", "Tr", "H", "q"],
    }


def layout_table_rows(n_nodes: int) -> list[dict[str, str]]:
    rows = []
    for name in [
        "S1_inlet_only",
        "S2_inlet_outlet",
        "S3_inlet_middle_outlet",
        "S4_five_sensors",
        "S5_noisy_inlet_outlet",
        "S6_dropout_five_sensors",
        "S7_xai4heat_substations",
        "S8_random_three_sensors",
        "S9_optimized_three_sensors",
        "S10_optimized_five_sensors",
        "S11_middle_only",
        "S12_inlet_two_middle_outlet",
        "S13_noisy_inlet_only",
        "S14_outlet_only",
        "S15_noisy_inlet_outlet_5pct",
        "S16_peak_dropout_five_sensors",
        "S17_optimized_two_sensors",
    ]:
        rows.append({"layout": name, "representative_nodes": ", ".join(map(str, sensor_nodes_for_layout(name, n_nodes)))})
    return rows
