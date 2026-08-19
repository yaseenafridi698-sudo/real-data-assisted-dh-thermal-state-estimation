from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .dataset import denormalize_state


def _pump_head(alpha: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    sys = config["system"]
    return sys["pump_c1"] * alpha**2 + sys["pump_c2"] * alpha + sys["pump_c3"]


def physics_informed_loss(
    pred_norm: torch.Tensor,
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
    stats: dict[str, Any],
    loss_weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    trn = dict(config["training"])
    if loss_weights:
        trn.update(loss_weights)
    scales = trn.get("physics_loss_scales", {}) or {}

    def scale_for(name: str) -> float:
        value = scales.get(name, 1.0)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 1.0
        return max(value, 1e-12)

    pred = denormalize_state(pred_norm, stats, device=pred_norm.device)
    target_norm = batch["y"]
    target = batch["target_physical"]
    node_weights = torch.ones_like(pred_norm[..., :1])
    node_weights[:, :, 0, :] = node_weights[:, :, 0, :] + 0.5
    node_weights[:, :, -1, :] = node_weights[:, :, -1, :] + 1.0
    sensor_node_weight = (batch["sensor_masks"].amax(dim=-1, keepdim=True) > 0.5).to(pred_norm.dtype)
    node_weights = node_weights + 0.75 * sensor_node_weight
    if target.shape[1] > 1:
        transient = torch.zeros_like(node_weights)
        d_ts = torch.abs(target[:, 1:, :, 0:1] - target[:, :-1, :, 0:1])
        transient[:, 1:, :, :] = d_ts / torch.clamp(torch.nanmean(d_ts), min=1e-6)
        node_weights = node_weights + 0.25 * torch.clamp(transient, max=4.0)
    state_mse = torch.sum(node_weights * (pred_norm - target_norm) ** 2) / torch.clamp(node_weights.sum() * pred_norm.shape[-1], min=1.0)

    sensor_mask = batch["sensor_masks"]
    sensor_values = batch["sensor_values"]
    if sensor_mask.sum() > 0:
        sensor_mse = torch.sum(((pred - sensor_values) * sensor_mask) ** 2) / torch.clamp(sensor_mask.sum(), min=1.0)
    else:
        sensor_mse = torch.zeros((), device=pred.device)

    sys = config["system"]
    dt = float(sys["dt_s"])
    dx = float(sys["dx_m"])
    rho = float(sys["rho"])
    cp = float(sys["cp"])
    diameter = float(sys["diameter_m"])
    area = torch.tensor(3.141592653589793 * diameter**2 / 4.0, dtype=pred.dtype, device=pred.device)
    U = float(sys["heat_loss_U_W_m2K"])
    perimeter = float(sys["pipe_perimeter_m"])
    K_loss = U * perimeter / max(rho * cp * float(area.detach().cpu()), 1e-12)

    Ts = pred[..., 0]
    Tr = pred[..., 1]
    H = pred[..., 2]
    q = torch.clamp(pred[..., 3], min=1e-4)
    Ta = batch["ambient"]
    alpha = batch["alpha"]
    source_temp = batch["source_temp"]

    if pred.shape[1] > 1 and pred.shape[2] > 2:
        v = q / area
        dTs_dt = (Ts[:, 1:, 1:] - Ts[:, :-1, 1:]) / dt
        adv_Ts = v[:, :-1, 1:] * (Ts[:, :-1, 1:] - Ts[:, :-1, :-1]) / dx
        r_Ts = dTs_dt + adv_Ts + K_loss * (Ts[:, :-1, 1:] - Ta[:, :-1, None])

        dTr_dt = (Tr[:, 1:, :-1] - Tr[:, :-1, :-1]) / dt
        adv_Tr = -v[:, :-1, :-1] * (Tr[:, :-1, 1:] - Tr[:, :-1, :-1]) / dx
        r_Tr = dTr_dt + adv_Tr + K_loss * (Tr[:, :-1, :-1] - Ta[:, :-1, None])
        raw_thermal_residual = torch.mean(r_Ts**2) + torch.mean(r_Tr**2)
        raw_smoothness = torch.mean((pred[:, 1:] - pred[:, :-1]) ** 2) * 0.01 + torch.mean((pred[:, :, 1:] - pred[:, :, :-1]) ** 2) * 0.01
        thermal_residual = raw_thermal_residual / scale_for("thermal_residual_scale")
        smoothness = raw_smoothness / scale_for("smoothness_scale")
    else:
        thermal_residual = torch.zeros((), device=pred.device)
        smoothness = torch.zeros((), device=pred.device)

    pump_drop = _pump_head(alpha, config) - float(sys["outlet_head_m"])
    pred_drop = H[:, :, 0] - H[:, :, -1]
    raw_hydraulic_residual = F.mse_loss(pred_drop / 100.0, pump_drop / 100.0)
    hydraulic_residual = raw_hydraulic_residual / scale_for("hydraulic_residual_scale")

    load_kw = batch["heat_load_kw"]
    load_scale = torch.clamp(torch.nanmean(torch.abs(load_kw)), min=1.0)
    pred_supply_seg = 0.5 * (Ts[..., :-1] + Ts[..., 1:])
    pred_return_seg = 0.5 * (Tr[..., :-1] + Tr[..., 1:])
    target_supply_seg = 0.5 * (target[..., :-1, 0] + target[..., 1:, 0])
    target_return_seg = 0.5 * (target[..., :-1, 1] + target[..., 1:, 1])
    pred_heat_loss = U * perimeter * dx * torch.sum(
        (pred_supply_seg - Ta[..., None]) + (pred_return_seg - Ta[..., None]), dim=-1
    ) / 1000.0
    target_heat_loss = U * perimeter * dx * torch.sum(
        (target_supply_seg - Ta[..., None]) + (target_return_seg - Ta[..., None]), dim=-1
    ) / 1000.0
    heat_scale = torch.clamp(torch.mean(torch.abs(target_heat_loss)), min=1.0)
    raw_heat_loss_residual = F.mse_loss(pred_heat_loss / heat_scale, target_heat_loss / heat_scale)
    heat_loss_residual = raw_heat_loss_residual / scale_for("heat_loss_scale")

    source_heat_kw = rho * cp * q[:, :, 0] * (Ts[:, :, 0] - Tr[:, :, 0]) / 1000.0
    delivered_kw = rho * cp * q[:, :, -1] * torch.clamp(Ts[:, :, -1] - Tr[:, :, -1], min=0.0) / 1000.0
    total_temperature = Ts + Tr
    trapz_temperature = 0.5 * total_temperature[..., 0] + torch.sum(total_temperature[..., 1:-1], dim=-1) + 0.5 * total_temperature[..., -1]
    pipe_energy_kj = rho * cp * float(area.detach().cpu()) * dx * trapz_temperature / 1000.0
    storage_kw = torch.zeros_like(pipe_energy_kj)
    if pipe_energy_kj.shape[1] > 1:
        storage_kw[:, 1:] = (pipe_energy_kj[:, 1:] - pipe_energy_kj[:, :-1]) / dt
        storage_kw[:, 0] = storage_kw[:, 1]
    physical_energy_residual_kw = source_heat_kw - delivered_kw - pred_heat_loss - storage_kw
    raw_energy_residual = torch.mean((physical_energy_residual_kw / load_scale) ** 2)
    energy_residual = raw_energy_residual / scale_for("energy_residual_scale")

    boundary_source = F.mse_loss(Ts[:, :, 0], source_temp)
    outlet_return_mask = sensor_mask[:, :, -1, 1]
    if outlet_return_mask.sum() > 0:
        outlet_return = torch.sum(((Tr[:, :, -1] - sensor_values[:, :, -1, 1]) * outlet_return_mask) ** 2) / torch.clamp(outlet_return_mask.sum(), min=1.0)
    else:
        outlet_return = F.mse_loss(Tr[:, :, -1], target[:, :, -1, 1])
    raw_boundary_residual = boundary_source + 0.25 * outlet_return + raw_hydraulic_residual
    boundary_residual = raw_boundary_residual / scale_for("boundary_residual_scale")

    total = (
        trn["lambda_state"] * state_mse
        + trn["lambda_sensor"] * sensor_mse
        + trn["lambda_thermal"] * thermal_residual
        + trn["lambda_hydraulic"] * hydraulic_residual
        + trn["lambda_boundary"] * boundary_residual
        + trn.get("lambda_energy", 0.0) * energy_residual
        + trn.get("lambda_heat_loss", 0.0) * heat_loss_residual
        + trn["lambda_smooth"] * smoothness
    )
    logs = {
        "state_mse": float(state_mse.detach().cpu()),
        "real_sensor_mse": float(sensor_mse.detach().cpu()),
        "thermal_residual": float(thermal_residual.detach().cpu()),
        "hydraulic_residual": float(hydraulic_residual.detach().cpu()),
        "boundary_residual": float(boundary_residual.detach().cpu()),
        "energy_residual": float(energy_residual.detach().cpu()),
        "heat_loss_residual": float(heat_loss_residual.detach().cpu()),
        "smoothness": float(smoothness.detach().cpu()),
        "total_loss": float(total.detach().cpu()),
    }
    return total, logs


def mse_only_loss(pred_norm: torch.Tensor, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
    state_mse = F.mse_loss(pred_norm, batch["y"])
    return state_mse, {
        "state_mse": float(state_mse.detach().cpu()),
        "real_sensor_mse": 0.0,
        "thermal_residual": 0.0,
        "hydraulic_residual": 0.0,
        "boundary_residual": 0.0,
        "energy_residual": 0.0,
        "smoothness": 0.0,
        "total_loss": float(state_mse.detach().cpu()),
    }
