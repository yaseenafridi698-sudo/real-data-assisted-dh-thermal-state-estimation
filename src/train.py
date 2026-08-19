from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from .losses import mse_only_loss, physics_informed_loss
from .utils import ensure_dir, get_device


def _safe_float(value: torch.Tensor | float, default: float = 1.0) -> float:
    try:
        if torch.is_tensor(value):
            value = float(value.detach().cpu())
        else:
            value = float(value)
    except (TypeError, ValueError):
        value = default
    if not bool(torch.isfinite(torch.tensor(value))):
        return default
    return max(abs(value), 1e-12)


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def estimate_physics_loss_scales(train_loader, config: dict[str, Any], max_batches: int = 6) -> dict[str, float]:
    sys = config["system"]
    dt = float(sys["dt_s"])
    dx = float(sys["dx_m"])
    rho = float(sys["rho"])
    cp = float(sys["cp"])
    diameter = float(sys["diameter_m"])
    area = torch.tensor(3.141592653589793 * diameter**2 / 4.0, dtype=torch.float32)
    U = float(sys["heat_loss_U_W_m2K"])
    perimeter = float(sys["pipe_perimeter_m"])
    k_loss = U * perimeter / max(rho * cp * float(area), 1e-12)
    scales: dict[str, list[float]] = {
        "thermal_residual_scale": [],
        "hydraulic_residual_scale": [],
        "boundary_residual_scale": [],
        "energy_residual_scale": [],
        "smoothness_scale": [],
        "heat_loss_scale": [],
    }
    for idx, batch in enumerate(train_loader):
        if idx >= max_batches:
            break
        target = batch["target_physical"].float()
        Ts = target[..., 0]
        Tr = target[..., 1]
        H = target[..., 2]
        q = torch.clamp(target[..., 3], min=1e-4)
        Ta = batch["ambient"].float()
        alpha = batch["alpha"].float()
        source_temp = batch["source_temp"].float()
        heat_load_kw = batch["heat_load_kw"].float()
        if target.shape[1] > 1 and target.shape[2] > 2:
            v = q / area
            r_ts = (Ts[:, 1:, 1:] - Ts[:, :-1, 1:]) / dt + v[:, :-1, 1:] * (Ts[:, :-1, 1:] - Ts[:, :-1, :-1]) / dx + k_loss * (Ts[:, :-1, 1:] - Ta[:, :-1, None])
            r_tr = (Tr[:, 1:, :-1] - Tr[:, :-1, :-1]) / dt - v[:, :-1, :-1] * (Tr[:, :-1, 1:] - Tr[:, :-1, :-1]) / dx + k_loss * (Tr[:, :-1, :-1] - Ta[:, :-1, None])
            scales["thermal_residual_scale"].append(_safe_float(torch.mean(r_ts**2) + torch.mean(r_tr**2)))
            smooth = torch.mean((target[:, 1:] - target[:, :-1]) ** 2) * 0.01 + torch.mean((target[:, :, 1:] - target[:, :, :-1]) ** 2) * 0.01
            scales["smoothness_scale"].append(_safe_float(smooth))
        pump_drop = sys["pump_c1"] * alpha**2 + sys["pump_c2"] * alpha + sys["pump_c3"] - float(sys["outlet_head_m"])
        pred_drop = H[:, :, 0] - H[:, :, -1]
        scales["hydraulic_residual_scale"].append(_safe_float(torch.mean((pred_drop / 100.0 - pump_drop / 100.0) ** 2)))
        supply_seg = 0.5 * (Ts[..., :-1] + Ts[..., 1:])
        return_seg = 0.5 * (Tr[..., :-1] + Tr[..., 1:])
        total_heat_loss = U * perimeter * dx * torch.sum(
            (supply_seg - Ta[..., None]) + (return_seg - Ta[..., None]), dim=-1
        ) / 1000.0
        source_heat_kw = rho * cp * q[:, :, 0] * (Ts[:, :, 0] - Tr[:, :, 0]) / 1000.0
        delivered_kw = rho * cp * q[:, :, -1] * torch.clamp(Ts[:, :, -1] - Tr[:, :, -1], min=0.0) / 1000.0
        total_temperature = Ts + Tr
        trapz_temperature = 0.5 * total_temperature[..., 0] + torch.sum(total_temperature[..., 1:-1], dim=-1) + 0.5 * total_temperature[..., -1]
        pipe_energy_kj = rho * cp * float(area) * dx * trapz_temperature / 1000.0
        storage_kw = torch.zeros_like(pipe_energy_kj)
        if pipe_energy_kj.shape[1] > 1:
            storage_kw[:, 1:] = (pipe_energy_kj[:, 1:] - pipe_energy_kj[:, :-1]) / dt
            storage_kw[:, 0] = storage_kw[:, 1]
        load_scale = torch.clamp(torch.nanmean(torch.abs(heat_load_kw)), min=1.0)
        dynamic_energy_residual = source_heat_kw - delivered_kw - total_heat_loss - storage_kw
        scales["energy_residual_scale"].append(_safe_float(torch.mean((dynamic_energy_residual / load_scale) ** 2)))
        raw_hydraulic = torch.mean((pred_drop / 100.0 - pump_drop / 100.0) ** 2)
        boundary = torch.mean((Ts[:, :, 0] - source_temp) ** 2) + 0.25 * torch.mean((Tr[:, :, -1] - target[:, :, -1, 1]) ** 2) + raw_hydraulic
        scales["boundary_residual_scale"].append(_safe_float(boundary))
        normalized_heat_loss = total_heat_loss / torch.clamp(torch.mean(torch.abs(total_heat_loss)), min=1.0)
        scales["heat_loss_scale"].append(_safe_float(torch.var(normalized_heat_loss), default=1.0))
    return {key: float(pd.Series(values).median()) if values else 1.0 for key, values in scales.items()}


def _scheduled_loss_weights(epoch: int, config: dict[str, Any], base_weights: dict[str, float] | None) -> dict[str, float]:
    training = config["training"]
    weights = dict(base_weights or {})
    mode = str(weights.pop("training_mode", training.get("training_mode", "balanced_mode")))
    if mode == "accuracy_mode":
        multipliers = {
            "lambda_sensor": 1.0,
            "lambda_thermal": 0.15,
            "lambda_hydraulic": 0.1,
            "lambda_boundary": 0.25,
            "lambda_energy": 0.1,
            "lambda_heat_loss": 0.1,
            "lambda_smooth": 0.2,
        }
    elif mode == "physics_mode":
        multipliers = {
            "lambda_sensor": 1.5,
            "lambda_thermal": 1.5,
            "lambda_hydraulic": 1.5,
            "lambda_boundary": 1.5,
            "lambda_energy": 1.5,
            "lambda_heat_loss": 1.5,
            "lambda_smooth": 1.0,
        }
    else:
        multipliers = {key: 1.0 for key in ["lambda_sensor", "lambda_thermal", "lambda_hydraulic", "lambda_boundary", "lambda_energy", "lambda_heat_loss", "lambda_smooth"]}
    for key, mult in multipliers.items():
        weights[key] = float(weights.get(key, training.get(key, 0.0))) * mult
    warmup = int(training.get("warmup_epochs", 0))
    ramp_epochs = max(1, int(training.get("physics_ramp_epochs", 1)))
    physics_keys = ["lambda_thermal", "lambda_hydraulic", "lambda_boundary", "lambda_energy", "lambda_heat_loss", "lambda_smooth"]
    if epoch < warmup:
        weights["lambda_sensor"] = float(weights.get("lambda_sensor", training.get("lambda_sensor", 1.0)))
        for key in physics_keys:
            weights[key] = 0.0
        return weights
    if epoch < warmup + ramp_epochs:
        factor = float(epoch - warmup + 1) / float(ramp_epochs)
        weights["lambda_sensor"] = float(weights.get("lambda_sensor", training.get("lambda_sensor", 1.0)))
        for key in physics_keys:
            weights[key] = float(weights.get(key, training.get(key, 0.0))) * factor
        return weights
    return weights


def train_model(
    model: nn.Module,
    model_name: str,
    train_loader,
    val_loader,
    config: dict[str, Any],
    stats: dict[str, Any],
    output_dir: str | Path,
    quick: bool = False,
    loss_mode: str = "physics",
    loss_weights: dict[str, float] | None = None,
    epochs_override: int | None = None,
) -> tuple[nn.Module, pd.DataFrame]:
    device = get_device()
    model = model.to(device)
    epochs = int(epochs_override if epochs_override is not None else (config["training"]["epochs_quick"] if quick else config["training"]["epochs_full"]))
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    best_selection = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    training = config["training"]
    patience = int(training.get("early_stopping_patience", 5 if quick else 15))
    min_epochs = int(training.get("min_epochs_before_early_stopping", 1 if quick else 1))
    selection_metric = str(training.get("selection_metric", "normalized_state_mse"))
    patience_left = patience
    history = []
    if loss_mode != "mse":
        scales = estimate_physics_loss_scales(train_loader, config)
        config.setdefault("training", {})["physics_loss_scales"] = scales
        out_dir = ensure_dir(output_dir)
        (out_dir / "physics_loss_scales.json").write_text(json.dumps(scales, indent=2), encoding="utf-8")
        if "v3" in model_name.lower():
            (out_dir / "physics_loss_scales_v3.json").write_text(json.dumps(scales, indent=2), encoding="utf-8")

    iterator = tqdm(range(epochs), desc=f"train {model_name}", leave=False)
    for epoch in iterator:
        model.train()
        train_losses = []
        train_logs = []
        scheduled_weights = _scheduled_loss_weights(epoch, config, loss_weights)
        for batch in train_loader:
            batch = _to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            pred = model(batch["x"])
            if loss_mode == "mse":
                loss, logs = mse_only_loss(pred, batch)
            else:
                loss, logs = physics_informed_loss(pred, batch, config, stats, loss_weights=scheduled_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            train_losses.append(float(loss.detach().cpu()))
            train_logs.append(logs)

        model.eval()
        val_losses = []
        selection_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = _to_device(batch, device)
                pred = model(batch["x"])
                if loss_mode == "mse":
                    loss, _ = mse_only_loss(pred, batch)
                else:
                    loss, _ = physics_informed_loss(pred, batch, config, stats, loss_weights=scheduled_weights)
                val_losses.append(float(loss.detach().cpu()))
                # Use the same direct normalized-state error to select the
                # checkpoint for every architecture. Physics losses remain in
                # optimisation and reporting, but do not receive a different
                # early-stopping criterion from MSE baselines.
                selection_losses.append(float(F.mse_loss(pred, batch["y"]).detach().cpu()))
        train_loss = float(pd.Series(train_losses).mean())
        val_loss = float(pd.Series(val_losses).mean())
        selection_loss = float(pd.Series(selection_losses).mean())
        row = {
            "epoch": epoch + 1,
            "model": model_name,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "selection_loss": selection_loss,
            "selection_metric": selection_metric,
        }
        if train_logs:
            row.update(pd.DataFrame(train_logs).mean().to_dict())
        history.append(row)
        iterator.set_postfix(train=f"{train_loss:.3g}", select=f"{selection_loss:.3g}")
        if selection_loss < best_selection:
            best_selection = selection_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_left = patience
        else:
            patience_left -= 1
            if epoch + 1 >= min_epochs and patience_left <= 0:
                break

    model.load_state_dict(best_state)
    out_dir = ensure_dir(output_dir)
    torch.save(model.state_dict(), out_dir / f"{model_name}_best.pt")
    history_df = pd.DataFrame(history)
    history_df.to_csv(out_dir / f"{model_name}_training_history.csv", index=False)
    return model, history_df
