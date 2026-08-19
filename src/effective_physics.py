from __future__ import annotations

import copy
from typing import Any


CALIBRATED_SYSTEM_KEYS = (
    "heat_loss_U_W_m2K",
    "friction_factor",
    "effective_velocity_factor",
    "return_temperature_offset",
    "flow_proxy_blend",
    "effective_pipe_delay_factor",
)


def apply_calibrated_params_to_config(
    config: dict[str, Any], params: dict[str, Any] | None
) -> dict[str, Any]:
    """Copy a configuration and install the simulator's calibrated parameters."""
    effective = copy.deepcopy(config)
    calibrated = params or {}
    system = effective.setdefault("system", {})
    for key in CALIBRATED_SYSTEM_KEYS:
        if key in calibrated:
            system[key] = float(calibrated[key])
    effective["calibrated_parameters"] = dict(calibrated)
    return effective
