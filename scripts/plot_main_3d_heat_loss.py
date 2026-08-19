from __future__ import annotations

from _main_3d_utils import (
    infer_sensor_distances,
    load_heat_loss_density_field,
    plot_surface,
    write_heat_loss_unit_report,
)


def main() -> None:
    x, y, z, report = load_heat_loss_density_field()
    write_heat_loss_unit_report(report)
    plot_surface(
        X=x,
        Y=y,
        Z=z,
        stem="main_3d_heat_loss_surface",
        z_label="Heat-loss density (kW/km)",
        colorbar_label="Heat loss (kW/km)",
        title="Heat-loss density field",
        cmap="magma",
        sensor_distances_km=infer_sensor_distances("optimized_three"),
    )


if __name__ == "__main__":
    main()
