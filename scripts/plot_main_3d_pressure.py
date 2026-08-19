from __future__ import annotations

from _main_3d_utils import infer_sensor_distances, load_virtual_field, plot_surface


def main() -> None:
    x, y, z = load_virtual_field("head", pressure=True)
    plot_surface(
        X=x,
        Y=y,
        Z=z,
        stem="main_3d_pressure_surface",
        z_label="Pressure (kPa)",
        colorbar_label="Pressure (kPa)",
        title="Simulator-assisted pressure field",
        cmap="plasma",
        sensor_distances_km=infer_sensor_distances("optimized_three"),
    )


if __name__ == "__main__":
    main()
