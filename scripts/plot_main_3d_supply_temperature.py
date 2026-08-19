from __future__ import annotations

from _main_3d_utils import infer_sensor_distances, load_virtual_field, plot_surface


def main() -> None:
    x, y, z = load_virtual_field("supply_temperature")
    plot_surface(
        X=x,
        Y=y,
        Z=z,
        stem="main_3d_supply_temperature_surface",
        z_label=r"Supply temperature ($^\circ$C)",
        colorbar_label=r"$T_s$ ($^\circ$C)",
        title="Supply-temperature field",
        cmap="viridis",
        sensor_distances_km=infer_sensor_distances("optimized_three"),
    )


if __name__ == "__main__":
    main()
