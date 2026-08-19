from __future__ import annotations

import shutil

from _main_3d_utils import FIGURES_DIR, PAPER_FIGURES_DIR, infer_sensor_distances, load_virtual_field, plot_surface


def main() -> None:
    x, y, z = load_virtual_field("return_temperature")
    plot_surface(
        X=x,
        Y=y,
        Z=z,
        stem="main_3d_return_temperature_surface",
        z_label=r"Return temperature ($^\circ$C)",
        colorbar_label=r"$T_r$ ($^\circ$C)",
        title="Return-temperature field",
        cmap="cividis",
        sensor_distances_km=infer_sensor_distances("optimized_three"),
    )
    # Keep the older supplementary filename as an alias for manuscript compatibility.
    for suffix in [".pdf", ".png"]:
        for folder in [FIGURES_DIR, PAPER_FIGURES_DIR]:
            src = folder / f"main_3d_return_temperature_surface{suffix}"
            dst = folder / f"supp_3d_return_temperature_surface{suffix}"
            if src.exists() and src.resolve() != dst.resolve():
                shutil.copy2(src, dst)


if __name__ == "__main__":
    main()
