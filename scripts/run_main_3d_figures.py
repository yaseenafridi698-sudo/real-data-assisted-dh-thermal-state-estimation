from __future__ import annotations

from _main_3d_utils import make_contact_sheet, write_latex_snippets
from plot_main_3d_heat_loss import main as heat_loss_main
from plot_main_3d_pressure import main as pressure_main
from plot_main_3d_return_temperature import main as return_temperature_main
from plot_main_3d_supply_temperature import main as supply_temperature_main


def main() -> None:
    supply_temperature_main()
    heat_loss_main()
    pressure_main()
    return_temperature_main()
    make_contact_sheet(
        [
            "main_3d_supply_temperature_surface",
            "main_3d_heat_loss_surface",
            "main_3d_pressure_surface",
            "main_3d_return_temperature_surface",
        ]
    )
    write_latex_snippets()
    print("Main 3D thermo-hydraulic figures generated.")


if __name__ == "__main__":
    main()
