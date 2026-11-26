"""Example: simple vertical launch with the project's default parameters.

Usage (from the repository root):

    python examples/simple_launch.py
    python examples/simple_launch.py --variable-gravity
    python examples/simple_launch.py --no-show          # only saves the plots
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

# Lets the script run directly (python examples/simple_launch.py) without installing the package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402

from src.rocket import Rocket  # noqa: E402
from src.simulation import (  # noqa: E402
    SimulationConfig,
    format_ideal_comparison,
    format_report,
    plot_results,
    plot_velocity_comparison,
    simulate,
    summarize,
)

PLOTS_DIR = ROOT / "plots"


def parse_args() -> argparse.Namespace:
    """Read the command-line options."""
    parser = argparse.ArgumentParser(description="1D rocket trajectory simulator.")
    parser.add_argument("--variable-gravity", action="store_true",
                        help="use altitude-dependent gravity (g ~ 1/r²).")
    parser.add_argument("--dt", type=float, default=0.01,
                        help="integration time step in seconds (default: 0.01).")
    parser.add_argument("--no-show", action="store_true",
                        help="do not open the plot windows (only save the PNGs).")
    return parser.parse_args()


def main() -> None:
    """Configure the rocket, simulate, print the report and generate the plots."""
    args = parse_args()

    # ----------------------------------------------------------------- #
    # Example parameters
    # ----------------------------------------------------------------- #
    rocket = Rocket(
        dry_mass=100.0,           # dry mass [kg]
        fuel_mass=200.0,         # propellant mass [kg]
        thrust=5000.0,           # constant thrust [N]
        mass_flow_rate=3.5,      # consumption rate [kg/s]  -> 40 s burn
        drag_coefficient=1.5,    # Cd [-]
        reference_area=0.9,      # reference area [m²]
    )
    config = SimulationConfig(
        dt=args.dt,
        initial_altitude=0.0,    # [m]
        initial_velocity=0.0,    # [m/s]
        variable_gravity=args.variable_gravity,
    )

    # ----------------------------------------------------------------- #
    # Simulations: real flight (with drag) and a no-drag reference
    # ----------------------------------------------------------------- #
    result = simulate(rocket, config)
    result_no_drag = simulate(rocket, replace(config, drag_enabled=False))

    gravity_mode = "variable (1/r²)" if config.variable_gravity else "constant"
    print(f"\nGravity: {gravity_mode} | time step dt = {config.dt} s\n")
    print(format_report(rocket, summarize(result)))
    print()
    print(format_ideal_comparison(rocket, result, result_no_drag))

    # ----------------------------------------------------------------- #
    # Plots
    # ----------------------------------------------------------------- #
    saved = plot_results(result, save_dir=PLOTS_DIR)
    saved += plot_velocity_comparison(rocket, result, result_no_drag, save_dir=PLOTS_DIR)

    print("\nPlots saved to:")
    for path in saved:
        print(f"  - {path.relative_to(ROOT)}")

    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
