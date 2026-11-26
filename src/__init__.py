"""1D Rocket Trajectory Simulator.

Public API:

>>> from src import Rocket, SimulationConfig, simulate
"""

from .physics import G0, air_density, drag_force, gravity, ideal_velocity, net_acceleration
from .rocket import Rocket
from .simulation import (
    FlightSummary,
    SimulationConfig,
    SimulationResult,
    format_ideal_comparison,
    format_report,
    plot_results,
    plot_velocity_comparison,
    simulate,
    summarize,
)

__version__ = "1.0.0"

__all__ = [
    "G0",
    "FlightSummary",
    "Rocket",
    "SimulationConfig",
    "SimulationResult",
    "air_density",
    "drag_force",
    "format_ideal_comparison",
    "format_report",
    "gravity",
    "ideal_velocity",
    "net_acceleration",
    "plot_results",
    "plot_velocity_comparison",
    "simulate",
    "summarize",
]
