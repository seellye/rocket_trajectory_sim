"""Numerical integration (RK4), flight summary and plot generation.

The rocket state is ``(altitude h, velocity v)`` and the dynamics are:

    dh/dt = v
    dv/dt = (T - D - m·g) / m

where ``T`` is thrust, ``D`` is drag, ``m(t)`` is the variable mass and ``g``
is gravity. Integration uses 4th-order Runge-Kutta (RK4).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from .physics import air_density, drag_force, gravity, ideal_velocity, net_acceleration
from .rocket import Rocket

FloatArray = NDArray[np.float64]

_EPS: float = 1e-9
"""Numerical tolerance [s] for time comparisons."""


# =========================================================================== #
# Configuration and result
# =========================================================================== #
@dataclass(frozen=True)
class SimulationConfig:
    """Numerical and environment parameters of the simulation.

    Attributes:
        dt: Integration time step [s].
        t_max: Maximum simulation time [s].
        initial_altitude: Initial altitude [m].
        initial_velocity: Initial velocity [m/s] (positive upward).
        variable_gravity: If ``True``, ``g`` decreases with altitude.
        drag_enabled: If ``False``, drag is turned off (useful for comparisons).
    """

    dt: float = 0.01
    t_max: float = 600.0
    initial_altitude: float = 0.0
    initial_velocity: float = 0.0
    variable_gravity: bool = False
    drag_enabled: bool = True

    def __post_init__(self) -> None:
        """Validate the configuration."""
        if self.dt <= 0:
            raise ValueError("dt must be > 0.")
        if self.t_max <= 0:
            raise ValueError("t_max must be > 0.")


@dataclass(frozen=True)
class SimulationResult:
    """Time series produced by :func:`simulate`.

    All arrays have the same length (one sample per step).

    Attributes:
        time: Time [s].
        altitude: Altitude [m].
        velocity: Vertical velocity [m/s].
        acceleration: Vertical acceleration [m/s²].
        mass: Instantaneous mass [kg].
        thrust: Thrust [N].
        drag: Drag, signed [N].
        density: Air density [kg/m³].
        burn_time: Burn time of the simulated rocket [s].
        landed: ``True`` if the rocket returned to the ground before ``t_max``.
    """

    time: FloatArray
    altitude: FloatArray
    velocity: FloatArray
    acceleration: FloatArray
    mass: FloatArray
    thrust: FloatArray
    drag: FloatArray
    density: FloatArray
    burn_time: float
    landed: bool


class _Forces(NamedTuple):
    """Forces and acceleration evaluated at a given state (internal use)."""

    thrust: float
    drag: float
    mass: float
    density: float
    acceleration: float


# =========================================================================== #
# Simulation
# =========================================================================== #
def _evaluate(
    rocket: Rocket,
    config: SimulationConfig,
    t: float,
    altitude: float,
    velocity: float,
    burning: bool,
) -> _Forces:
    """Evaluate thrust, drag, mass and acceleration at the given state."""
    mass = rocket.mass_at(t)  # m(t) = m0 - mdot * t
    thrust = rocket.commanded_thrust(t) if burning else 0.0
    density = air_density(altitude) if config.drag_enabled else 0.0
    drag = drag_force(velocity, density, rocket.drag_coefficient, rocket.reference_area)
    g = gravity(altitude, config.variable_gravity)
    acceleration = net_acceleration(thrust, drag, mass, g)
    return _Forces(thrust, drag, mass, density, acceleration)


def _rk4_step(
    derivatives: Callable[[float, float, float], tuple[float, float]],
    t: float,
    altitude: float,
    velocity: float,
    step: float,
) -> tuple[float, float]:
    """Advance one 4th-order Runge-Kutta step.

    Args:
        derivatives: Function ``(t, h, v) -> (dh/dt, dv/dt)``.
        t: Current time [s].
        altitude: Current altitude [m].
        velocity: Current velocity [m/s].
        step: Step size [s].

    Returns:
        New ``(altitude, velocity)`` pair after ``step`` seconds.
    """
    half = 0.5 * step
    k1h, k1v = derivatives(t, altitude, velocity)
    k2h, k2v = derivatives(t + half, altitude + half * k1h, velocity + half * k1v)
    k3h, k3v = derivatives(t + half, altitude + half * k2h, velocity + half * k2v)
    k4h, k4v = derivatives(t + step, altitude + step * k3h, velocity + step * k3v)
    new_altitude = altitude + step / 6.0 * (k1h + 2.0 * k2h + 2.0 * k3h + k4h)
    new_velocity = velocity + step / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    return new_altitude, new_velocity


def simulate(rocket: Rocket, config: SimulationConfig | None = None) -> SimulationResult:
    """Simulate the 1D vertical flight of a rocket with RK4.

    The simulation ends when the rocket returns to the ground (impact) or when
    ``config.t_max`` is reached. The step is adjusted so that one step ends
    exactly at burnout, which keeps RK4 from stepping across the thrust
    discontinuity.

    Args:
        rocket: Rocket to simulate.
        config: Numerical and environment configuration. If ``None``, the
            defaults are used.

    Returns:
        :class:`SimulationResult` with time, altitude, velocity and
        acceleration (plus mass, thrust, drag and air density).

    Raises:
        ValueError: If the rocket cannot lift off (thrust <= weight on the ground).
    """
    cfg = config if config is not None else SimulationConfig()

    t = 0.0
    altitude = cfg.initial_altitude
    velocity = cfg.initial_velocity

    initial = _evaluate(rocket, cfg, t, altitude, velocity, rocket.is_burning(t))
    if altitude <= 0.0 and velocity <= 0.0 and initial.acceleration <= 0.0:
        raise ValueError(
            "The rocket cannot lift off: initial thrust <= weight. "
            f"(T/W = {rocket.thrust_to_weight:.2f})"
        )

    # Sample lists: (t, h, v, forces)
    times: list[float] = [t]
    altitudes: list[float] = [altitude]
    velocities: list[float] = [velocity]
    samples: list[_Forces] = [initial]
    landed = False

    while t < cfg.t_max - _EPS:
        burning = rocket.is_burning(t)  # constant throughout the step
        step = min(cfg.dt, cfg.t_max - t)
        next_t = t + step

        # Adjust the step so that it ends exactly at burnout.
        if burning and next_t > rocket.burn_time + _EPS:
            step = rocket.burn_time - t
            next_t = rocket.burn_time

        def derivatives(tt: float, hh: float, vv: float) -> tuple[float, float]:
            return vv, _evaluate(rocket, cfg, tt, hh, vv, burning).acceleration

        new_altitude, new_velocity = _rk4_step(derivatives, t, altitude, velocity, step)

        # Ground impact: linearly interpolate the instant when h = 0.
        if new_altitude <= 0.0 and new_velocity < 0.0:
            drop = altitude - new_altitude
            fraction = altitude / drop if drop > 0.0 else 1.0
            next_t = t + fraction * step
            new_velocity = velocity + fraction * (new_velocity - velocity)
            new_altitude = 0.0
            landed = True

        t, altitude, velocity = next_t, new_altitude, new_velocity
        times.append(t)
        altitudes.append(altitude)
        velocities.append(velocity)
        samples.append(_evaluate(rocket, cfg, t, altitude, velocity, rocket.is_burning(t)))

        if landed:
            break

    return SimulationResult(
        time=np.array(times),
        altitude=np.array(altitudes),
        velocity=np.array(velocities),
        acceleration=np.array([s.acceleration for s in samples]),
        mass=np.array([s.mass for s in samples]),
        thrust=np.array([s.thrust for s in samples]),
        drag=np.array([s.drag for s in samples]),
        density=np.array([s.density for s in samples]),
        burn_time=rocket.burn_time,
        landed=landed,
    )


# =========================================================================== #
# Flight summary
# =========================================================================== #
@dataclass(frozen=True)
class FlightSummary:
    """Key figures of a simulated flight (SI units)."""

    burn_time: float
    burnout_altitude: float
    burnout_velocity: float
    apogee: float
    time_to_apogee: float
    max_velocity: float
    time_of_max_velocity: float
    max_acceleration: float
    max_deceleration: float
    max_dynamic_pressure: float
    time_of_max_q: float
    flight_time: float
    impact_velocity: float
    landed: bool


def summarize(result: SimulationResult) -> FlightSummary:
    """Extract the key figures from a :class:`SimulationResult`."""
    t, h, v, a = result.time, result.altitude, result.velocity, result.acceleration
    dynamic_pressure = 0.5 * result.density * v**2  # q = ½ ρ v²

    i_apogee = int(np.argmax(h))
    i_vmax = int(np.argmax(v))
    i_maxq = int(np.argmax(dynamic_pressure))

    return FlightSummary(
        burn_time=result.burn_time,
        burnout_altitude=float(np.interp(result.burn_time, t, h)),
        burnout_velocity=float(np.interp(result.burn_time, t, v)),
        apogee=float(h[i_apogee]),
        time_to_apogee=float(t[i_apogee]),
        max_velocity=float(v[i_vmax]),
        time_of_max_velocity=float(t[i_vmax]),
        max_acceleration=float(np.max(a)),
        max_deceleration=float(np.min(a)),
        max_dynamic_pressure=float(dynamic_pressure[i_maxq]),
        time_of_max_q=float(t[i_maxq]),
        flight_time=float(t[-1]),
        impact_velocity=float(v[-1]) if result.landed else float("nan"),
        landed=result.landed,
    )


def format_report(rocket: Rocket, summary: FlightSummary) -> str:
    """Build the text report (rocket parameters + flight data)."""
    bar = "=" * 58

    def row(label: str, value: float, unit: str, decimals: int = 1) -> str:
        return f"  {label:<32}{value:>12.{decimals}f} {unit}".rstrip()

    lines = [
        bar,
        "  1D ROCKET TRAJECTORY SIMULATOR",
        bar,
        "  Rocket",
        row("Initial mass", rocket.initial_mass, "kg"),
        row("Dry mass", rocket.dry_mass, "kg"),
        row("Nominal thrust", rocket.thrust, "N", 0),
        row("Thrust-to-weight ratio (T/W)", rocket.thrust_to_weight, "", 2),
        row("Specific impulse (Isp)", rocket.specific_impulse, "s"),
        "",
        "  Flight results",
        row("Burn time", summary.burn_time, "s"),
        row("Burnout altitude", summary.burnout_altitude, "m"),
        row("Burnout velocity", summary.burnout_velocity, "m/s"),
        row("Maximum altitude (apogee)", summary.apogee, "m"),
        row("Time to apogee", summary.time_to_apogee, "s"),
        row("Maximum velocity", summary.max_velocity, "m/s"),
        row("  occurs at t =", summary.time_of_max_velocity, "s"),
        row("Maximum acceleration", summary.max_acceleration, "m/s²"),
        row("Maximum deceleration", summary.max_deceleration, "m/s²"),
        row("Max dynamic pressure (Max-Q)", summary.max_dynamic_pressure / 1000.0, "kPa"),
        row("  occurs at t =", summary.time_of_max_q, "s"),
        row("Total flight time", summary.flight_time, "s"),
    ]
    if summary.landed:
        lines.append(row("Impact velocity", summary.impact_velocity, "m/s"))
    else:
        lines.append("  (!) The rocket did not land before t_max.")
    lines.append(bar)
    return "\n".join(lines)


def format_ideal_comparison(
    rocket: Rocket,
    result: SimulationResult,
    result_no_drag: SimulationResult,
) -> str:
    """Compare the burnout velocity with the ideal rocket equation.

    The "velocity budget" is broken down into:

    * ideal Δv (Tsiolkovsky, no gravity and no drag);
    * gravity loss = ideal Δv - burnout velocity *without drag*;
    * drag loss = burnout velocity without drag - burnout velocity with drag.

    Args:
        rocket: Simulated rocket.
        result: Full simulation (with drag).
        result_no_drag: Same simulation with ``drag_enabled=False``.
    """
    ideal = rocket.ideal_delta_v
    v_bo_no_drag = summarize(result_no_drag).burnout_velocity
    v_bo = summarize(result).burnout_velocity
    gravity_loss = ideal - v_bo_no_drag
    drag_loss = v_bo_no_drag - v_bo
    bar = "=" * 58

    def row(label: str, value: float, unit: str) -> str:
        return f"  {label:<38}{value:>8.1f} {unit}"

    return "\n".join(
        [
            bar,
            "  COMPARISON WITH THE IDEAL ROCKET EQUATION",
            bar,
            row("Ideal Δv (Tsiolkovsky)", ideal, "m/s"),
            row("(-) gravity loss", gravity_loss, "m/s"),
            row("(-) drag loss", drag_loss, "m/s"),
            row("(=) simulated burnout velocity", v_bo, "m/s"),
            f"  Efficiency: {100.0 * v_bo / ideal:.1f}% of ideal Δv",
            bar,
        ]
    )


# =========================================================================== #
# Plots
# =========================================================================== #
@dataclass(frozen=True)
class _PlotSpec:
    """Description of a plot (series vs. time)."""

    attribute: str
    title: str
    ylabel: str
    filename: str
    color: str


_LABEL_BOX: dict[str, object] = {"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.85}

_PLOT_SPECS: tuple[_PlotSpec, ...] = (
    _PlotSpec("altitude", "Altitude vs Time", "Altitude [m]", "altitude_vs_time.png", "tab:blue"),
    _PlotSpec("velocity", "Velocity vs Time", "Velocity [m/s]", "velocity_vs_time.png", "tab:green"),
    _PlotSpec(
        "acceleration", "Acceleration vs Time", "Acceleration [m/s²]", "acceleration_vs_time.png", "tab:orange"
    ),
)


def _draw_panel(ax: Axes, spec: _PlotSpec, result: SimulationResult, show_xlabel: bool = True) -> None:
    """Draw one time series with burnout and peak markers."""
    t = result.time
    y = getattr(result, spec.attribute)

    ax.plot(t, y, color=spec.color, linewidth=2)
    ax.axvline(
        result.burn_time, color="tab:red", linestyle="--", linewidth=1,
        label=f"Burnout (t = {result.burn_time:.1f} s)",
    )

    if spec.attribute == "altitude":
        i = int(np.argmax(y))
        ax.plot(t[i], y[i], "o", color="black")
        ax.annotate(
            f"Apogee: {y[i]:.0f} m\nt = {t[i]:.1f} s",
            (t[i], y[i]), textcoords="offset points", xytext=(0, 12),
            ha="center", bbox=_LABEL_BOX,
        )
    elif spec.attribute == "velocity":
        i = int(np.argmax(y))
        ax.plot(t[i], y[i], "o", color="black")
        ax.annotate(
            f"Max: {y[i]:.0f} m/s\nt = {t[i]:.1f} s",
            (t[i], y[i]), textcoords="offset points", xytext=(0, 12),
            ha="center", bbox=_LABEL_BOX,
        )
        ax.axhline(0.0, color="gray", linewidth=0.8)
    else:
        ax.axhline(0.0, color="gray", linewidth=0.8)

    # Small margin at the bottom and a larger one on top, to fit the annotations.
    low, high = float(np.min(y)), float(np.max(y))
    span = (high - low) or 1.0
    ax.set_ylim(low - 0.04 * span, high + 0.25 * span)
    ax.set_title(spec.title)
    ax.set_ylabel(spec.ylabel)
    if show_xlabel:
        ax.set_xlabel("Time [s]")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")


def _save(fig: plt.Figure, save_dir: Path | None, filename: str) -> list[Path]:
    """Save the figure in ``save_dir`` (if given) and return the paths."""
    if save_dir is None:
        return []
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / filename
    fig.savefig(path, dpi=150)
    return [path]


def plot_results(result: SimulationResult, save_dir: Path | None = None) -> list[Path]:
    """Generate the altitude, velocity and acceleration plots.

    Creates 3 individual figures (one per quantity) and 1 overview figure with
    the three panels stacked (``trajectory_overview.png``).

    The figures are left open: call ``plt.show()`` to display them or
    ``plt.close("all")`` to release them.

    Args:
        result: Output of :func:`simulate`.
        save_dir: Folder where the PNGs are saved. If ``None``, nothing is saved.

    Returns:
        List of the saved file paths.
    """
    saved: list[Path] = []

    for spec in _PLOT_SPECS:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        _draw_panel(ax, spec, result)
        fig.tight_layout()
        saved += _save(fig, save_dir, spec.filename)

    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    for ax, spec in zip(axes, _PLOT_SPECS):
        _draw_panel(ax, spec, result, show_xlabel=False)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("1D Rocket Trajectory Simulator", fontsize=14)
    fig.tight_layout()
    saved += _save(fig, save_dir, "trajectory_overview.png")

    return saved


def plot_velocity_comparison(
    rocket: Rocket,
    result: SimulationResult,
    result_no_drag: SimulationResult,
    save_dir: Path | None = None,
) -> list[Path]:
    """Compare velocity with drag, without drag and from the ideal equation.

    The ideal curve ``v = v_e · ln(m0 / m(t))`` only exists during the burn.

    Args:
        rocket: Simulated rocket.
        result: Simulation with drag.
        result_no_drag: Simulation without drag (``drag_enabled=False``).
        save_dir: Folder where the PNG is saved. If ``None``, nothing is saved.

    Returns:
        List with the saved file path.
    """
    burn_times = np.linspace(0.0, rocket.burn_time, 200)
    v_ideal = [
        ideal_velocity(rocket.exhaust_velocity, rocket.initial_mass, rocket.mass_at(float(t)))
        for t in burn_times
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(result.time, result.velocity, linewidth=2, label="Simulation with drag")
    ax.plot(result_no_drag.time, result_no_drag.velocity, "--", linewidth=2, label="Simulation without drag")
    ax.plot(burn_times, v_ideal, ":", linewidth=2.5, color="black",
            label="Ideal rocket (Tsiolkovsky: no gravity, no drag)")
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_xlim(0.0, float(result.time[-1]))
    ax.set_ylim(1.2 * float(np.min(result.velocity)), 1.08 * max(v_ideal))  # visible part only
    ax.set_title("Effect of drag and gravity on velocity")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Velocity [m/s]")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()

    return _save(fig, save_dir, "velocity_comparison.png")
