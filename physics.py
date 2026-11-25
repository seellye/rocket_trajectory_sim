"""Pure physics functions used by the simulator.

All quantities are in SI units: meters, seconds, kilograms and newtons. The
functions in this module hold no state: the same inputs always produce the
same output.
"""

from __future__ import annotations

import math

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
G0: float = 9.80665
"""Gravitational acceleration at sea level [m/s²]."""

EARTH_RADIUS: float = 6_371_000.0
"""Mean Earth radius [m] (used by the variable-gravity model)."""

SEA_LEVEL_DENSITY: float = 1.225
"""Air density at sea level, standard atmosphere [kg/m³]."""

SCALE_HEIGHT: float = 8_500.0
"""Scale height of the exponential atmosphere [m]."""


# --------------------------------------------------------------------------- #
# Environment: gravity and atmosphere
# --------------------------------------------------------------------------- #
def gravity(altitude: float, variable: bool = False) -> float:
    """Return the gravitational acceleration [m/s²] at the given altitude.

    Args:
        altitude: Altitude above sea level [m].
        variable: If ``False`` (default), use a constant ``g`` (9.80665 m/s²).
            If ``True``, apply the inverse-square law:

            ``g(h) = g0 * (R / (R + h))²``

    Returns:
        Gravitational acceleration [m/s²].
    """
    if not variable:
        return G0
    height = max(altitude, 0.0)
    return G0 * (EARTH_RADIUS / (EARTH_RADIUS + height)) ** 2


def air_density(altitude: float) -> float:
    """Air density [kg/m³] from the simplified exponential model.

    ``rho(h) = rho0 * exp(-h / H)``

    with ``rho0 = 1.225 kg/m³`` and ``H = 8500 m``. This is a good
    approximation for the troposphere and lower stratosphere, which is enough
    for a 1D simulator.

    Args:
        altitude: Altitude above sea level [m]. Negative values are treated
            as zero.

    Returns:
        Air density [kg/m³].
    """
    return SEA_LEVEL_DENSITY * math.exp(-max(altitude, 0.0) / SCALE_HEIGHT)


# --------------------------------------------------------------------------- #
# Forces and dynamics
# --------------------------------------------------------------------------- #
def drag_force(
    velocity: float,
    density: float,
    drag_coefficient: float,
    reference_area: float,
) -> float:
    """Aerodynamic drag force [N], signed.

    ``D = ½ · rho · v · |v| · Cd · A``

    Using ``v·|v|`` instead of ``v²`` makes drag always oppose the motion: it
    is positive while ascending (``v > 0``) and negative while descending
    (``v < 0``). That way ``a = (T - D - m·g) / m`` holds in both phases of
    the flight.

    Args:
        velocity: Vertical velocity [m/s] (positive upward).
        density: Air density [kg/m³].
        drag_coefficient: Drag coefficient ``Cd`` (dimensionless).
        reference_area: Reference area ``A`` [m²].

    Returns:
        Drag force [N].
    """
    return 0.5 * density * velocity * abs(velocity) * drag_coefficient * reference_area


def net_acceleration(thrust: float, drag: float, mass: float, g: float) -> float:
    """Vertical acceleration [m/s²] from Newton's second law.

    ``a = (T - D - m·g) / m``

    Args:
        thrust: Thrust [N] (upward).
        drag: Drag [N], signed (see :func:`drag_force`).
        mass: Instantaneous rocket mass [kg].
        g: Gravitational acceleration [m/s²].

    Returns:
        Acceleration [m/s²] (positive upward).
    """
    return (thrust - drag - mass * g) / mass


def ideal_velocity(exhaust_velocity: float, initial_mass: float, mass: float) -> float:
    """Velocity from the ideal rocket equation (Tsiolkovsky) [m/s].

    ``v = v_e · ln(m0 / m)``

    Valid for a rocket with no gravity and no drag. It serves as an upper
    bound to compare against the simulation.

    Args:
        exhaust_velocity: Effective exhaust velocity ``v_e = T / ṁ`` [m/s].
        initial_mass: Initial mass ``m0`` [kg].
        mass: Current mass ``m`` [kg].

    Returns:
        Velocity gained [m/s].
    """
    return exhaust_velocity * math.log(initial_mass / mass)
