"""Rocket model: configurable parameters and time-dependent mass and thrust."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .physics import G0, ideal_velocity

ThrustProfile = Callable[[float], float]
"""Function ``t [s] -> thrust [N]`` for variable thrust."""

_BURN_TOLERANCE: float = 1e-9
"""Numerical tolerance [s] used to decide whether the engine is still on."""


@dataclass(frozen=True)
class Rocket:
    """Single-stage rocket with configurable thrust and mass flow rate.

    Attributes:
        dry_mass: Dry mass (structure + payload, no propellant) [kg].
        fuel_mass: Initial propellant mass [kg].
        thrust: Nominal thrust [N]. This is the constant thrust used when there
            is no ``thrust_profile``, and also the basis for ``exhaust_velocity``.
        mass_flow_rate: Propellant consumption rate ``ṁ`` [kg/s].
        drag_coefficient: Drag coefficient ``Cd`` (dimensionless).
        reference_area: Reference area for drag ``A`` [m²].
        thrust_profile: Optional function ``t -> thrust [N]`` for variable
            thrust. If ``None``, thrust is constant and equal to ``thrust``.
            Note: with a variable profile, the "ideal" quantities
            (``exhaust_velocity``, ``specific_impulse``, ``ideal_delta_v``)
            use the nominal thrust and are only approximations.

    Example:
        >>> rocket = Rocket(50.0, 100.0, 3000.0, 2.5, 0.5, 0.1)
        >>> rocket.burn_time
        40.0
    """

    dry_mass: float
    fuel_mass: float
    thrust: float
    mass_flow_rate: float
    drag_coefficient: float
    reference_area: float
    thrust_profile: ThrustProfile | None = None

    def __post_init__(self) -> None:
        """Validate the physical parameters."""
        if self.dry_mass <= 0:
            raise ValueError("dry_mass must be > 0.")
        if self.fuel_mass < 0:
            raise ValueError("fuel_mass must be >= 0.")
        if self.thrust < 0:
            raise ValueError("thrust must be >= 0.")
        if self.mass_flow_rate <= 0:
            raise ValueError("mass_flow_rate must be > 0.")
        if self.drag_coefficient < 0:
            raise ValueError("drag_coefficient must be >= 0.")
        if self.reference_area <= 0:
            raise ValueError("reference_area must be > 0.")

    # ------------------------------------------------------------------ #
    # Derived properties
    # ------------------------------------------------------------------ #
    @property
    def initial_mass(self) -> float:
        """Total liftoff mass, ``m0 = dry_mass + fuel_mass`` [kg]."""
        return self.dry_mass + self.fuel_mass

    @property
    def burn_time(self) -> float:
        """Burn time ``t_b = fuel_mass / ṁ`` [s]."""
        return self.fuel_mass / self.mass_flow_rate

    @property
    def exhaust_velocity(self) -> float:
        """Effective exhaust velocity ``v_e = T / ṁ`` [m/s]."""
        return self.thrust / self.mass_flow_rate

    @property
    def specific_impulse(self) -> float:
        """Specific impulse ``Isp = v_e / g0`` [s]."""
        return self.exhaust_velocity / G0

    @property
    def thrust_to_weight(self) -> float:
        """Thrust-to-weight ratio at liftoff, ``T / (m0 · g0)``."""
        return self.thrust / (self.initial_mass * G0)

    @property
    def ideal_delta_v(self) -> float:
        """Ideal Tsiolkovsky Δv, ``v_e · ln(m0 / dry_mass)`` [m/s].

        With no gravity and no drag, this is the theoretical maximum for
        this rocket.
        """
        return ideal_velocity(self.exhaust_velocity, self.initial_mass, self.dry_mass)

    # ------------------------------------------------------------------ #
    # Time-dependent quantities
    # ------------------------------------------------------------------ #
    def is_burning(self, t: float) -> bool:
        """Return whether the engine still has propellant at time ``t`` [s]."""
        return t < self.burn_time - _BURN_TOLERANCE

    def mass_at(self, t: float) -> float:
        """Instantaneous mass [kg]: ``m(t) = m0 - ṁ·t`` until the fuel runs out.

        After burnout the mass stays equal to the dry mass.
        """
        fuel_left = max(self.fuel_mass - self.mass_flow_rate * max(t, 0.0), 0.0)
        return self.dry_mass + fuel_left

    def commanded_thrust(self, t: float) -> float:
        """Thrust the engine would produce at ``t`` [N], ignoring fuel state.

        Uses ``thrust_profile(t)`` if present; otherwise the constant thrust.
        The simulator combines this value with :meth:`is_burning`.
        """
        value = self.thrust_profile(t) if self.thrust_profile is not None else self.thrust
        return max(value, 0.0)

    def thrust_at(self, t: float) -> float:
        """Actual thrust [N] at ``t``: zero after burnout."""
        return self.commanded_thrust(t) if self.is_burning(t) else 0.0
