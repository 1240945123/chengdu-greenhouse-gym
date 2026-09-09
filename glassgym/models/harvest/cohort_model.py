from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite
from typing import Collection

import pandas as pd


@dataclass(frozen=True)
class HarvestCohortParameters:
    base_temperature_c: float
    maturity_thermal_time_deg_day: float
    dry_matter_fraction: float
    upper_temperature_c: float = 30.0
    minimum_pick_fresh_kg_m2: float = 0.0

    def __post_init__(self) -> None:
        values = {
            "base_temperature_c": self.base_temperature_c,
            "maturity_thermal_time_deg_day": self.maturity_thermal_time_deg_day,
            "dry_matter_fraction": self.dry_matter_fraction,
            "upper_temperature_c": self.upper_temperature_c,
            "minimum_pick_fresh_kg_m2": self.minimum_pick_fresh_kg_m2,
        }
        for name, value in values.items():
            if not isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.maturity_thermal_time_deg_day <= 0.0:
            raise ValueError("maturity_thermal_time_deg_day must be positive")
        if self.upper_temperature_c <= self.base_temperature_c:
            raise ValueError("upper_temperature_c must exceed base_temperature_c")
        if not 0.0 < self.dry_matter_fraction <= 1.0:
            raise ValueError("dry_matter_fraction must be in (0, 1]")
        if self.minimum_pick_fresh_kg_m2 < 0.0:
            raise ValueError("minimum_pick_fresh_kg_m2 must be non-negative")


@dataclass(frozen=True)
class HarvestStepResult:
    harvested_dry_matter_kg_m2: float
    harvested_fresh_kg_m2: float
    mature_dry_matter_kg_m2: float
    standing_dry_matter_kg_m2: float
    cumulative_harvested_fresh_kg_m2: float


@dataclass
class _FruitCohort:
    dry_matter_kg_m2: float
    maturity_clock_deg_day: float


class HarvestCohortModel:
    """Age-structures net fruit dry matter into explicit mature harvest batches.

    Cohorts represent dry-matter increments, not individual fruits or trusses.
    """

    def __init__(
        self,
        parameters: HarvestCohortParameters,
        *,
        initial_fruit_dry_matter_kg_m2: float = 0.0,
        initial_fruit_maturity_fraction: float = 0.0,
    ):
        self.parameters = parameters
        self._cohorts: list[_FruitCohort] = []
        self._thermal_clock_deg_day = 0.0
        initial_mass = float(initial_fruit_dry_matter_kg_m2)
        maturity_fraction = float(initial_fruit_maturity_fraction)
        if not isfinite(initial_mass) or initial_mass < 0.0:
            raise ValueError("initial_fruit_dry_matter_kg_m2 must be finite and non-negative")
        if not isfinite(maturity_fraction) or not 0.0 <= maturity_fraction <= 1.0:
            raise ValueError("initial_fruit_maturity_fraction must be in [0, 1]")
        self.cumulative_net_fruit_growth_dry_matter_kg_m2 = initial_mass
        self.cumulative_harvested_dry_matter_kg_m2 = 0.0
        if initial_mass > 0.0:
            self._cohorts.append(
                _FruitCohort(
                    dry_matter_kg_m2=initial_mass,
                    maturity_clock_deg_day=(
                        (1.0 - maturity_fraction)
                        * self.parameters.maturity_thermal_time_deg_day
                    ),
                )
            )

    @property
    def standing_dry_matter_kg_m2(self) -> float:
        return float(sum(cohort.dry_matter_kg_m2 for cohort in self._cohorts))

    @property
    def mature_dry_matter_kg_m2(self) -> float:
        return float(
            sum(
                cohort.dry_matter_kg_m2
                for cohort in self._cohorts
                if self._thermal_clock_deg_day >= cohort.maturity_clock_deg_day
            )
        )

    @property
    def cumulative_harvested_fresh_kg_m2(self) -> float:
        return (
            self.cumulative_harvested_dry_matter_kg_m2
            / self.parameters.dry_matter_fraction
        )

    def step(
        self,
        net_fruit_dry_matter_change_kg_m2: float,
        air_temperature_c: float,
        dt_hours: float,
        pick: bool = False,
    ) -> HarvestStepResult:
        net_change = float(net_fruit_dry_matter_change_kg_m2)
        temperature = float(air_temperature_c)
        duration = float(dt_hours)
        if not isfinite(net_change):
            raise ValueError("net_fruit_dry_matter_change_kg_m2 must be finite")
        if not isfinite(temperature):
            raise ValueError("air_temperature_c must be finite")
        if not isfinite(duration) or duration < 0.0:
            raise ValueError("dt_hours must be finite and non-negative")

        standing_before = self.standing_dry_matter_kg_m2
        if net_change < -standing_before - 1e-12:
            raise ValueError("net fruit dry-matter loss exceeds standing fruit mass")
        growth = max(net_change, 0.0)
        loss = max(-net_change, 0.0)
        self.cumulative_net_fruit_growth_dry_matter_kg_m2 += net_change
        if duration == 0.0:
            if growth > 0.0:
                self._append_cohort(growth)
        else:
            # Treat allocation as a constant flow over the external interval.
            # Fixed internal integration prevents large environment steps from
            # assigning a full interval of thermal age to newly formed fruit.
            internal_steps = max(1, int(ceil(duration / 0.25)))
            internal_dt = duration / internal_steps
            growth_per_step = growth / internal_steps
            thermal_increment = max(
                min(temperature, self.parameters.upper_temperature_c)
                - self.parameters.base_temperature_c,
                0.0,
            ) * internal_dt / 24.0
            for _ in range(internal_steps):
                self._thermal_clock_deg_day += thermal_increment
                if growth_per_step > 0.0:
                    self._append_cohort(growth_per_step)
        if loss > 0.0:
            self._apply_proportional_loss(min(loss, standing_before))

        harvested_dm = 0.0
        if pick:
            mature_dm = self.mature_dry_matter_kg_m2
            mature_fresh = mature_dm / self.parameters.dry_matter_fraction
            if mature_fresh >= self.parameters.minimum_pick_fresh_kg_m2:
                self._cohorts = [
                    cohort
                    for cohort in self._cohorts
                    if self._thermal_clock_deg_day < cohort.maturity_clock_deg_day
                ]
                harvested_dm = mature_dm
                self.cumulative_harvested_dry_matter_kg_m2 += harvested_dm

        return HarvestStepResult(
            harvested_dry_matter_kg_m2=harvested_dm,
            harvested_fresh_kg_m2=(
                harvested_dm / self.parameters.dry_matter_fraction
            ),
            mature_dry_matter_kg_m2=self.mature_dry_matter_kg_m2,
            standing_dry_matter_kg_m2=self.standing_dry_matter_kg_m2,
            cumulative_harvested_fresh_kg_m2=(
                self.cumulative_harvested_fresh_kg_m2
            ),
        )

    def _append_cohort(self, dry_matter_kg_m2: float) -> None:
        self._cohorts.append(
            _FruitCohort(
                dry_matter_kg_m2=dry_matter_kg_m2,
                maturity_clock_deg_day=(
                    self._thermal_clock_deg_day
                    + self.parameters.maturity_thermal_time_deg_day
                ),
            )
        )

    def _apply_proportional_loss(self, loss_dry_matter_kg_m2: float) -> None:
        standing = self.standing_dry_matter_kg_m2
        if standing <= 0.0 or loss_dry_matter_kg_m2 >= standing - 1e-12:
            self._cohorts.clear()
            return
        retained_fraction = 1.0 - loss_dry_matter_kg_m2 / standing
        for cohort in self._cohorts:
            cohort.dry_matter_kg_m2 *= retained_fraction


def is_scheduled_pick(
    timestamp: pd.Timestamp,
    *,
    weekdays: Collection[int],
    local_hour: int,
    timezone: str = "Asia/Shanghai",
) -> bool:
    valid_weekdays, hour = _validate_schedule(weekdays, local_hour)
    time = _local_timestamp(timestamp, timezone)
    if pd.isna(time):
        raise ValueError("timestamp must be valid")
    return (
        time.weekday() in valid_weekdays
        and time.hour == hour
        and time.minute == 0
        and time.second == 0
        and time.microsecond == 0
    )


def scheduled_pick_between(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    weekdays: Collection[int],
    local_hour: int,
    timezone: str = "Asia/Shanghai",
) -> pd.Timestamp | None:
    """Return the first scheduled event in the half-open interval ``(start, end]``."""
    valid_weekdays, hour = _validate_schedule(weekdays, local_hour)
    start_local = _local_timestamp(start, timezone)
    end_local = _local_timestamp(end, timezone)
    if pd.isna(start_local) or pd.isna(end_local):
        raise ValueError("start and end timestamps must be valid")
    if end_local <= start_local:
        raise ValueError("end must be later than start")
    for date in pd.date_range(start_local.normalize(), end_local.normalize(), freq="D"):
        candidate = date + pd.Timedelta(hours=hour)
        if candidate.weekday() in valid_weekdays and start_local < candidate <= end_local:
            return candidate
    return None


def _validate_schedule(
    weekdays: Collection[int],
    local_hour: int,
) -> tuple[set[int], int]:
    if isinstance(local_hour, bool) or not isinstance(local_hour, int):
        raise ValueError("local_hour must be an integer")
    if not 0 <= local_hour <= 23:
        raise ValueError("local_hour must be between 0 and 23")
    valid_weekdays: set[int] = set()
    for day in weekdays:
        if isinstance(day, bool) or not isinstance(day, int):
            raise ValueError("weekdays must contain integers from 0 through 6")
        valid_weekdays.add(day)
    if not valid_weekdays.issubset(range(7)):
        raise ValueError("weekdays must contain integers from 0 through 6")
    return valid_weekdays, local_hour


def _local_timestamp(timestamp: pd.Timestamp, timezone: str) -> pd.Timestamp:
    time = pd.Timestamp(timestamp)
    if pd.isna(time):
        return time
    if time.tzinfo is None:
        return time.tz_localize(timezone)
    return time.tz_convert(timezone)
