from abc import ABC, abstractmethod
from typing import SupportsFloat, Dict
import numpy as np
from glassgym.core.types import RewardContext
from glassgym.components.price_model import FixedPrice

class BaseReward(ABC):
    @abstractmethod
    def compute_reward(self, ctx: RewardContext) -> tuple[SupportsFloat, Dict[str, float]]:
        ...

    @staticmethod
    def scale_reward(r: float, min_r: float, max_r: float) -> float:
        return (r - min_r) / (max_r - min_r)

class GreenhouseReward(BaseReward):
    """	
    Economic reward function for the GreenLight environment.
    The reward is computed as the difference between the gains and the costs.
    The gains are computed as the fruit growth per pot per day multiplied by the fruit price.
    The costs are computed as the sum of the heating, co2, off peak and on peak electricity costs.

    Args:
        elec_price (float): price for off peak electricity [€/kWh]
        heating_price (float): price for heating [€/kWh]
        co2_price (float): price for co2 [€/kg]
        fruit_price (float): price for the fruit [€/kg]
        dmfm (float): ration of dry matter to fresh matter
        pen_lamp (float): weight for the lamp violation
    """

    def __init__(
                self,
                elec_price: float,
                heating_price: float,
                co2_price: float,
                fruit_price: float,
                pen_lamp: float,
                dmfm: float,
                dt: int,
                p: np.ndarray,
                ) -> None:
        super(GreenhouseReward, self).__init__()


        ctx = RewardContext(
            t=0,
            dt=dt,
            Np=0,
            x_prev=np.zeros(26),
            x=np.zeros(26),
            u=np.zeros(6),
            p=p,
            d=np.zeros(10),
            obs={},
            day_of_year=0,
            hour_of_day=0,
        )

        # variable prices for the electricity, heating co2
        self.elec_price_model = FixedPrice(elec_price)
        self.heating_price_model = FixedPrice(heating_price)
        self.co2_price_model = FixedPrice(co2_price)
        self.fruit_price_model = FixedPrice(fruit_price)

        self.dmfm = dmfm            # ratio of dry matter to fresh matter; Assumption
        self.pen_lamp = pen_lamp
        self._init_costs()
        self._init_violations()
        self.max_profit = self.max_profit_reward(ctx)
        self.min_profit = self.min_profit_reward(ctx)
        self.min_state_violations = self.min_violations()
        self.max_state_violations = self.max_violations()

    def min_violations(self):
        return np.zeros(3)

    def max_violations(self):
        co2_violation = 2500
        temp_violation = 15
        rh_violation = 15
        return np.array([co2_violation, temp_violation, rh_violation])

    def max_profit_reward(self, ctx: RewardContext) -> float:
        """
        Computes the maximum possible reward for the current timestep.
        The maximum reward is computed as the maximum possible gains minus the minimum possible costs.
        The maximum possible gains are computed as the maximum fruit growth per pot per day multiplied by the fruit price.
        The minimum possible costs are computed as the sum of the heating, co2, off peak and on peak electricity costs.
        Returns:
            float: The maximum possible reward.
        """
        max_gains = ctx.p[154] * ctx.dt * 1e-6 /self.dmfm * self.fruit_price_model.get_price(ctx)
        return max_gains

    def min_profit_reward(self, ctx: RewardContext) -> float:
        """
        Computes the minimum possible reward for the current timestep.
        The minimum reward is computed as the minimum possible gains minus the maximum possible costs.
        The minimum possible gains are computed as the minimum fruit growth per pot per day multiplied by the fruit price.
        The maximum possible costs are computed as the sum of the heating, co2, off peak and on peak electricity costs.
        Returns:
            float: The minimum possible reward.
        """
        max_heating = ctx.p[108] / ctx.p[46] * ctx.dt/3600*1e-3 * self.heating_price_model.get_price(ctx)     # convert W/aFlr to kWh/m2
        max_elec = ctx.p[172] * ctx.dt/3600*1e-3 * self.elec_price_model.get_price(ctx)                          # convert W/aFlr to kWh/m2
        max_cost = ctx.p[109] / ctx.p[46] * ctx.dt * 1e-6 * self.co2_price_model.get_price(ctx)        # convert to kg/m2
        max_costs = sum([max_heating, max_elec, max_cost])
        max_costs = -max_costs
        return max_costs

    def _init_violations(self):
        self.temp_violation = 0
        self.co2_violation = 0
        self.rh_violation = 0
        self.lamp_violation = 0

    def _init_costs(self):
        self.variable_costs = 0
        self.gains = 0
        self.profit = 0
        self.heat_costs = 0
        self.co2_costs = 0
        self.elec_costs = 0

    def _variable_costs(self, ctx: RewardContext) -> tuple[float, dict[str, float]]:
        """
        Calculate the variable costs based on the given GreenLight model.
        These costs reflect the daily variable costs for heating, co2, off peak and on peak electricity.
        Has the same unit as the gains, which are computed as €/m2/day.
        Returns:
            float: The total variable costs.
        """
        heating_energy = ctx.u[0] * ctx.p[108] / ctx.p[46] * ctx.dt/3600 * 1e-3   # convert W/aFlr to kWh/m2
        elec_use = ctx.u[4] * ctx.p[172] * ctx.dt/3600*1e-3                          # convert W/aFlr to kWh/m2
        co2_dosing = ctx.u[1] * ctx.p[109] / ctx.p[46] * ctx.dt * 1e-6          # convert to kg/m2
        self.heat_cost = heating_energy * self.heating_price_model.get_price(ctx)
        self.co2_cost = co2_dosing * self.co2_price_model.get_price(ctx)
        self.elec_cost = elec_use * self.elec_price_model.get_price(ctx)
        return sum([self.heat_cost, self.co2_cost, self.elec_cost]), {
            "heat_cost": self.heat_cost,
            "co2_cost": self.co2_cost,
            "elec_cost": self.elec_cost,
        }

    def _gains(self, ctx: RewardContext) -> tuple[float, dict[str, float]]:
        """
        Computes the daily gains based on the given GreenLight model.
        These gains are computed as the gains per pot per day.
        Does the following steps:
        1. Computes the fruit growth in dry weight (DW) in (mg/m2)
        2. Converts the fruit DW to fruit fresh weight (FFW) in (kg/m2) using dmfm conversion factor
        3. Multiplies the daily FFW growth by the fruit price, which resembles €/kg.
        """
        fruit_growth_dm = ctx.x[25] - ctx.x_prev[25]
        fruit_growth_ffw = fruit_growth_dm * 1e-6 / self.dmfm
        fruit_price = float(self.fruit_price_model.get_price(ctx))
        gains = fruit_growth_ffw * fruit_price
        return gains, {
            "fruit_growth_dm": float(fruit_growth_dm),
            "fruit_growth_ffw": float(fruit_growth_ffw),
            "fruit_price": fruit_price,
            "revenue": gains,
        }

    def _output_violations(self, ctx: RewardContext) -> tuple[float, dict[str, float]]:
        """
        Function that computes the absolute penalties for violating system constraints.
        System constraints are currently non-dynamical, and based on observation bounds of gym environment.
        We do not look at dry mass bounds, since those are non-existent in real greenhouse.
        """
        indoor_climate_obs = np.asarray(ctx.obs["IndoorClimateObservations"][:3], dtype=np.float64)
        lowerbound = ctx.constraints_low[:] - indoor_climate_obs
        lowerbound[lowerbound < 0] = 0
        upperbound = indoor_climate_obs - ctx.constraints_high[:]
        upperbound[upperbound < 0] = 0
        co2_violation = lowerbound[0] + upperbound[0]
        temp_violation = lowerbound[1] + upperbound[1]
        rh_violation = lowerbound[2] + upperbound[2]
        return lowerbound+upperbound

    def _control_violation(self, ctx: RewardContext) -> tuple[float, dict[str, float]]:
        """
        Checks if lamps are used during night hours (after 8 PM).
        Sets lamp_violation to 1 if lamps are on after 20:00,
        otherwise sets it to 0.
        """
        if ctx.hour_of_day >= 20:
            if ctx.u[4] > 0:
                self.lamp_violation = 1
        lamp_violation = 0
        return lamp_violation

    def _control_penalty(self, ctx: RewardContext) -> tuple[float, dict[str, float]]:
        lamp_violation = self._control_violation(ctx)
        return lamp_violation * self.pen_lamp, {
            "lamp_penalty": lamp_violation * self.pen_lamp,
        }

    def compute_reward(self, ctx: RewardContext) -> SupportsFloat:
        variable_costs, variable_costs_info = self._variable_costs(ctx)
        gains, gains_info = self._gains(ctx)
        profit = gains - variable_costs

        violations = self._output_violations(ctx)
        control_penalty, control_penalty_info = self._control_penalty(ctx)

        scaled_profit = self.scale_reward(profit, self.min_profit, self.max_profit)
        scaled_pen = self.scale_reward(violations, self.min_state_violations, self.max_state_violations)
        reward = scaled_profit - sum(scaled_pen) - control_penalty
        info = {
            "reward": reward,
            "profit": profit,
            "control_penalty": control_penalty,
            "penalty": sum(scaled_pen),
            "variable_costs": variable_costs,
            "gains": gains,
            "temp_penalty": scaled_pen[0],
            "co2_penalty": scaled_pen[1],
            "rh_penalty": scaled_pen[2],
            **variable_costs_info,
            **gains_info,
            **control_penalty_info,
        }
        return reward, info


class ChengduClimateReward(BaseReward):
    """Climate-control reward for the Chengdu greenhouse benchmark.

    Heating and CO2 injection are intentionally excluded. The reward is zero in
    the configured comfort bands with idle controls and negative otherwise.
    """

    ALLOWED_CONTROL_INDICES = np.array([3, 5, 6, 7], dtype=int)

    def __init__(
        self,
        dt: int,
        p: np.ndarray,
        temp_day_low: float = 20.0,
        temp_day_high: float = 28.0,
        temp_night_low: float = 16.0,
        temp_night_high: float = 24.0,
        rh_low: float = 60.0,
        rh_high: float = 85.0,
        day_start: float = 6.0,
        day_end: float = 20.0,
        temperature_weight: float = 1.0,
        humidity_weight: float = 1.0,
        lamp_weight: float = 0.1,
        effort_weight: float = 0.2,
        action_change_weight: float = 0.3,
    ) -> None:
        del dt, p
        if not temp_day_low < temp_day_high:
            raise ValueError("temp_day_low must be lower than temp_day_high")
        if not temp_night_low < temp_night_high:
            raise ValueError("temp_night_low must be lower than temp_night_high")
        if not rh_low < rh_high:
            raise ValueError("rh_low must be lower than rh_high")

        self.temp_day_low = float(temp_day_low)
        self.temp_day_high = float(temp_day_high)
        self.temp_night_low = float(temp_night_low)
        self.temp_night_high = float(temp_night_high)
        self.rh_low = float(rh_low)
        self.rh_high = float(rh_high)
        self.day_start = float(day_start)
        self.day_end = float(day_end)
        self.temperature_weight = float(temperature_weight)
        self.humidity_weight = float(humidity_weight)
        self.lamp_weight = float(lamp_weight)
        self.effort_weight = float(effort_weight)
        self.action_change_weight = float(action_change_weight)

    @staticmethod
    def _distance_outside(value: float, low: float, high: float) -> float:
        return max(low - value, 0.0) + max(value - high, 0.0)

    def compute_reward(self, ctx: RewardContext) -> tuple[SupportsFloat, Dict[str, float]]:
        indoor = np.asarray(ctx.obs["IndoorClimateObservations"], dtype=np.float64)
        temperature = float(indoor[1])
        humidity = float(indoor[2])
        is_day = self.day_start <= float(ctx.hour_of_day) < self.day_end
        if is_day:
            temp_low, temp_high = self.temp_day_low, self.temp_day_high
        else:
            temp_low, temp_high = self.temp_night_low, self.temp_night_high

        controls = np.asarray(ctx.u, dtype=np.float64)
        # 控制向量维度自适应：V5 环境 8 维（uVent/uBlScr/uFan/uPad），
        # 旧 benchmark 环境 6 维（uThScr/uVent/uLamp/uBlScr）。
        if controls.shape[0] >= 8:
            allowed_idx = self.ALLOWED_CONTROL_INDICES
        else:
            allowed_idx = np.array([2, 3, 4, 5], dtype=int)
        allowed = controls[allowed_idx]
        previous = controls if ctx.u_prev is None else np.asarray(ctx.u_prev, dtype=np.float64)
        previous_allowed = previous[allowed_idx]

        temperature_penalty = self.temperature_weight * (
            self._distance_outside(temperature, temp_low, temp_high) / 10.0
        )
        humidity_penalty = self.humidity_weight * (
            self._distance_outside(humidity, self.rh_low, self.rh_high) / 40.0
        )
        lamp_penalty = self.lamp_weight * float(controls[4])
        effort_penalty = self.effort_weight * float(np.mean(np.abs(allowed)))
        action_change_penalty = self.action_change_weight * float(
            np.mean(np.abs(allowed - previous_allowed))
        )
        total_penalty = (
            temperature_penalty
            + humidity_penalty
            + lamp_penalty
            + effort_penalty
            + action_change_penalty
        )
        reward = -float(total_penalty)
        return reward, {
            "reward": reward,
            "temperature_penalty": float(temperature_penalty),
            "humidity_penalty": float(humidity_penalty),
            "lamp_penalty": float(lamp_penalty),
            "effort_penalty": float(effort_penalty),
            "action_change_penalty": float(action_change_penalty),
            "temperature_target": 0.5 * (temp_low + temp_high),
            "humidity_target": 0.5 * (self.rh_low + self.rh_high),
            "temperature_low": float(temp_low),
            "temperature_high": float(temp_high),
            "humidity_low": self.rh_low,
            "humidity_high": self.rh_high,
        }


class ChengduYieldClimateReward(BaseReward):
    """Yield + comfort + effort reward for the Chengdu low-tech greenhouse.

    Extends ``ChengduClimateReward`` with an explicit fruit-yield incentive so the
    RL agent learns to *produce* rather than merely hold the climate inside the
    comfort band. The reward decomposes as

        reward = yield_weight * scaled_yield_gain
                 - temperature_penalty
                 - humidity_penalty
                 - effort_penalty
                 - action_change_penalty

    where ``scaled_yield_gain`` is the per-step ripe-fruit fresh-weight increment
    (kg/m2) normalised by the theoretical maximum growth rate, so it lies in
    [0, 1]. Heating, CO2 injection and lighting are intentionally excluded, and
    the two optimised V5 actuators (uVent, uPadFan) are both effort-penalised.
    """

    # V5 optimised actuators: uVent (3) + uBlScr (5) + uPadFan (6) + uPadPump (7)
    ALLOWED_CONTROL_INDICES = np.array([3, 5, 6, 7], dtype=int)

    def __init__(
        self,
        dt: int,
        p: np.ndarray,
        dmfm: float = 0.081,
        temp_day_low: float = 20.0,
        temp_day_high: float = 28.0,
        temp_night_low: float = 16.0,
        temp_night_high: float = 24.0,
        rh_low: float = 60.0,
        rh_high: float = 85.0,
        day_start: float = 6.0,
        day_end: float = 20.0,
        yield_weight: float = 1.0,
        temperature_weight: float = 1.0,
        humidity_weight: float = 1.0,
        effort_weight: float = 0.2,
        action_change_weight: float = 0.3,
    ) -> None:
        if not temp_day_low < temp_day_high:
            raise ValueError("temp_day_low must be lower than temp_day_high")
        if not temp_night_low < temp_night_high:
            raise ValueError("temp_night_low must be lower than temp_night_high")
        if not rh_low < rh_high:
            raise ValueError("rh_low must be lower than rh_high")
        if not np.isfinite(dmfm) or not 0.0 < dmfm < 1.0:
            raise ValueError("dmfm must be in (0, 1)")

        self.dmfm = float(dmfm)
        # Theoretical maximum per-step fruit fresh-weight gain (kg/m2), used to
        # normalise the yield term into [0, 1]. p[154] is the maximum fruit
        # growth rate in mg{CH2O} m-2 s-1.
        self.max_yield_gain = float(p[154]) * float(dt) * 1e-6 / self.dmfm
        if not np.isfinite(self.max_yield_gain) or self.max_yield_gain <= 0.0:
            raise ValueError("yield normalisation requires a positive maximum growth rate")

        self.temp_day_low = float(temp_day_low)
        self.temp_day_high = float(temp_day_high)
        self.temp_night_low = float(temp_night_low)
        self.temp_night_high = float(temp_night_high)
        self.rh_low = float(rh_low)
        self.rh_high = float(rh_high)
        self.day_start = float(day_start)
        self.day_end = float(day_end)
        self.yield_weight = float(yield_weight)
        self.temperature_weight = float(temperature_weight)
        self.humidity_weight = float(humidity_weight)
        self.effort_weight = float(effort_weight)
        self.action_change_weight = float(action_change_weight)

    @staticmethod
    def _distance_outside(value: float, low: float, high: float) -> float:
        return max(low - value, 0.0) + max(value - high, 0.0)

    def compute_reward(self, ctx: RewardContext) -> tuple[SupportsFloat, Dict[str, float]]:
        indoor = np.asarray(ctx.obs["IndoorClimateObservations"], dtype=np.float64)
        temperature = float(indoor[1])
        humidity = float(indoor[2])
        is_day = self.day_start <= float(ctx.hour_of_day) < self.day_end
        if is_day:
            temp_low, temp_high = self.temp_day_low, self.temp_day_high
        else:
            temp_low, temp_high = self.temp_night_low, self.temp_night_high

        # Fruit-yield increment: cFruit dry matter at index 25 (mg/m2). Clamped at
        # zero so a harvest-driven dip in standing fruit does not read as a penalty.
        fruit_dm_growth = max(0.0, float(ctx.x[25] - ctx.x_prev[25]))
        yield_gain_ffw = fruit_dm_growth * 1e-6 / self.dmfm
        yield_scaled = yield_gain_ffw / self.max_yield_gain
        yield_term = self.yield_weight * yield_scaled

        controls = np.asarray(ctx.u, dtype=np.float64)
        # 控制向量维度自适应：V5 环境 8 维（uVent/uBlScr/uFan/uPad），
        # 旧 benchmark 环境 6 维（uThScr/uVent/uLamp/uBlScr）。
        if controls.shape[0] >= 8:
            allowed_idx = self.ALLOWED_CONTROL_INDICES
        else:
            allowed_idx = np.array([2, 3, 4, 5], dtype=int)
        allowed = controls[allowed_idx]
        previous = controls if ctx.u_prev is None else np.asarray(ctx.u_prev, dtype=np.float64)
        previous_allowed = previous[allowed_idx]

        temperature_penalty = self.temperature_weight * (
            self._distance_outside(temperature, temp_low, temp_high) / 10.0
        )
        humidity_penalty = self.humidity_weight * (
            self._distance_outside(humidity, self.rh_low, self.rh_high) / 40.0
        )
        effort_penalty = self.effort_weight * float(np.mean(np.abs(allowed)))
        action_change_penalty = self.action_change_weight * float(
            np.mean(np.abs(allowed - previous_allowed))
        )
        total_penalty = (
            temperature_penalty
            + humidity_penalty
            + effort_penalty
            + action_change_penalty
        )
        reward = yield_term - total_penalty
        return reward, {
            "reward": reward,
            "yield_term": float(yield_term),
            "yield_gain_ffw": float(yield_gain_ffw),
            "fruit_dm_growth": fruit_dm_growth,
            "temperature_penalty": float(temperature_penalty),
            "humidity_penalty": float(humidity_penalty),
            "effort_penalty": float(effort_penalty),
            "action_change_penalty": float(action_change_penalty),
            "temperature_low": float(temp_low),
            "temperature_high": float(temp_high),
        }


REWARDS_MODULES = {
    "GreenhouseReward": GreenhouseReward,
    "ChengduClimateReward": ChengduClimateReward,
    "ChengduYieldClimateReward": ChengduYieldClimateReward,
}
