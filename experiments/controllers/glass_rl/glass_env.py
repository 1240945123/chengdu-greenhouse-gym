"""GlassGreenhouse 离散档位 gym 环境（可行性验证版）。

动作空间：MultiDiscrete([2,2,2,3,2,2,4,2,2]) = 9 执行器档位
  [外遮阳, 顶保温, 四周保温, 顶窗, 补光, CO2, 风机, 水泵, 卷膜]
档位 -> 连续控制映射到 GlassGreenhouse 10 维 u。

reward（综合最优）：
  reward = yield_term(果实生长) - 温度惩罚 - 湿度惩罚 - 能耗惩罚
  * 温度白天 [20,28] / 夜间 [16,24]，湿度 [60,85]
  * 能耗按执行器动作成本（风机/水泵/补光/CO2 电费）

天气：从 greenhouse_1h.csv 的室外列驱动（固定序列）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import casadi as ca
import gymnasium as gym
from gymnasium import spaces

from glassgym.models.GlassGreenhouse.utils import define_model
from glassgym.environments.utils import init_state, vaporPres2rh, rh2vaporDens, vaporDens2pres, satVp
from glassgym.configs.default_params import init_default_params

DATA = "data/processed/chengdu_agri/greenhouse_001/aligned/greenhouse_1h.csv"
CALIB = "results/chengdu_agri_greenhouse_001/real_greenhouse/glass_calibrated_params.json"

# 执行器档位大小（顺序同动作空间）
LEVEL_DIMS = [2, 2, 2, 3, 2, 2, 4, 2, 2]
# 档位 -> u 连续值映射
ROOF_VENT_LEVEL = {0: 0.0, 1: 0.5, 2: 1.0}  # 顶窗 3 档: 关/半/全
FAN_LEVEL = {0: 0.0, 1: 0.25, 2: 0.5, 3: 1.0}  # 风机 MultiDiscrete 4 档(0-3) -> 真实档 {0,1,2,4台} 归一化


class GlassGreenhouseEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        params_path: str = CALIB,
        weather_path: str = DATA,
        start_day_index: int = 0,
        day_indices: list[int] | None = None,
        episode_days: int = 7,
        dt_seconds: int = 3600,
        yield_weight: float = 1.0,
        temperature_weight: float = 1.0,
        humidity_weight: float = 1.0,
        effort_weight: float = 0.2,
        cooling_weight: float = 0.5,
        cooling_mode: str = "overheat",
        obs_include_outdoor: bool = True,
        n_forecast_hours: int = 0,
        screen_shade_weight: float = 0.5,
        comfort_weight: float = 0.0,
        smooth_weight: float = 0.0,
        crop_start: str = "mature",
        disable_supplements: bool = False,
    ) -> None:
        super().__init__()
        self.dt = int(dt_seconds)
        self.episode_days = int(episode_days)
        self.action_space = spaces.MultiDiscrete(LEVEL_DIMS)
        # obs: [t_air, rh, co2, t_can, hour, fruit, tsum, (t_out),
        #       (未来 n 小时 × [辐射, 室外温度])]
        self.n_forecast_hours = int(n_forecast_hours)
        n_obs = (8 if obs_include_outdoor else 7) + 2 * self.n_forecast_hours
        self.observation_space = spaces.Box(
            low=-1e6, high=1e6, shape=(n_obs,), dtype=np.float32
        )
        self.obs_include_outdoor = bool(obs_include_outdoor)
        self.cooling_mode = cooling_mode
        self.screen_shade_weight = float(screen_shade_weight)
        # 舒适带内正奖励权重：温湿联合达标即 +comfort_weight/步。
        # 原 reward 只有"越界惩罚"（dist），进入舒适带后无梯度引导"居中最优"，
        # 导致策略在带边缘反复穿越。加正奖励直接对齐"舒适率"评价指标。
        self.comfort_weight = float(comfort_weight)
        # 动作平滑惩罚权重：惩罚相邻时刻执行器档位的跳变（chattering）。
        # 主要针对 SAC 连续动作投影到离散档位后的高频抖动（湿度在干/湿间摆振）。
        self.smooth_weight = float(smooth_weight)
        self._prev_u = None

        # 参数
        self.params = np.asarray(init_default_params(228), dtype=float)
        import json
        cal = json.load(open("results/chengdu_agri_greenhouse_001/physics_v4_daily_balance/selected_params.json"))
        for k, v in cal["multipliers"].items():
            self.params[int(k)] = float(v)
        glass = json.load(open(params_path))
        for k, v in glass["multipliers"].items():
            self.params[int(k)] = float(v)
        self.params[219] = 0.0005
        self.params[226] = 0.2
        # 外遮阳网（非遮光幕）透光率。默认 p[89]=p[90]=0.01 是 blackout screen
        # （催花遮光幕）的 1% 透光，但真实 GH63 外遮阳是降温遮阳网（透光 ~25%），
        # 与 GlassGreenhouse 能量模型的 `(1 - 0.75*u_bl_scr)` 一致。若沿用 0.01，
        # 白天一开遮阳 PAR 被挡掉 99%，光合从 ~23 g/m2/天崩到 3-6 g/m2/天。
        self.params[89] = 0.25   # tauBlScrNir 遮阳网 NIR 透光率
        self.params[90] = 0.25   # tauBlScrPar 遮阳网 PAR 透光率

        # 郫都作物参数标定（对齐 2026 郫都基地实测 3/30、4/27、5/19 茎/叶/果干重，
        # 密度 2.71 株/m²，误差从 40% 降到 ~6%）。详见 docs/notes/05-作物参数标定.md
        self.params[68] = 0.85   # tauRfNir 玻璃 NIR 透光率（原塑料棚 0.57）
        self.params[69] = 0.85   # tauRfPar 玻璃 PAR 透光率
        self.params[137] = 0.385 * 1.7    # alpha 量子效率（原 0.385 偏低）
        self.params[152] = 3.47e-7 * 0.5  # cLeafM 叶维持呼吸（原 3%/天偏高）
        self.params[154] = 0.328 * 1.15   # rgFruit 果实分配率
        self.params[156] = 0.074 * 2.0    # rgStem 茎分配率（成都番茄茎粗，原值偏低）

        # 修正 solar_gain_scale：glass 标定值 0.910165 是错误拟合「清棚后 59.6°C
        # 异常值」的结果，导致生长季 4-6 月温室被假加热 ~20°C（仿真 57-64°C vs
        # 真实 37-42°C）。用生长季（4/1-7/10，人工正常操作）逐日最高温重新标定，
        # 最优值 ≈0.42（生长季 RMSE 4.4°C，各月峰值差 1-2°C）。7 月清棚后 59.6°C
        # 属无人温室异常态，不纳入标定。详见 docs/notes/07-物理模型通风不足外推失效.md
        self.params[209] = 0.42   # solar_gain_scale 修正


        # 模型
        self.F = define_model(nx=28, nu=10, nd=11, n_params=228, dt=self.dt)

        # 天气（室外列）: global_radiation, outdoor_temp, co2(400), wind, sky
        w = pd.read_csv(weather_path)
        w["timestamp"] = pd.to_datetime(w["timestamp"])
        self.weather = w[["timestamp", "global_radiation", "outdoor_air_temperature",
                          "outdoor_relative_humidity", "wind_speed"]].copy()
        self.weather.columns = ["timestamp", "rad", "temp", "rh", "wind"]
        self.weather["co2"] = 420.0

        # reward 权重
        self.yield_weight = float(yield_weight)
        self.temperature_weight = float(temperature_weight)
        self.humidity_weight = float(humidity_weight)
        self.effort_weight = float(effort_weight)
        self.cooling_weight = float(cooling_weight)
        self._last_t_out = 20.0  # 室外温度缓存（降温奖励用）

        self.start_day_index = int(start_day_index)
        self.day_indices = (
            list(day_indices) if day_indices is not None else None
        )
        self.crop_start = crop_start
        self.disable_supplements = bool(disable_supplements)
        self.x = None
        self._step_in_episode = 0
        self._N_steps = 24 * self.episode_days  # 24 步/天 @1h（匹配逐时天气）
        self._prev_fruit = None
        self._hour = 0.0

    def _reset_crop(self):
        d0 = np.zeros(11)
        d0[0] = 100.0; d0[1] = 20.0; d0[3] = 420.0; d0[6] = 18.0
        x = init_state(d0)
        if self.crop_start == "early_fruiting":
            # 5 月初坐果初期：叶/茎中等、果实少量、发育已完成(模型域内稳定)
            x[22] = 15000.0   # cBuf 启动缓冲
            x[23] = 30000.0   # cLeaf
            x[24] = 50000.0   # cStem
            x[25] = 5000.0    # cFruit
            x[26] = 620.0     # tCanSum：定植(4/1)后约30天发育积温(~20°C×30d)
        elif self.crop_start == "seedling":
            # 定植苗（GH63 实测构建）。tCanSum 从 0 起：发育(果实汇)随积温逐步开启，
            # 避免成熟发育量在幼苗期就打开果实汇、与叶争夺碳源导致亏碳。
            x[22] = 2000.0    # cBuf 定植苗储备（约3天幼苗光合量）
            x[23] = 5842.0    # cLeaf
            x[24] = 7323.0    # cStem
            x[25] = 0.0       # cFruit
            x[26] = 0.0       # tCanSum 从定植起算
        return x

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.x = self._reset_crop()
        self._step_in_episode = 0
        self._prev_fruit = float(self.x[25])
        self._prev_u = None
        # 从天气序列起点开始（多场景时随机选）
        if self.day_indices is not None:
            self._day_index = int(np.random.choice(self.day_indices))
        else:
            self._day_index = self.start_day_index
        self._w_idx = self._day_index * 24
        self._hour = 0.0
        obs = self._get_obs()
        return obs, {}

    def _action_to_u(self, action) -> np.ndarray:
        a = np.asarray(action, dtype=int)
        u = np.zeros(10)
        u[0] = 0.0                                  # uBoil 无锅炉
        u[1] = 0.0 if self.disable_supplements else float(a[5])  # uCO2 CO2发生器
        u[2] = float(a[1])                          # uThScr 顶保温
        u[3] = ROOF_VENT_LEVEL[int(a[3])]           # uRoofVent 顶窗
        u[4] = 0.0 if self.disable_supplements else float(a[4])  # uLamp 补光
        u[5] = float(a[0])                          # uBlScr 外遮阳
        u[6] = FAN_LEVEL[int(a[6])]                 # uFan 风机
        u[7] = float(a[7])                          # uPad 水泵
        u[8] = float(a[2])                          # uSideScr 四周保温
        u[9] = float(a[8])                          # uPadCurtain 卷膜
        return u

    def _weather_at(self, idx: int) -> np.ndarray:
        i = min(max(int(idx), 0), len(self.weather) - 1)
        row = self.weather.iloc[i]
        rad = float(row["rad"]) if pd.notna(row["rad"]) else 0.0
        temp = float(row["temp"]) if pd.notna(row["temp"]) else 20.0
        rh = float(row["rh"]) if pd.notna(row["rh"]) else 70.0
        wind = float(row["wind"]) if pd.notna(row["wind"]) else 0.0
        d = np.zeros(11)
        d[0] = max(rad, 0.0)
        d[1] = temp
        d[2] = vaporDens2pres(temp, rh2vaporDens(temp, rh))
        d[3] = 420.0
        d[4] = max(wind, 0.0)
        d[5] = temp - 8.0
        d[6] = 18.0
        return d

    def _get_obs(self) -> np.ndarray:
        t = float(self.x[2]); rh = float(vaporPres2rh(self.x[2], self.x[15]))
        obs = [
            t, rh, float(self.x[0]), float(self.x[4]),  # 温度 湿度 CO2 冠层
            float(self._hour / 24.0),                    # 时间
            float(self.x[25]) / 1e5,                     # 果实生物量(归一化)
            float(self.x[26]) / 1e3,                     # 积温
        ]
        if self.obs_include_outdoor:
            obs.append(float(self._last_t_out))           # 室外温度（前瞻控制）
        # 未来 n 小时（含当前时刻）的辐射与室外温度预报。
        # 动机（见 docs/notes/15 / P0-3 归因诊断）：原 obs 里唯一的室外信息是
        # _last_t_out（**上一小时**的值，滞后 1h），而模型预测类控制器直接读
        # 当前/未来天气。蒸馏实验证明仅补齐这一信息赤字，学生舒适率就从
        # 44.6% 升到 61.1%。RL 侧同理需要同等信息才能做前瞻控制。
        n = int(getattr(self, "n_forecast_hours", 0))
        for k in range(n):
            dk = self._weather_at(self._w_idx + k)
            obs.append(float(dk[0]) / 1000.0)  # 辐射 W/m2 -> kW/m2
            obs.append(float(dk[1]))           # 室外温度 °C
        return np.array(obs, dtype=np.float32)

    def _integrate(self, u: np.ndarray):
        """物理积分 + reward（u 为 10 维连续控制）。

        积分失败保护：作物缓冲池在高温/病态动作下可能数值发散
        （CV_CONV_FAILURE / CV_TOO_MUCH_WORK）。此时保持状态、推进时间、
        返回 -10 惩罚，让 RL 学会避开导致发散的动作，训练不中断。
        """
        d = self._weather_at(self._w_idx)
        try:
            r = self.F(x0=ca.DM(self.x), u=ca.DM(u),
                       p=ca.vertcat(ca.DM(d), ca.DM(self.params)))
            self.x = np.asarray(r["xf"]).flatten()
        except Exception:
            self._w_idx += 1
            self._step_in_episode += 1
            self._hour = (self._hour + self.dt / 3600.0) % 24.0
            self._prev_u = np.asarray(u, dtype=float).copy()
            terminated = self._step_in_episode >= self._N_steps
            info = {
                "integration_failed": True,
                "temperature": float(self.x[2]),
                "rh": float(vaporPres2rh(self.x[2], self.x[15])),
                "temp_penalty": 0.0, "yield_term": 0.0,
                "effort": 0.0, "fruit_mg": float(self.x[25]),
                "hour": self._hour,
            }
            return self._get_obs(), -10.0, terminated, False, info

        self._w_idx += 1
        self._step_in_episode += 1
        self._hour = (self._hour + self.dt / 3600.0) % 24.0

        temperature = float(self.x[2])
        rh = float(vaporPres2rh(self.x[2], self.x[15]))
        is_day = 6.0 <= self._hour < 20.0
        t_lo, t_hi = (20.0, 28.0) if is_day else (16.0, 24.0)
        t_dist = max(t_lo - temperature, 0.0) + max(temperature - t_hi, 0.0)
        rh_dist = max(60.0 - rh, 0.0) + max(rh - 85.0, 0.0)
        temp_pen = self.temperature_weight * t_dist / 10.0
        rh_pen = self.humidity_weight * rh_dist / 40.0

        fruit_gain = max(0.0, float(self.x[25]) - self._prev_fruit)
        yield_term = self.yield_weight * fruit_gain * 1e-6 / 0.081 * 100.0
        self._prev_fruit = float(self.x[25])

        # 能耗（基于连续 u，人工/RL 公平对比）
        effort = self.effort_weight * (
            0.15*u[6] + 0.10*u[7] + 0.05*u[5] + 0.20*u[4] + 0.15*u[1] + 0.05*u[3]
        )
        # 降温奖励（两种模式）：
        # - 'outdoor'（旧版基线）：室内比室外低多少就奖励多少。缺点是温度已经
        #   低于舒适区时仍持续奖励，诱导 RL 无限降温 + 过度开湿帘，导致湿度 85-100%。
        # - 'overheat'（优化版）：只在温度超过舒适区上限时才奖励降温，温度已
        #   舒适（≤ t_hi）即不再奖励，避免"过度降温 + 湿帘加湿"。
        t_out = float(d[1])
        self._last_t_out = t_out  # 缓存室外温度，供 obs 前瞻使用
        if self.cooling_mode == "outdoor":
            cooling_bonus = self.cooling_weight * max(0.0, t_out - temperature) / 10.0
        else:  # "overheat"
            cooling_bonus = self.cooling_weight * max(0.0, temperature - t_hi) / 10.0
        # 白天顶保温幕挡光惩罚（分温度）：顶保温展开时作物模型按 1-u[2]*(1-p[80])
        # 挡掉 25% PAR（p[80]=0.75）。但只有「温度已 ≥ 舒适下限」时，白天开保温才
        # 是纯挡光损失（已经够暖，不需保温）；温度低于舒适下限（如 4 月幼苗期冷）
        # 时，白天保温收益 > 挡光损失，不应罚。分温度惩罚让 PPO 学会「冷时保温、
        # 暖时拉开顶保温」。注：四周保温 u[8] 不在作物模型 u[:6] 内，不挡光。
        screen_pen = self.screen_shade_weight * is_day * u[2] * float(temperature >= t_lo)
        # 舒适带内正奖励：温湿联合达标即给正奖励（对齐"舒适率"评价指标），
        # 让策略有"停留在带内"的正向梯度，而非仅在越界时被罚。
        comfort_bonus = self.comfort_weight * float(
            (t_lo <= temperature <= t_hi) and (60.0 <= rh <= 85.0)
        )
        # 动作平滑惩罚：抑制相邻时刻档位跳变（chattering）。仅统计可调执行器
        # 通道（1-9），CO2 通道 0 不参与。
        u_now = np.asarray(u, dtype=float)
        if self.smooth_weight > 0.0 and self._prev_u is not None:
            smooth_pen = self.smooth_weight * float(np.mean(np.abs(u_now[1:] - self._prev_u[1:])))
        else:
            smooth_pen = 0.0
        self._prev_u = u_now.copy()
        reward = (yield_term - temp_pen - rh_pen - effort
                  + cooling_bonus - screen_pen + comfort_bonus - smooth_pen)

        terminated = self._step_in_episode >= self._N_steps
        info = {
            "integration_failed": False,
            "temperature": temperature, "rh": rh,
            "temp_penalty": temp_pen, "yield_term": yield_term,
            "effort": effort, "fruit_mg": float(self.x[25]),
            "cooling_bonus": float(cooling_bonus),
            "comfort_bonus": float(comfort_bonus),
            "hour": self._hour,
        }
        return self._get_obs(), float(reward), terminated, False, info

    def step(self, action):
        return self._integrate(self._action_to_u(action))

    def step_with_u(self, u: np.ndarray):
        """直接注入连续控制 u（人工 benchmark 用）。"""
        u = np.asarray(u, dtype=float).reshape(10)
        return self._integrate(u)


if __name__ == "__main__":
    env = GlassGreenhouseEnv(episode_days=1, start_day_index=200)
    obs, _ = env.reset(seed=0)
    print("action_space:", env.action_space)
    print("obs:", obs)
    # 随机策略跑
    total = 0.0
    for _ in range(20):
        act = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(act)
        total += reward
    print(f"随机策略 20 步 reward 累计: {total:.2f}")
    print(f"温度 {info['temperature']:.1f}°C, RH {info['rh']:.0f}%, fruit {info['fruit_mg']:.0f}mg")
