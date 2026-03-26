"""
Eco-Adaptive Home Storage Environment
=====================================
A Gymnasium-compatible environment for smart grid energy management.

State Space (5-dimensional continuous):
    - hour: Current hour [0, 23]
    - soc: Battery State of Charge [0, 1]
    - p_net: Net load (Solar - Consumption) in kW
    - c_grid: Grid electricity price in €/kWh
    - i_co2: Carbon intensity in gCO2/kWh

Action Space (Discrete):
    - 0: Charge (battery from grid/PV)
    - 1: Discharge (battery to home/grid)
    - 2: Idle (no battery activity)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any


class SmartGridEnv(gym.Env):
    """
    Smart Grid Home Energy Management Environment.

    The agent controls a home battery system to minimize electricity costs,
    carbon emissions, and battery degradation.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        battery_capacity: float = 10.0,  # kWh
        max_charge_rate: float = 3.0,    # kW
        max_discharge_rate: float = 3.0, # kW
        battery_efficiency: float = 0.95,
        time_step_minutes: int = 30,
        episode_hours: int = 24,
        alpha: float = 1.0,   # Cost weight
        beta: float = 0.5,    # Emissions weight
        gamma_wear: float = 0.1,  # Battery wear weight
        render_mode: Optional[str] = None,
        seed: Optional[int] = None
    ):
        super().__init__()

        # Battery parameters
        self.battery_capacity = battery_capacity  # kWh
        self.max_charge_rate = max_charge_rate    # kW
        self.max_discharge_rate = max_discharge_rate  # kW
        self.battery_efficiency = battery_efficiency

        # Time parameters
        self.time_step_minutes = time_step_minutes
        self.time_step_hours = time_step_minutes / 60.0
        self.episode_hours = episode_hours
        self.max_steps = int(episode_hours * 60 / time_step_minutes)

        # Reward weights
        self.alpha = alpha
        self.beta = beta
        self.gamma_wear = gamma_wear

        # Action space: 0=Charge, 1=Discharge, 2=Idle
        self.action_space = spaces.Discrete(3)

        # State space: [hour, soc, p_net, c_grid, i_co2]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -10.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([23.0, 1.0, 10.0, 0.5, 800.0], dtype=np.float32),
            dtype=np.float32
        )

        self.render_mode = render_mode

        # Initialize state variables
        self.current_step = 0
        self.current_hour = 0.0
        self.soc = 0.5
        self.last_action = 2  # Start with Idle

        # Episode tracking
        self.episode_costs = []
        self.episode_emissions = []
        self.episode_soc_history = []
        self.episode_actions = []

        # Set random seed
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        else:
            self._np_random = np.random.default_rng()

    def _get_solar_production(self, hour: float) -> float:
        """
        Simulate solar panel production based on hour of day.
        Peak production around noon, zero at night.
        Returns production in kW.
        """
        if 6 <= hour <= 20:
            # Bell curve centered at 13:00
            peak_hour = 13.0
            width = 4.0
            max_production = 4.0  # kW peak
            production = max_production * np.exp(-((hour - peak_hour) ** 2) / (2 * width ** 2))
            # Add some randomness
            production *= (0.8 + 0.4 * self._np_random.random())
            return production
        return 0.0

    def _get_household_consumption(self, hour: float) -> float:
        """
        Simulate household electricity consumption based on hour.
        Higher consumption in morning and evening.
        Returns consumption in kW.
        """
        base_load = 0.5  # kW base load

        # Morning peak (7-9)
        morning_peak = 1.5 * np.exp(-((hour - 8) ** 2) / 2)

        # Evening peak (18-21)
        evening_peak = 2.5 * np.exp(-((hour - 19.5) ** 2) / 3)

        # Midday small peak
        midday = 0.5 * np.exp(-((hour - 12) ** 2) / 2)

        consumption = base_load + morning_peak + evening_peak + midday
        # Add randomness
        consumption *= (0.85 + 0.3 * self._np_random.random())

        return consumption

    def _get_grid_price(self, hour: float) -> float:
        """
        Simulate time-of-use electricity pricing.
        Returns price in €/kWh.
        """
        # Off-peak: 22:00 - 06:00 (night)
        # Peak: 07:00 - 21:00 (day)
        # Super-peak: 18:00 - 21:00 (evening)

        if 0 <= hour < 6 or hour >= 22:
            # Off-peak night rate
            return 0.12
        elif 18 <= hour < 21:
            # Super-peak evening rate
            return 0.35
        elif 7 <= hour < 18:
            # Day rate
            return 0.22
        else:
            # Transition periods
            return 0.18

    def _get_carbon_intensity(self, hour: float) -> float:
        """
        Simulate grid carbon intensity based on hour.
        Lower during day (more solar/wind), higher at night (more gas/coal).
        Returns intensity in gCO2/kWh.
        """
        # Base intensity
        base = 400  # gCO2/kWh

        # Lower during midday (solar generation)
        solar_reduction = 150 * np.exp(-((hour - 13) ** 2) / 8)

        # Higher during evening peak (gas peaker plants)
        peak_increase = 100 * np.exp(-((hour - 19) ** 2) / 4)

        intensity = base - solar_reduction + peak_increase
        # Add some randomness
        intensity *= (0.9 + 0.2 * self._np_random.random())

        return max(150, intensity)  # Minimum 150 gCO2/kWh

    def _get_observation(self) -> np.ndarray:
        """Construct the current observation vector."""
        p_solar = self._get_solar_production(self.current_hour)
        p_consumption = self._get_household_consumption(self.current_hour)
        p_net = p_solar - p_consumption  # Positive = surplus, Negative = deficit

        c_grid = self._get_grid_price(self.current_hour)
        i_co2 = self._get_carbon_intensity(self.current_hour)

        return np.array([
            self.current_hour,
            self.soc,
            p_net,
            c_grid,
            i_co2
        ], dtype=np.float32)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        self.current_step = 0
        self.current_hour = 0.0

        # Random initial SoC between 30% and 70%
        self.soc = 0.3 + 0.4 * self._np_random.random()
        self.last_action = 2

        # Reset tracking
        self.episode_costs = []
        self.episode_emissions = []
        self.episode_soc_history = [self.soc]
        self.episode_actions = []

        observation = self._get_observation()
        info = {"soc": self.soc, "hour": self.current_hour}

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one time step within the environment.

        Args:
            action: 0=Charge, 1=Discharge, 2=Idle

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Get current state values
        p_solar = self._get_solar_production(self.current_hour)
        p_consumption = self._get_household_consumption(self.current_hour)
        p_net = p_solar - p_consumption
        c_grid = self._get_grid_price(self.current_hour)
        i_co2 = self._get_carbon_intensity(self.current_hour)

        # Initialize energy flows
        grid_import = 0.0  # Energy bought from grid
        grid_export = 0.0  # Energy sold to grid
        battery_change = 0.0  # Change in battery energy

        # Process action
        if action == 0:  # Charge
            # Maximum energy we can charge in this time step
            max_charge_energy = self.max_charge_rate * self.time_step_hours
            # Available capacity in battery
            available_capacity = (1.0 - self.soc) * self.battery_capacity
            # Actual charge (limited by rate and capacity)
            charge_energy = min(max_charge_energy, available_capacity / self.battery_efficiency)

            if p_net > 0:
                # Use solar surplus first
                solar_to_battery = min(p_net * self.time_step_hours, charge_energy)
                remaining_charge = charge_energy - solar_to_battery
                grid_import = remaining_charge + max(0, p_consumption * self.time_step_hours - p_solar * self.time_step_hours)
            else:
                # No surplus, charge from grid
                grid_import = charge_energy + abs(p_net) * self.time_step_hours

            battery_change = charge_energy * self.battery_efficiency

        elif action == 1:  # Discharge
            # Maximum energy we can discharge
            max_discharge_energy = self.max_discharge_rate * self.time_step_hours
            # Available energy in battery
            available_energy = self.soc * self.battery_capacity
            # Actual discharge
            discharge_energy = min(max_discharge_energy, available_energy) * self.battery_efficiency

            # Use discharge to cover consumption deficit
            if p_net < 0:
                # There's a deficit
                deficit = abs(p_net) * self.time_step_hours
                battery_to_home = min(discharge_energy, deficit)
                grid_import = max(0, deficit - battery_to_home)
                # Export any remaining discharge
                grid_export = max(0, discharge_energy - battery_to_home)
            else:
                # Surplus exists, export discharge
                grid_export = discharge_energy

            battery_change = -discharge_energy / self.battery_efficiency

        else:  # Idle
            if p_net < 0:
                # Import to cover deficit
                grid_import = abs(p_net) * self.time_step_hours
            else:
                # Export surplus
                grid_export = p_net * self.time_step_hours

        # Update battery SoC
        self.soc += battery_change / self.battery_capacity
        self.soc = np.clip(self.soc, 0.0, 1.0)

        # Calculate costs and emissions
        import_cost = grid_import * c_grid
        export_revenue = grid_export * c_grid * 0.5  # Feed-in tariff at 50% of retail
        net_cost = import_cost - export_revenue

        emissions = grid_import * i_co2 / 1000  # Convert to kgCO2

        # Battery wear penalty (penalize action changes)
        wear_penalty = abs(action - self.last_action) if self.last_action != 2 else 0

        # Calculate reward (negative cost)
        reward = -(
            self.alpha * net_cost +
            self.beta * emissions +
            self.gamma_wear * wear_penalty
        )

        # Track episode data
        self.episode_costs.append(net_cost)
        self.episode_emissions.append(emissions)
        self.episode_soc_history.append(self.soc)
        self.episode_actions.append(action)
        self.last_action = action

        # Advance time
        self.current_step += 1
        self.current_hour = (self.current_step * self.time_step_minutes / 60) % 24

        # Check termination
        terminated = self.current_step >= self.max_steps
        truncated = False

        # Get new observation
        observation = self._get_observation()

        info = {
            "soc": self.soc,
            "hour": self.current_hour,
            "cost": net_cost,
            "emissions": emissions,
            "grid_import": grid_import,
            "grid_export": grid_export,
            "total_cost": sum(self.episode_costs),
            "total_emissions": sum(self.episode_emissions)
        }

        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment (for debugging)."""
        if self.render_mode == "human":
            print(f"Hour: {self.current_hour:.1f}, SoC: {self.soc:.2%}, "
                  f"Total Cost: €{sum(self.episode_costs):.3f}, "
                  f"Total Emissions: {sum(self.episode_emissions):.2f} kgCO2")

    def get_episode_summary(self) -> Dict[str, float]:
        """Return summary statistics for the episode."""
        return {
            "total_cost": sum(self.episode_costs),
            "total_emissions": sum(self.episode_emissions),
            "avg_soc": np.mean(self.episode_soc_history),
            "action_changes": sum(1 for i in range(1, len(self.episode_actions))
                                  if self.episode_actions[i] != self.episode_actions[i-1]),
            "charge_actions": self.episode_actions.count(0),
            "discharge_actions": self.episode_actions.count(1),
            "idle_actions": self.episode_actions.count(2)
        }


# For compatibility with RL-Glue style interface
class SmartGridEnvRLGlue:
    """Wrapper to provide RL-Glue compatible interface."""

    def __init__(self, env_info: Dict = None):
        env_info = env_info or {}
        self.env = SmartGridEnv(**env_info)
        self.reward_obs_term = None

    def env_init(self, env_info: Dict = None):
        """Initialize environment."""
        if env_info:
            self.env = SmartGridEnv(**env_info)

    def env_start(self):
        """Start a new episode."""
        obs, _ = self.env.reset()
        return obs

    def env_step(self, action):
        """Take a step in the environment."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.reward_obs_term = (reward, obs, terminated or truncated)
        return self.reward_obs_term

    def env_cleanup(self):
        """Cleanup."""
        pass

    def env_message(self, message):
        """Handle messages."""
        if message == "get_summary":
            return self.env.get_episode_summary()
        return None


if __name__ == "__main__":
    # Quick test
    env = SmartGridEnv(render_mode="human")
    obs, info = env.reset(seed=42)
    print(f"Initial observation: {obs}")

    total_reward = 0
    for _ in range(48):  # 24 hours with 30-min steps
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        env.render()
        if terminated:
            break

    print(f"\nEpisode Summary:")
    print(env.get_episode_summary())
    print(f"Total Reward: {total_reward:.3f}")
