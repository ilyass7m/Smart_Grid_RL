"""
Smart Grid Environment (RL-GLUE Compatible)
============================================
A home energy management environment for the Eco-Adaptive Home Storage project.

State Space (5-dimensional continuous):
    - hour: Current hour [0, 23.5]
    - soc: Battery State of Charge [0, 1]
    - p_net: Net load (Consumption - Solar) in kW
    - c_grid: Grid electricity price in €/kWh
    - i_co2: Carbon intensity in gCO2/kWh

Action Space (Discrete):
    - 0: Charge (battery from grid/PV)
    - 1: Discharge (battery to home/grid)
    - 2: Idle (no battery activity)
"""

import numpy as np
from typing import Tuple, Dict, Any, List
from rl_glue import BaseEnvironment


class SmartGridEnvironment(BaseEnvironment):
    """
    Smart Grid Home Energy Management Environment.
    RL-GLUE compatible implementation.
    """

    def __init__(self):
        # Parameters will be set in env_init
        self.alpha = 1.0
        self.beta = 0.6
        self.gamma_wear = 0.05

        self.battery_capacity_kwh = 10.0
        self.max_power_kw = 3.0
        self.eta_charge = 0.95
        self.eta_discharge = 0.95

        self.steps_per_episode = 48
        self.dt_hours = 0.5

        self.seed = 42
        self.rng = np.random.default_rng(self.seed)

        # State variables
        self.current_step = 0
        self.soc = 0.5
        self.prev_action = 2

        # Daily profiles
        self.hour_profile = None
        self.p_net_profile = None
        self.price_profile = None
        self.carbon_profile = None

        # Episode tracking
        self.episode_info = {
            'costs': [],
            'emissions': [],
            'baseline_costs': [],
            'baseline_emissions': [],
            'soc_history': [],
            'actions': [],
            'prices': [],
            'carbons': []
        }

    def env_init(self, env_info: Dict[str, Any] = None) -> None:
        """Initialize environment with parameters."""
        env_info = env_info or {}

        # Reward weights
        self.alpha = env_info.get('alpha', 1.0)
        self.beta = env_info.get('beta', 0.6)
        self.gamma_wear = env_info.get('gamma_wear', 0.05)

        # Battery parameters
        self.battery_capacity_kwh = env_info.get('battery_capacity_kwh', 10.0)
        self.max_power_kw = env_info.get('max_power_kw', 3.0)
        self.eta_charge = env_info.get('eta_charge', 0.95)
        self.eta_discharge = env_info.get('eta_discharge', 0.95)

        # Time parameters
        self.steps_per_episode = env_info.get('steps_per_episode', 48)
        self.dt_hours = env_info.get('dt_hours', 0.5)

        # Random seed
        self.seed = env_info.get('seed', 42)
        self.rng = np.random.default_rng(self.seed)

    def _generate_synthetic_profiles(self) -> Tuple[np.ndarray, ...]:
        """
        Generate realistic daily profiles for load, price, and carbon.

        Returns:
            Tuple of (hours, p_net, prices, carbon) arrays
        """
        t = np.arange(self.steps_per_episode)
        hours = t * self.dt_hours

        # Add daily variation
        day_shift = self.rng.uniform(-1.0, 1.0)

        # Solar production (bell curve, peak around noon-1pm)
        solar_shape = np.sin((hours - 6.0 + day_shift) * np.pi / 12.0)
        solar_kw = np.clip(solar_shape, 0.0, None)
        solar_kw *= self.rng.uniform(2.5, 4.0)  # Random peak capacity

        # Household consumption
        base_load = 1.4 + 0.4 * np.sin((hours - 3.0) * 2 * np.pi / 24.0)
        morning_peak = 1.0 * np.exp(-0.5 * ((hours - 7.5) / 1.8) ** 2)
        evening_peak = 1.6 * np.exp(-0.5 * ((hours - 19.0) / 2.2) ** 2)
        noise_load = self.rng.normal(0.0, 0.15, size=self.steps_per_episode)
        load_kw = np.clip(base_load + morning_peak + evening_peak + noise_load, 0.5, None)

        # Net load (positive = deficit, negative = surplus)
        p_net = load_kw - solar_kw

        # Electricity price (time-of-use with evening peak)
        price_base = 0.20 + 0.05 * np.sin((hours - 13.0) * 2 * np.pi / 24.0)
        price_evening = 0.12 * np.exp(-0.5 * ((hours - 19.0) / 2.5) ** 2)
        price_noise = self.rng.normal(0.0, 0.008, size=self.steps_per_episode)
        c_grid = np.clip(price_base + price_evening + price_noise, 0.08, 0.45)

        # Carbon intensity (higher during peak, lower midday due to solar)
        carbon_base = 360 + 80 * np.sin((hours - 15.0) * 2 * np.pi / 24.0)
        carbon_evening = 160 * np.exp(-0.5 * ((hours - 20.0) / 2.3) ** 2)
        carbon_noise = self.rng.normal(0.0, 15.0, size=self.steps_per_episode)
        i_co2 = np.clip(carbon_base + carbon_evening + carbon_noise, 150.0, 700.0)

        return (
            hours.astype(np.float32),
            p_net.astype(np.float32),
            c_grid.astype(np.float32),
            i_co2.astype(np.float32)
        )

    def _get_observation(self) -> np.ndarray:
        """Get current state observation."""
        return np.array([
            self.hour_profile[self.current_step],
            self.soc,
            self.p_net_profile[self.current_step],
            self.price_profile[self.current_step],
            self.carbon_profile[self.current_step]
        ], dtype=np.float32)

    def env_start(self) -> np.ndarray:
        """Start a new episode."""
        self.current_step = 0
        self.soc = float(self.rng.uniform(0.3, 0.7))
        self.prev_action = 2

        # Generate new daily profiles
        (
            self.hour_profile,
            self.p_net_profile,
            self.price_profile,
            self.carbon_profile
        ) = self._generate_synthetic_profiles()

        # Reset episode tracking
        self.episode_info = {
            'costs': [],
            'emissions': [],
            'baseline_costs': [],
            'baseline_emissions': [],
            'soc_history': [self.soc],
            'actions': [],
            'prices': [],
            'carbons': []
        }

        return self._get_observation()

    def env_step(self, action: int) -> Tuple[float, np.ndarray, bool]:
        """
        Take an action in the environment.

        Args:
            action: 0=Charge, 1=Discharge, 2=Idle

        Returns:
            Tuple of (reward, next_state, is_terminal)
        """
        action = int(action)

        # Get current profile values
        p_net = float(self.p_net_profile[self.current_step])
        price = float(self.price_profile[self.current_step])
        carbon = float(self.carbon_profile[self.current_step])

        # Battery power flow
        batt_power_kw = 0.0

        if action == 0:  # Charge
            max_energy_store = (1.0 - self.soc) * self.battery_capacity_kwh
            max_charge_kw = max_energy_store / (self.dt_hours * self.eta_charge + 1e-8)
            batt_power_kw = min(self.max_power_kw, max_charge_kw)
            delta_soc = (batt_power_kw * self.dt_hours * self.eta_charge) / self.battery_capacity_kwh
            self.soc = min(1.0, self.soc + delta_soc)

        elif action == 1:  # Discharge
            available_energy = self.soc * self.battery_capacity_kwh
            max_discharge_kw = (available_energy * self.eta_discharge) / (self.dt_hours + 1e-8)
            batt_power_kw = -min(self.max_power_kw, max_discharge_kw)
            energy_from_battery = (-batt_power_kw * self.dt_hours) / self.eta_discharge
            delta_soc = energy_from_battery / self.battery_capacity_kwh
            self.soc = max(0.0, self.soc - delta_soc)

        # Grid interaction
        net_grid_kw = p_net + batt_power_kw
        import_grid_kw = max(net_grid_kw, 0.0)

        # Calculate costs and emissions
        cost_t = import_grid_kw * price * self.dt_hours
        emissions_t = import_grid_kw * carbon * self.dt_hours / 1000.0  # kg CO2

        # Baseline (no storage)
        baseline_import_kw = max(p_net, 0.0)
        baseline_cost = baseline_import_kw * price * self.dt_hours
        baseline_emissions = baseline_import_kw * carbon * self.dt_hours / 1000.0

        # Action switching penalty
        switch_penalty = abs(action - self.prev_action)

        # Multi-objective reward
        reward = -(
            self.alpha * cost_t +
            self.beta * emissions_t +
            self.gamma_wear * switch_penalty
        )

        # Track episode data
        self.episode_info['costs'].append(cost_t)
        self.episode_info['emissions'].append(emissions_t)
        self.episode_info['baseline_costs'].append(baseline_cost)
        self.episode_info['baseline_emissions'].append(baseline_emissions)
        self.episode_info['soc_history'].append(self.soc)
        self.episode_info['actions'].append(action)
        self.episode_info['prices'].append(price)
        self.episode_info['carbons'].append(carbon)

        self.prev_action = action
        self.current_step += 1

        # Check termination
        terminated = self.current_step >= self.steps_per_episode

        if not terminated:
            obs = self._get_observation()
        else:
            obs = np.array([23.5, self.soc, 0.0, 0.0, 0.0], dtype=np.float32)

        return reward, obs, terminated

    def env_cleanup(self) -> None:
        """Cleanup after episode."""
        pass

    def env_message(self, message: str) -> Any:
        """Handle messages from experiment."""
        if message == "get_episode_summary":
            return self.get_episode_summary()
        elif message == "get_profiles":
            return {
                'hours': self.hour_profile,
                'p_net': self.p_net_profile,
                'prices': self.price_profile,
                'carbons': self.carbon_profile
            }
        return None

    def get_episode_summary(self) -> Dict[str, float]:
        """Get summary statistics for the episode."""
        if not self.episode_info['costs']:
            return {}

        total_cost = sum(self.episode_info['costs'])
        total_emissions = sum(self.episode_info['emissions'])
        baseline_cost = sum(self.episode_info['baseline_costs'])
        baseline_emissions = sum(self.episode_info['baseline_emissions'])

        actions = self.episode_info['actions']
        action_changes = sum(1 for i in range(1, len(actions))
                            if actions[i] != actions[i-1])

        return {
            'total_cost': total_cost,
            'total_emissions': total_emissions,
            'baseline_cost': baseline_cost,
            'baseline_emissions': baseline_emissions,
            'euros_saved': baseline_cost - total_cost,
            'co2_avoided': baseline_emissions - total_emissions,
            'avg_soc': np.mean(self.episode_info['soc_history']),
            'action_changes': action_changes,
            'charge_count': actions.count(0),
            'discharge_count': actions.count(1),
            'idle_count': actions.count(2)
        }


# Factory function for creating environment
def create_environment(**kwargs) -> SmartGridEnvironment:
    """Create and initialize a SmartGridEnvironment."""
    env = SmartGridEnvironment()
    env.env_init(kwargs)
    return env
