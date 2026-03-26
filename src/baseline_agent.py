"""
Baseline Agents for Smart Grid Energy Management
=================================================
Rule-based and simple strategies to compare against DQN.

Agents:
1. RandomAgent: Random action selection
2. GreedyAgent: Rule-based greedy strategy (charge cheap, discharge expensive)
3. ThresholdAgent: SoC threshold-based strategy
"""

import numpy as np
from typing import Dict, Optional


class RandomAgent:
    """Agent that selects actions randomly."""

    def __init__(self, num_actions: int = 3, seed: int = None):
        self.num_actions = num_actions
        self.rng = np.random.default_rng(seed)

    def agent_init(self, agent_info: Dict = None):
        if agent_info:
            self.num_actions = agent_info.get('num_actions', 3)
            seed = agent_info.get('seed', None)
            if seed:
                self.rng = np.random.default_rng(seed)

    def _select_action(self, state: np.ndarray) -> int:
        """Select random action (for consistency with other agents)."""
        return self.rng.integers(0, self.num_actions)

    def agent_start(self, state: np.ndarray) -> int:
        return self._select_action(state)

    def agent_step(self, reward: float, state: np.ndarray) -> int:
        return self._select_action(state)

    def agent_end(self, reward: float):
        pass

    def agent_cleanup(self):
        pass

    def agent_message(self, message: str):
        return None


class GreedyAgent:
    """
    Greedy Rule-Based Agent.

    Strategy:
    - Charge when electricity is cheap (off-peak) or when there's solar surplus
    - Discharge when electricity is expensive (peak hours)
    - Idle otherwise
    """

    def __init__(
        self,
        low_price_threshold: float = 0.15,
        high_price_threshold: float = 0.25,
        soc_min: float = 0.2,
        soc_max: float = 0.9
    ):
        self.low_price_threshold = low_price_threshold
        self.high_price_threshold = high_price_threshold
        self.soc_min = soc_min
        self.soc_max = soc_max

    def agent_init(self, agent_info: Dict = None):
        if agent_info:
            self.low_price_threshold = agent_info.get('low_price_threshold', 0.15)
            self.high_price_threshold = agent_info.get('high_price_threshold', 0.25)
            self.soc_min = agent_info.get('soc_min', 0.2)
            self.soc_max = agent_info.get('soc_max', 0.9)

    def _select_action(self, state: np.ndarray) -> int:
        """
        Select action based on current state.

        State: [hour, soc, p_net, c_grid, i_co2]
        Actions: 0=Charge, 1=Discharge, 2=Idle
        """
        hour, soc, p_net, c_grid, i_co2 = state

        # If battery is full, don't charge
        if soc >= self.soc_max:
            if c_grid > self.high_price_threshold:
                return 1  # Discharge during expensive periods
            return 2  # Idle

        # If battery is empty, don't discharge
        if soc <= self.soc_min:
            if c_grid < self.low_price_threshold or p_net > 0:
                return 0  # Charge during cheap periods or solar surplus
            return 2  # Idle

        # Normal operation
        if p_net > 0.5:
            # Significant solar surplus - charge
            return 0

        if c_grid > self.high_price_threshold:
            # Expensive - discharge
            return 1

        if c_grid < self.low_price_threshold:
            # Cheap - charge
            return 0

        # Default: Idle
        return 2

    def agent_start(self, state: np.ndarray) -> int:
        return self._select_action(state)

    def agent_step(self, reward: float, state: np.ndarray) -> int:
        return self._select_action(state)

    def agent_end(self, reward: float):
        pass

    def agent_cleanup(self):
        pass

    def agent_message(self, message: str):
        return None


class EcoGreedyAgent:
    """
    Eco-Aware Greedy Agent.

    Strategy incorporates both price AND carbon intensity:
    - Charge when carbon intensity is low (clean grid) and price is reasonable
    - Discharge when carbon intensity is high (dirty grid)
    - Prioritizes ecological impact over pure cost savings
    """

    def __init__(
        self,
        low_co2_threshold: float = 300,
        high_co2_threshold: float = 450,
        low_price_threshold: float = 0.18,
        high_price_threshold: float = 0.28,
        soc_min: float = 0.2,
        soc_max: float = 0.9
    ):
        self.low_co2_threshold = low_co2_threshold
        self.high_co2_threshold = high_co2_threshold
        self.low_price_threshold = low_price_threshold
        self.high_price_threshold = high_price_threshold
        self.soc_min = soc_min
        self.soc_max = soc_max

    def agent_init(self, agent_info: Dict = None):
        if agent_info:
            self.low_co2_threshold = agent_info.get('low_co2_threshold', 300)
            self.high_co2_threshold = agent_info.get('high_co2_threshold', 450)

    def _select_action(self, state: np.ndarray) -> int:
        """
        Select action based on current state with eco-awareness.

        State: [hour, soc, p_net, c_grid, i_co2]
        """
        hour, soc, p_net, c_grid, i_co2 = state

        # Enforce SoC limits
        if soc >= self.soc_max:
            if i_co2 > self.high_co2_threshold or c_grid > self.high_price_threshold:
                return 1
            return 2

        if soc <= self.soc_min:
            if i_co2 < self.low_co2_threshold or c_grid < self.low_price_threshold:
                return 0
            return 2

        # Solar surplus - always charge
        if p_net > 0.5:
            return 0

        # Clean grid AND cheap - charge
        if i_co2 < self.low_co2_threshold and c_grid < self.low_price_threshold:
            return 0

        # Dirty grid OR expensive - discharge
        if i_co2 > self.high_co2_threshold or c_grid > self.high_price_threshold:
            return 1

        # Clean grid - prefer charging
        if i_co2 < self.low_co2_threshold:
            return 0

        return 2

    def agent_start(self, state: np.ndarray) -> int:
        return self._select_action(state)

    def agent_step(self, reward: float, state: np.ndarray) -> int:
        return self._select_action(state)

    def agent_end(self, reward: float):
        pass

    def agent_cleanup(self):
        pass

    def agent_message(self, message: str):
        return None


class ThresholdAgent:
    """
    Time-based Threshold Agent.

    Simple time-of-use strategy:
    - Charge during night (off-peak): 00:00 - 06:00
    - Discharge during evening peak: 18:00 - 21:00
    - Idle otherwise
    """

    def __init__(
        self,
        charge_hours: tuple = (0, 6),
        discharge_hours: tuple = (18, 21),
        soc_min: float = 0.2,
        soc_max: float = 0.9
    ):
        self.charge_start, self.charge_end = charge_hours
        self.discharge_start, self.discharge_end = discharge_hours
        self.soc_min = soc_min
        self.soc_max = soc_max

    def agent_init(self, agent_info: Dict = None):
        pass

    def _select_action(self, state: np.ndarray) -> int:
        hour, soc, p_net, c_grid, i_co2 = state

        # SoC limits
        if soc >= self.soc_max:
            if self.discharge_start <= hour < self.discharge_end:
                return 1
            return 2

        if soc <= self.soc_min:
            return 0 if p_net > 0 else 2

        # Time-based rules
        if self.charge_start <= hour < self.charge_end:
            return 0  # Charge at night
        elif self.discharge_start <= hour < self.discharge_end:
            return 1  # Discharge during peak
        elif p_net > 0.5:
            return 0  # Charge from solar surplus

        return 2

    def agent_start(self, state: np.ndarray) -> int:
        return self._select_action(state)

    def agent_step(self, reward: float, state: np.ndarray) -> int:
        return self._select_action(state)

    def agent_end(self, reward: float):
        pass

    def agent_cleanup(self):
        pass

    def agent_message(self, message: str):
        return None


# Factory function to get agent by name
def get_baseline_agent(name: str, **kwargs):
    """Get baseline agent by name."""
    agents = {
        'random': RandomAgent,
        'greedy': GreedyAgent,
        'eco_greedy': EcoGreedyAgent,
        'threshold': ThresholdAgent
    }

    if name not in agents:
        raise ValueError(f"Unknown agent: {name}. Available: {list(agents.keys())}")

    return agents[name](**kwargs)


if __name__ == "__main__":
    # Test agents
    from environment import SmartGridEnv

    env = SmartGridEnv(seed=42)
    agents = {
        'Random': RandomAgent(seed=42),
        'Greedy': GreedyAgent(),
        'EcoGreedy': EcoGreedyAgent(),
        'Threshold': ThresholdAgent()
    }

    print("Testing baseline agents over 1 episode each:\n")

    for name, agent in agents.items():
        obs, _ = env.reset(seed=42)
        agent.agent_init({})
        action = agent.agent_start(obs)

        total_reward = 0
        done = False

        while not done:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            if not done:
                action = agent.agent_step(reward, obs)

        summary = env.get_episode_summary()
        print(f"{name:12s}: Cost={summary['total_cost']:.3f}€, "
              f"Emissions={summary['total_emissions']:.2f}kg, "
              f"Reward={total_reward:.2f}")
