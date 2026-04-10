"""
RL Agents for Smart Grid Energy Management (RL-GLUE Compatible)
================================================================
Implements various agents for the Eco-Adaptive Home Storage project.

Agents:
1. DQNAgent: Deep Q-Network with experience replay
2. GreedyAgent: Price-based rule strategy
3. EcoGreedyAgent: Carbon-aware rule strategy
4. ThresholdAgent: Time-based scheduling
5. RandomAgent: Uniform random baseline
6. IdleAgent: No-storage baseline (always idle)
"""

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import random

# Try to import torch (optional for baseline agents)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. DQN agent will not work.")

from rl_glue import BaseAgent


# =============================================================================
# Neural Network Components for DQN
# =============================================================================

class QNetwork(nn.Module):
    """Q-value neural network."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [64, 64]):
        super().__init__()

        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, action_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ReplayBuffer:
    """Experience replay buffer for DQN."""

    def __init__(self, capacity: int = 20000, seed: int = None):
        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(self, state, action, reward, next_state, done):
        """Add transition to buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample a batch of transitions."""
        batch = self.rng.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


@dataclass
class DQNConfig:
    """Configuration for DQN agent."""
    lr: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64
    buffer_size: int = 20000
    min_buffer_size: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    target_update_freq: int = 100
    hidden_dims: Tuple[int, ...] = (64, 64)


# =============================================================================
# DQN Agent
# =============================================================================

class DQNAgent(BaseAgent):
    """
    Deep Q-Network Agent.

    Features:
    - Experience replay buffer
    - Target network with periodic updates
    - Epsilon-greedy exploration with decay
    """

    def __init__(self):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DQNAgent but is not available.")

        self.config = None
        self.device = None
        self.q_net = None
        self.target_net = None
        self.optimizer = None
        self.replay = None

        self.epsilon = 1.0
        self.training_steps = 0

        self.state_dim = 5
        self.action_dim = 3

        self.last_state = None
        self.last_action = None

        # Training mode flag
        self.training = True

    def agent_init(self, agent_info: Dict[str, Any] = None) -> None:
        """Initialize the DQN agent."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DQNAgent.")
        agent_info = agent_info or {}

        # Set random seeds
        seed = agent_info.get('seed', 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Device
        self.device = torch.device(
            agent_info.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )

        # Configuration
        self.config = DQNConfig(
            lr=agent_info.get('lr', 1e-3),
            gamma=agent_info.get('gamma', 0.99),
            batch_size=agent_info.get('batch_size', 64),
            buffer_size=agent_info.get('buffer_size', 20000),
            min_buffer_size=agent_info.get('min_buffer_size', 500),
            epsilon_start=agent_info.get('epsilon_start', 1.0),
            epsilon_end=agent_info.get('epsilon_end', 0.05),
            epsilon_decay=agent_info.get('epsilon_decay', 0.995),
            target_update_freq=agent_info.get('target_update_freq', 100),
            hidden_dims=tuple(agent_info.get('hidden_dims', [64, 64]))
        )

        self.state_dim = agent_info.get('state_dim', 5)
        self.action_dim = agent_info.get('action_dim', 3)
        self.training = agent_info.get('training', True)

        # Networks
        self.q_net = QNetwork(
            self.state_dim, self.action_dim, list(self.config.hidden_dims)
        ).to(self.device)

        self.target_net = QNetwork(
            self.state_dim, self.action_dim, list(self.config.hidden_dims)
        ).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.config.lr)

        # Replay buffer
        self.replay = ReplayBuffer(self.config.buffer_size, seed)

        # Epsilon
        self.epsilon = self.config.epsilon_start
        self.training_steps = 0

    def _select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """Select action using epsilon-greedy policy."""
        if not greedy and self.training and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_net(state_t)
            return int(torch.argmax(q_values, dim=1).item())

    def _update(self) -> Optional[float]:
        """Perform one step of gradient descent."""
        if len(self.replay) < max(self.config.batch_size, self.config.min_buffer_size):
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(self.config.batch_size)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Current Q values
        q_values = self.q_net(states_t).gather(1, actions_t)

        # Target Q values
        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]
            targets = rewards_t + self.config.gamma * (1.0 - dones_t) * max_next_q

        # Loss and optimization
        loss = nn.functional.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        # Update target network
        self.training_steps += 1
        if self.training_steps % self.config.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())

    def agent_start(self, state: np.ndarray) -> int:
        """First action of episode."""
        self.last_state = state
        self.last_action = self._select_action(state)
        return self.last_action

    def agent_step(self, reward: float, state: np.ndarray) -> int:
        """Take a step: store, learn, select next action."""
        if self.training:
            self.replay.push(
                self.last_state,
                self.last_action,
                reward,
                state,
                False
            )
            self._update()

        action = self._select_action(state)
        self.last_state = state
        self.last_action = action
        return action

    def agent_end(self, reward: float) -> None:
        """End of episode."""
        if self.training:
            self.replay.push(
                self.last_state,
                self.last_action,
                reward,
                np.zeros(self.state_dim, dtype=np.float32),
                True
            )
            self._update()
            # Decay epsilon
            self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

    def agent_cleanup(self) -> None:
        """Cleanup."""
        pass

    def agent_message(self, message: str) -> Any:
        """Handle messages."""
        if message == "get_epsilon":
            return self.epsilon
        elif message == "set_eval_mode":
            self.training = False
            return True
        elif message == "set_train_mode":
            self.training = True
            return True
        elif message == "get_q_values":
            if self.last_state is not None:
                with torch.no_grad():
                    state_t = torch.tensor(
                        self.last_state, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                    return self.q_net(state_t).cpu().numpy()[0]
            return None
        return None

    def save(self, filepath: str) -> None:
        """Save model weights."""
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps
        }, filepath)

    def load(self, filepath: str) -> None:
        """Load model weights."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint['training_steps']


# =============================================================================
# Baseline Agents
# =============================================================================

class GreedyAgent(BaseAgent):
    """
    Greedy Price-Based Agent.

    Strategy: Charge when price is low, discharge when price is high.
    Uses quantile-based thresholds.
    """

    def __init__(self):
        self.low_price_threshold = 0.15
        self.high_price_threshold = 0.28
        self.soc_min = 0.15
        self.soc_max = 0.90

    def agent_init(self, agent_info: Dict[str, Any] = None) -> None:
        agent_info = agent_info or {}
        self.low_price_threshold = agent_info.get('low_price_threshold', 0.15)
        self.high_price_threshold = agent_info.get('high_price_threshold', 0.28)
        self.soc_min = agent_info.get('soc_min', 0.15)
        self.soc_max = agent_info.get('soc_max', 0.90)

    def _select_action(self, state: np.ndarray) -> int:
        """Select action based on price."""
        hour, soc, p_net, price, carbon = state

        # SoC constraints
        if soc >= self.soc_max:
            return 1 if price >= self.high_price_threshold else 2
        if soc <= self.soc_min:
            return 0 if price <= self.low_price_threshold else 2

        # Price-based decisions
        if price <= self.low_price_threshold:
            return 0  # Charge when cheap
        elif price >= self.high_price_threshold:
            return 1  # Discharge when expensive
        else:
            return 2  # Idle otherwise

    def agent_start(self, state: np.ndarray) -> int:
        return self._select_action(state)

    def agent_step(self, reward: float, state: np.ndarray) -> int:
        return self._select_action(state)

    def agent_end(self, reward: float) -> None:
        pass

    def agent_message(self, message: str) -> Any:
        return None


class EcoGreedyAgent(BaseAgent):
    """
    Eco-Aware Greedy Agent.

    Strategy: Prioritizes carbon intensity over price.
    Charges during clean grid periods, discharges during dirty periods.
    """

    def __init__(self):
        self.low_carbon_threshold = 280
        self.high_carbon_threshold = 420
        self.low_price_threshold = 0.18
        self.high_price_threshold = 0.30
        self.soc_min = 0.15
        self.soc_max = 0.90

    def agent_init(self, agent_info: Dict[str, Any] = None) -> None:
        agent_info = agent_info or {}
        self.low_carbon_threshold = agent_info.get('low_carbon_threshold', 280)
        self.high_carbon_threshold = agent_info.get('high_carbon_threshold', 420)
        self.low_price_threshold = agent_info.get('low_price_threshold', 0.18)
        self.high_price_threshold = agent_info.get('high_price_threshold', 0.30)
        self.soc_min = agent_info.get('soc_min', 0.15)
        self.soc_max = agent_info.get('soc_max', 0.90)

    def _select_action(self, state: np.ndarray) -> int:
        """Select action based on carbon intensity and price."""
        hour, soc, p_net, price, carbon = state

        # SoC constraints
        if soc >= self.soc_max:
            if carbon > self.high_carbon_threshold or price > self.high_price_threshold:
                return 1
            return 2

        if soc <= self.soc_min:
            if carbon < self.low_carbon_threshold or price < self.low_price_threshold:
                return 0
            return 2

        # Solar surplus - always charge
        if p_net < -0.5:
            return 0

        # Carbon-first decisions
        if carbon < self.low_carbon_threshold:
            return 0  # Clean grid - charge
        elif carbon > self.high_carbon_threshold:
            return 1  # Dirty grid - discharge

        # Fall back to price
        if price < self.low_price_threshold:
            return 0
        elif price > self.high_price_threshold:
            return 1

        return 2

    def agent_start(self, state: np.ndarray) -> int:
        return self._select_action(state)

    def agent_step(self, reward: float, state: np.ndarray) -> int:
        return self._select_action(state)

    def agent_end(self, reward: float) -> None:
        pass

    def agent_message(self, message: str) -> Any:
        return None


class ThresholdAgent(BaseAgent):
    """
    Time-Based Threshold Agent.

    Strategy: Fixed schedule based on time of day.
    - Charge during off-peak hours (night)
    - Discharge during peak hours (evening)
    """

    def __init__(self):
        self.charge_start = 0
        self.charge_end = 6
        self.discharge_start = 17
        self.discharge_end = 21
        self.soc_min = 0.15
        self.soc_max = 0.90

    def agent_init(self, agent_info: Dict[str, Any] = None) -> None:
        agent_info = agent_info or {}
        self.charge_start = agent_info.get('charge_start', 0)
        self.charge_end = agent_info.get('charge_end', 6)
        self.discharge_start = agent_info.get('discharge_start', 17)
        self.discharge_end = agent_info.get('discharge_end', 21)
        self.soc_min = agent_info.get('soc_min', 0.15)
        self.soc_max = agent_info.get('soc_max', 0.90)

    def _select_action(self, state: np.ndarray) -> int:
        """Select action based on time of day."""
        hour, soc, p_net, price, carbon = state

        # SoC constraints
        if soc >= self.soc_max:
            if self.discharge_start <= hour < self.discharge_end:
                return 1
            return 2

        if soc <= self.soc_min:
            if self.charge_start <= hour < self.charge_end:
                return 0
            elif p_net < 0:  # Solar surplus
                return 0
            return 2

        # Time-based rules
        if self.charge_start <= hour < self.charge_end:
            return 0
        elif self.discharge_start <= hour < self.discharge_end:
            return 1
        elif p_net < -0.5:  # Solar surplus
            return 0

        return 2

    def agent_start(self, state: np.ndarray) -> int:
        return self._select_action(state)

    def agent_step(self, reward: float, state: np.ndarray) -> int:
        return self._select_action(state)

    def agent_end(self, reward: float) -> None:
        pass

    def agent_message(self, message: str) -> Any:
        return None


class RandomAgent(BaseAgent):
    """
    Random Agent.

    Baseline: Selects actions uniformly at random.
    """

    def __init__(self):
        self.num_actions = 3
        self.rng = np.random.default_rng(42)

    def agent_init(self, agent_info: Dict[str, Any] = None) -> None:
        agent_info = agent_info or {}
        self.num_actions = agent_info.get('num_actions', 3)
        seed = agent_info.get('seed', 42)
        self.rng = np.random.default_rng(seed)

    def _select_action(self, state: np.ndarray) -> int:
        return int(self.rng.integers(0, self.num_actions))

    def agent_start(self, state: np.ndarray) -> int:
        return self._select_action(state)

    def agent_step(self, reward: float, state: np.ndarray) -> int:
        return self._select_action(state)

    def agent_end(self, reward: float) -> None:
        pass

    def agent_message(self, message: str) -> Any:
        return None


class IdleAgent(BaseAgent):
    """
    Idle (No-Storage) Agent.

    Baseline: Always takes the Idle action.
    Represents the scenario without battery storage.
    """

    def __init__(self):
        pass

    def agent_init(self, agent_info: Dict[str, Any] = None) -> None:
        pass

    def _select_action(self, state: np.ndarray) -> int:
        return 2  # Always Idle

    def agent_start(self, state: np.ndarray) -> int:
        return 2

    def agent_step(self, reward: float, state: np.ndarray) -> int:
        return 2

    def agent_end(self, reward: float) -> None:
        pass

    def agent_message(self, message: str) -> Any:
        return None


# =============================================================================
# Agent Factory
# =============================================================================

# Import SARSA agents (always available, no torch dependency)
from sarsa_agent import SarsaAgent, ExpectedSarsaAgent

AGENT_REGISTRY = {
    'dqn': DQNAgent,
    'sarsa': SarsaAgent,
    'expected_sarsa': ExpectedSarsaAgent,
    'greedy': GreedyAgent,
    'eco_greedy': EcoGreedyAgent,
    'threshold': ThresholdAgent,
    'random': RandomAgent,
    'idle': IdleAgent
}


def get_agent(name: str) -> BaseAgent:
    """Get agent class by name."""
    if name.lower() not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent: {name}. Available: {list(AGENT_REGISTRY.keys())}")
    return AGENT_REGISTRY[name.lower()]


def create_agent(name: str, agent_info: Dict = None) -> BaseAgent:
    """Create and initialize an agent."""
    agent_class = get_agent(name)
    agent = agent_class()
    agent.agent_init(agent_info or {})
    return agent
