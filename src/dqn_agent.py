"""
Deep Q-Network (DQN) Agent
==========================
Implementation of DQN for the Smart Grid Energy Management problem.

Features:
- Experience Replay Buffer
- Target Network with periodic updates
- Epsilon-greedy exploration with decay
- Neural network function approximation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
from typing import Tuple, Dict, Optional
import random


# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, capacity: int = 100000, seed: int = None):
        self.buffer = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Sample a batch of experiences."""
        experiences = self.rng.sample(self.buffer, batch_size)

        states = torch.FloatTensor(np.array([e.state for e in experiences]))
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences]))
        dones = torch.FloatTensor([e.done for e in experiences])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """Neural Network for Q-value approximation."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [128, 128]):
        super(QNetwork, self).__init__()

        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, state):
        return self.network(state)


class DQNAgent:
    """
    Deep Q-Network Agent for Smart Grid Energy Management.

    Implements:
    - Experience replay
    - Target network
    - Epsilon-greedy exploration with decay
    """

    def __init__(
        self,
        state_dim: int = 5,
        action_dim: int = 3,
        hidden_dims: list = [128, 128],
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        seed: int = None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Epsilon-greedy parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Set random seeds
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Q-Networks
        self.q_network = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, seed)

        # Training tracking
        self.update_count = 0
        self.loss_history = []

        # Current episode state
        self.last_state = None
        self.last_action = None

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state observation
            training: If True, use epsilon-greedy; if False, use greedy

        Returns:
            Selected action
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()

    def agent_init(self, agent_info: Dict = None):
        """Initialize agent (RL-Glue compatible)."""
        if agent_info:
            if 'epsilon' in agent_info:
                self.epsilon = agent_info['epsilon']
            if 'learning_rate' in agent_info:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = agent_info['learning_rate']

    def agent_start(self, state: np.ndarray) -> int:
        """Start episode, return first action."""
        self.last_state = state
        self.last_action = self.select_action(state)
        return self.last_action

    def agent_step(self, reward: float, state: np.ndarray) -> int:
        """
        Take a step: store experience, learn, and return next action.
        """
        # Store experience
        self.replay_buffer.push(
            self.last_state,
            self.last_action,
            reward,
            state,
            False
        )

        # Learn from batch
        loss = self.learn()
        if loss is not None:
            self.loss_history.append(loss)

        # Select next action
        action = self.select_action(state)

        # Update state/action
        self.last_state = state
        self.last_action = action

        return action

    def agent_end(self, reward: float):
        """End of episode: store terminal experience and learn."""
        # Store terminal experience
        self.replay_buffer.push(
            self.last_state,
            self.last_action,
            reward,
            np.zeros(self.state_dim),  # Terminal state
            True
        )

        # Learn from batch
        loss = self.learn()
        if loss is not None:
            self.loss_history.append(loss)

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def learn(self) -> Optional[float]:
        """
        Sample from replay buffer and perform one step of gradient descent.

        Returns:
            Loss value or None if buffer is too small
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Compute loss
        loss = nn.MSELoss()(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def save(self, filepath: str):
        """Save model weights."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)

    def load(self, filepath: str):
        """Load model weights."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions given a state."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.q_network(state_tensor).cpu().numpy()[0]


class DQNAgentRLGlue:
    """Wrapper for RL-Glue compatibility."""

    def __init__(self):
        self.agent = None
        self.num_actions = 3
        self.num_states = 5

    def agent_init(self, agent_info: Dict = None):
        """Initialize the agent."""
        agent_info = agent_info or {}
        self.num_actions = agent_info.get('num_actions', 3)
        self.num_states = agent_info.get('num_states', 5)

        self.agent = DQNAgent(
            state_dim=self.num_states,
            action_dim=self.num_actions,
            hidden_dims=agent_info.get('hidden_dims', [128, 128]),
            learning_rate=agent_info.get('learning_rate', 1e-3),
            gamma=agent_info.get('gamma', 0.99),
            epsilon_start=agent_info.get('epsilon', 1.0),
            epsilon_end=agent_info.get('epsilon_end', 0.01),
            epsilon_decay=agent_info.get('epsilon_decay', 0.995),
            batch_size=agent_info.get('batch_size', 64),
            seed=agent_info.get('seed', None)
        )

    def agent_start(self, state):
        """Start episode."""
        return self.agent.agent_start(state)

    def agent_step(self, reward, state):
        """Take step."""
        return self.agent.agent_step(reward, state)

    def agent_end(self, reward):
        """End episode."""
        self.agent.agent_end(reward)

    def agent_cleanup(self):
        """Cleanup."""
        pass

    def agent_message(self, message):
        """Handle messages."""
        if message == "get_epsilon":
            return self.agent.epsilon
        return None


if __name__ == "__main__":
    # Quick test
    agent = DQNAgent(state_dim=5, action_dim=3, seed=42)

    # Simulate some experiences
    for i in range(200):
        state = np.random.randn(5).astype(np.float32)
        action = agent.select_action(state)
        reward = np.random.randn()
        next_state = np.random.randn(5).astype(np.float32)
        done = i == 199

        agent.replay_buffer.push(state, action, reward, next_state, done)
        loss = agent.learn()
        if loss:
            print(f"Step {i}: Loss = {loss:.4f}")

    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Q-values for random state: {agent.get_q_values(np.random.randn(5))}")
