"""
SARSA Agent with Tile Coding for Smart Grid
============================================
An on-policy TD control agent using tile coding for function approximation.


Key Features:
- Tile coding for state representation (handles continuous states)
- SARSA (on-policy) temporal difference learning
- Linear function approximation with efficient sparse updates
- Epsilon-greedy exploration with decay
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from rl_glue import BaseAgent


# =============================================================================
# Tile Coding Implementation
# =============================================================================

class IHT:
    """
    Index Hash Table for tile coding.
    Maps tile indices to a fixed-size table using hashing.
    """

    def __init__(self, size: int):
        self.size = size
        self.overfull_count = 0
        self.dictionary = {}

    def count(self) -> int:
        return len(self.dictionary)

    def full(self) -> bool:
        return len(self.dictionary) >= self.size

    def get_index(self, obj, read_only: bool = False) -> int:
        d = self.dictionary
        if obj in d:
            return d[obj]
        elif read_only:
            return None
        size = self.size
        count = len(d)
        if count >= size:
            if self.overfull_count == 0:
                print(f"IHT full, size={size}, starting to allow collisions")
            self.overfull_count += 1
            return hash(obj) % self.size
        else:
            d[obj] = count
            return count


def hash_coords(coordinates: List, m: int, read_only: bool = False) -> int:
    """Hash coordinates to an index."""
    if isinstance(m, IHT):
        return m.get_index(tuple(coordinates), read_only)
    if isinstance(m, int):
        return hash(tuple(coordinates)) % m
    return None


def tiles(
    iht_or_size,
    num_tilings: int,
    floats: List[float],
    ints: List[int] = None,
    read_only: bool = False
) -> List[int]:
    """
    Return list of tile indices for the given state.

    Args:
        iht_or_size: Index hash table or size for hashing
        num_tilings: Number of tilings
        floats: List of continuous state variables (scaled)
        ints: Optional list of integer features
        read_only: If True, only return existing tiles

    Returns:
        List of active tile indices
    """
    ints = ints or []
    qfloats = [int(np.floor(f * num_tilings)) for f in floats]

    result = []
    for tiling in range(num_tilings):
        tilingX2 = tiling * 2
        coords = [tiling]
        b = tiling
        for q in qfloats:
            coords.append((q + b) // num_tilings)
            b += tilingX2
        coords.extend(ints)
        result.append(hash_coords(coords, iht_or_size, read_only))

    return result


class SmartGridTileCoder:
    """
    Tile coder specifically designed for the Smart Grid environment.

    State dimensions and their ranges:
    - hour: [0, 24] - Hour of day
    - soc: [0, 1] - State of charge
    - p_net: [-5, 5] - Net load (clipped)
    - c_grid: [0, 0.5] - Grid price
    - i_co2: [100, 700] - Carbon intensity
    """

    def __init__(
        self,
        iht_size: int = 4096,
        num_tilings: int = 8,
        num_tiles: int = 8
    ):
        """
        Initialize tile coder.

        Args:
            iht_size: Size of index hash table
            num_tilings: Number of overlapping tilings
            num_tiles: Number of tiles per dimension per tiling
        """
        self.iht = IHT(iht_size)
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles

        # State variable ranges [min, max]
        self.ranges = {
            'hour': (0.0, 24.0),
            'soc': (0.0, 1.0),
            'p_net': (-5.0, 5.0),
            'c_grid': (0.0, 0.5),
            'i_co2': (100.0, 700.0)
        }

    def _scale(self, value: float, min_val: float, max_val: float) -> float:
        """Scale value to [0, num_tiles] range."""
        normalized = (value - min_val) / (max_val - min_val + 1e-8)
        normalized = np.clip(normalized, 0.0, 1.0)
        return normalized * self.num_tiles

    def get_tiles(self, state: np.ndarray) -> np.ndarray:
        """
        Get active tile indices for a state.

        Args:
            state: [hour, soc, p_net, c_grid, i_co2]

        Returns:
            Array of active tile indices
        """
        hour, soc, p_net, c_grid, i_co2 = state

        # Scale each dimension
        scaled = [
            self._scale(hour, *self.ranges['hour']),
            self._scale(soc, *self.ranges['soc']),
            self._scale(np.clip(p_net, -5, 5), *self.ranges['p_net']),
            self._scale(np.clip(c_grid, 0, 0.5), *self.ranges['c_grid']),
            self._scale(np.clip(i_co2, 100, 700), *self.ranges['i_co2'])
        ]

        return np.array(tiles(self.iht, self.num_tilings, scaled))

    def get_tiles_for_action(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Get tiles with action included as a feature.

        Args:
            state: State observation
            action: Action taken

        Returns:
            Array of active tile indices
        """
        hour, soc, p_net, c_grid, i_co2 = state

        scaled = [
            self._scale(hour, *self.ranges['hour']),
            self._scale(soc, *self.ranges['soc']),
            self._scale(np.clip(p_net, -5, 5), *self.ranges['p_net']),
            self._scale(np.clip(c_grid, 0, 0.5), *self.ranges['c_grid']),
            self._scale(np.clip(i_co2, 100, 700), *self.ranges['i_co2'])
        ]

        return np.array(tiles(self.iht, self.num_tilings, scaled, ints=[action]))


# =============================================================================
# SARSA Agent
# =============================================================================

class SarsaAgent(BaseAgent):
    """
    SARSA Agent with Tile Coding Function Approximation.

    This is an on-policy TD control agent that learns Q(s,a) while
    following an epsilon-greedy policy.

    Algorithm:
        SARSA (State-Action-Reward-State-Action):
        1. Observe state S, choose action A using ε-greedy
        2. Take action A, observe R, S'
        3. Choose A' using ε-greedy on S'
        4. Update: Q(S,A) += α * [R + γ*Q(S',A') - Q(S,A)]
        5. S = S', A = A'

    With tile coding, Q(s,a) = sum of weights at active tiles.
    """

    def __init__(self):
        # Tile coder
        self.tile_coder = None

        # Weights (one set per action)
        self.w = None
        self.num_actions = 3
        self.iht_size = 4096

        # Learning parameters
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay = 0.995

        # State tracking
        self.last_state = None
        self.last_action = None
        self.last_tiles = None

        # Training mode
        self.training = True

        # Statistics
        self.episode_count = 0

    def agent_init(self, agent_info: Dict[str, Any] = None) -> None:
        """Initialize the SARSA agent."""
        agent_info = agent_info or {}

        # Tile coding parameters
        self.iht_size = agent_info.get('iht_size', 4096)
        num_tilings = agent_info.get('num_tilings', 8)
        num_tiles = agent_info.get('num_tiles', 8)

        self.tile_coder = SmartGridTileCoder(
            iht_size=self.iht_size,
            num_tilings=num_tilings,
            num_tiles=num_tiles
        )

        # Learning parameters
        self.alpha = agent_info.get('alpha', 0.1) / num_tilings  # Divide by tilings
        self.gamma = agent_info.get('gamma', 0.99)
        self.epsilon = agent_info.get('epsilon', 1.0)
        self.epsilon_end = agent_info.get('epsilon_end', 0.05)
        self.epsilon_decay = agent_info.get('epsilon_decay', 0.995)

        self.num_actions = agent_info.get('num_actions', 3)
        self.training = agent_info.get('training', True)

        # Initialize weights to zero (or small values for optimism)
        initial_value = agent_info.get('initial_value', 0.0)
        self.w = np.ones((self.num_actions, self.iht_size)) * initial_value

        # Random seed
        seed = agent_info.get('seed', None)
        if seed is not None:
            np.random.seed(seed)

        self.episode_count = 0

    def _get_action_values(self, state: np.ndarray) -> np.ndarray:
        """
        Compute Q(s, a) for all actions.

        Args:
            state: Current state

        Returns:
            Array of Q-values for each action
        """
        tile_indices = self.tile_coder.get_tiles(state)
        q_values = np.array([
            np.sum(self.w[a][tile_indices]) for a in range(self.num_actions)
        ])
        return q_values

    def _select_action(self, state: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state

        Returns:
            Tuple of (action, tile_indices)
        """
        tile_indices = self.tile_coder.get_tiles(state)
        q_values = np.array([
            np.sum(self.w[a][tile_indices]) for a in range(self.num_actions)
        ])

        if self.training and np.random.random() < self.epsilon:
            action = np.random.randint(self.num_actions)
        else:
            # Break ties randomly
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            action = np.random.choice(best_actions)

        return action, tile_indices

    def agent_start(self, state: np.ndarray) -> int:
        """First action of episode."""
        action, tile_indices = self._select_action(state)

        self.last_state = state
        self.last_action = action
        self.last_tiles = tile_indices

        return action

    def agent_step(self, reward: float, state: np.ndarray) -> int:
        """
        Take a step: update Q-values and select next action.

        SARSA update:
            Q(S,A) += α * [R + γ*Q(S',A') - Q(S,A)]
        """
        # Select next action (for SARSA, we need A' before updating)
        action, tile_indices = self._select_action(state)

        if self.training:
            # Compute TD error
            # Q(S,A) = sum of weights at last_tiles for last_action
            q_current = np.sum(self.w[self.last_action][self.last_tiles])
            # Q(S',A') = sum of weights at current tiles for current action
            q_next = np.sum(self.w[action][tile_indices])

            # TD target
            td_target = reward + self.gamma * q_next
            td_error = td_target - q_current

            # Update weights at active tiles
            # For tile coding, gradient is 1 at active tiles, 0 elsewhere
            self.w[self.last_action][self.last_tiles] += self.alpha * td_error

        # Update state/action
        self.last_state = state
        self.last_action = action
        self.last_tiles = tile_indices

        return action

    def agent_end(self, reward: float) -> None:
        """
        End of episode update (terminal state).

        For terminal state, Q(S',A') = 0.
        """
        if self.training:
            # Q(S,A) = sum of weights at last_tiles
            q_current = np.sum(self.w[self.last_action][self.last_tiles])

            # TD error (no next state value since terminal)
            td_error = reward - q_current

            # Update weights
            self.w[self.last_action][self.last_tiles] += self.alpha * td_error

            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            self.episode_count += 1

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
        elif message == "get_weights":
            return self.w.copy()
        elif message == "get_tile_usage":
            return self.tile_coder.iht.count()
        return None

    def save(self, filepath: str) -> None:
        """Save model."""
        np.savez(filepath,
                 weights=self.w,
                 epsilon=self.epsilon,
                 episode_count=self.episode_count)

    def load(self, filepath: str) -> None:
        """Load model."""
        data = np.load(filepath)
        self.w = data['weights']
        self.epsilon = float(data['epsilon'])
        self.episode_count = int(data['episode_count'])


# =============================================================================
# Alternative: Expected SARSA Agent
# =============================================================================

class ExpectedSarsaAgent(SarsaAgent):
    """
    Expected SARSA Agent.

    Instead of using Q(S',A'), uses the expected value under the policy:
        E[Q(S',A')] = sum over a' of π(a'|S') * Q(S',a')

    This reduces variance compared to SARSA while remaining on-policy.
    """

    def agent_step(self, reward: float, state: np.ndarray) -> int:
        """
        Take a step using Expected SARSA update.

        Update:
            Q(S,A) += α * [R + γ * E[Q(S',A')] - Q(S,A)]
        where E[Q(S',A')] = sum over a' of π(a'|S') * Q(S',a')
        """
        # Get Q-values for next state
        tile_indices = self.tile_coder.get_tiles(state)
        q_values = np.array([
            np.sum(self.w[a][tile_indices]) for a in range(self.num_actions)
        ])

        # Select action for execution
        if self.training and np.random.random() < self.epsilon:
            action = np.random.randint(self.num_actions)
        else:
            max_q = np.max(q_values)
            best_actions = np.where(q_values == max_q)[0]
            action = np.random.choice(best_actions)

        if self.training:
            # Compute expected value under ε-greedy policy
            # π(a|s) = ε/|A| for non-greedy, (1-ε) + ε/|A| for greedy
            max_q = np.max(q_values)
            num_greedy = np.sum(q_values == max_q)

            expected_q = 0.0
            for a in range(self.num_actions):
                if q_values[a] == max_q:
                    # Greedy action
                    prob = (1 - self.epsilon) / num_greedy + self.epsilon / self.num_actions
                else:
                    # Non-greedy action
                    prob = self.epsilon / self.num_actions
                expected_q += prob * q_values[a]

            # TD error
            q_current = np.sum(self.w[self.last_action][self.last_tiles])
            td_target = reward + self.gamma * expected_q
            td_error = td_target - q_current

            # Update weights
            self.w[self.last_action][self.last_tiles] += self.alpha * td_error

        # Update state/action
        self.last_state = state
        self.last_action = action
        self.last_tiles = tile_indices

        return action


# =============================================================================
# Factory and Utilities
# =============================================================================

def create_sarsa_agent(agent_type: str = 'sarsa', **kwargs) -> BaseAgent:
    """
    Create a SARSA-family agent.

    Args:
        agent_type: 'sarsa' or 'expected_sarsa'
        **kwargs: Agent parameters

    Returns:
        Initialized agent
    """
    if agent_type.lower() == 'sarsa':
        agent = SarsaAgent()
    elif agent_type.lower() in ['expected_sarsa', 'expectedsarsa']:
        agent = ExpectedSarsaAgent()
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    agent.agent_init(kwargs)
    return agent


if __name__ == "__main__":
    # Quick test
    print("Testing SARSA Agent with Tile Coding")
    print("=" * 50)

    # Test tile coder
    tc = SmartGridTileCoder(iht_size=4096, num_tilings=8, num_tiles=8)
    test_state = np.array([12.0, 0.5, 1.5, 0.25, 400.0])
    tiles = tc.get_tiles(test_state)
    print(f"State: {test_state}")
    print(f"Tiles: {tiles}")
    print(f"Num tiles: {len(tiles)}")

    # Test SARSA agent
    agent = SarsaAgent()
    agent.agent_init({
        'iht_size': 4096,
        'num_tilings': 8,
        'num_tiles': 8,
        'alpha': 0.5,
        'epsilon': 0.1,
        'seed': 42
    })

    # Simulate a few steps
    state = np.array([0.0, 0.5, 1.0, 0.15, 350.0])
    action = agent.agent_start(state)
    print(f"\nFirst action: {action}")

    for i in range(5):
        next_state = state + np.random.randn(5) * 0.1
        reward = -np.random.rand()
        action = agent.agent_step(reward, next_state)
        state = next_state
        print(f"Step {i+1}: action={action}, reward={reward:.3f}")

    agent.agent_end(-0.5)
    print(f"\nFinal epsilon: {agent.epsilon:.4f}")
    print(f"Tile usage: {agent.tile_coder.iht.count()}")

    print("\nSARSA Agent test passed!")
