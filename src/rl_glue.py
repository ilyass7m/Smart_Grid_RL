"""
RL-GLUE Framework
=================
Standard interface for connecting agents and environments.
Based on the RL-GLUE specification used in the course.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Dict, Any, Optional


class BaseAgent(ABC):
    """
    Abstract base class for all RL agents.
    All agents must implement these methods for RL-GLUE compatibility.
    """

    @abstractmethod
    def agent_init(self, agent_info: Dict[str, Any] = None) -> None:
        """
        Initialize the agent with given parameters.

        Args:
            agent_info: Dictionary containing agent parameters
        """
        pass

    @abstractmethod
    def agent_start(self, state: np.ndarray) -> int:
        """
        First action of an episode.

        Args:
            state: Initial state observation

        Returns:
            First action to take
        """
        pass

    @abstractmethod
    def agent_step(self, reward: float, state: np.ndarray) -> int:
        """
        Take a step: receive reward and new state, return next action.

        Args:
            reward: Reward from previous action
            state: New state observation

        Returns:
            Next action to take
        """
        pass

    @abstractmethod
    def agent_end(self, reward: float) -> None:
        """
        Receive final reward when episode ends.

        Args:
            reward: Final reward
        """
        pass

    def agent_cleanup(self) -> None:
        """Cleanup after episode ends."""
        pass

    def agent_message(self, message: str) -> Any:
        """
        Receive a message from the experiment.

        Args:
            message: Message string

        Returns:
            Response to the message
        """
        return None


class BaseEnvironment(ABC):
    """
    Abstract base class for all environments.
    All environments must implement these methods for RL-GLUE compatibility.
    """

    @abstractmethod
    def env_init(self, env_info: Dict[str, Any] = None) -> None:
        """
        Initialize the environment.

        Args:
            env_info: Dictionary containing environment parameters
        """
        pass

    @abstractmethod
    def env_start(self) -> np.ndarray:
        """
        Start a new episode.

        Returns:
            Initial state observation
        """
        pass

    @abstractmethod
    def env_step(self, action: int) -> Tuple[float, np.ndarray, bool]:
        """
        Take an action in the environment.

        Args:
            action: Action to take

        Returns:
            Tuple of (reward, next_state, is_terminal)
        """
        pass

    def env_cleanup(self) -> None:
        """Cleanup after episode ends."""
        pass

    def env_message(self, message: str) -> Any:
        """
        Receive a message from the experiment.

        Args:
            message: Message string

        Returns:
            Response to the message
        """
        return None


class RLGlue:
    """
    RL-GLUE experiment manager.
    Connects agents and environments and runs episodes.
    """

    def __init__(self, env_class, agent_class):
        """
        Initialize RL-GLUE with environment and agent classes.

        Args:
            env_class: Environment class (not instance)
            agent_class: Agent class (not instance)
        """
        self.environment = env_class()
        self.agent = agent_class()

        # Episode tracking
        self.total_reward = 0.0
        self.num_steps = 0
        self.num_episodes = 0

        # Current state
        self.last_action = None
        self.last_state = None

    def rl_init(self, agent_info: Dict = None, env_info: Dict = None) -> None:
        """
        Initialize both agent and environment.

        Args:
            agent_info: Agent initialization parameters
            env_info: Environment initialization parameters
        """
        self.environment.env_init(env_info or {})
        self.agent.agent_init(agent_info or {})

        self.total_reward = 0.0
        self.num_steps = 0
        self.num_episodes = 0

    def rl_start(self) -> Tuple[np.ndarray, int]:
        """
        Start a new episode.

        Returns:
            Tuple of (initial_state, first_action)
        """
        self.total_reward = 0.0
        self.num_steps = 1

        self.last_state = self.environment.env_start()
        self.last_action = self.agent.agent_start(self.last_state)

        return self.last_state, self.last_action

    def rl_step(self) -> Tuple[float, np.ndarray, int, bool]:
        """
        Take one step in the episode.

        Returns:
            Tuple of (reward, state, action, is_terminal)
        """
        reward, state, terminal = self.environment.env_step(self.last_action)

        self.total_reward += reward
        self.num_steps += 1

        if terminal:
            self.agent.agent_end(reward)
            self.num_episodes += 1
            return reward, state, None, terminal
        else:
            action = self.agent.agent_step(reward, state)
            self.last_state = state
            self.last_action = action
            return reward, state, action, terminal

    def rl_episode(self, max_steps: int = 0) -> bool:
        """
        Run a complete episode.

        Args:
            max_steps: Maximum steps (0 = unlimited)

        Returns:
            True if episode completed normally, False if truncated
        """
        self.rl_start()

        is_terminal = False
        while not is_terminal:
            _, _, _, is_terminal = self.rl_step()

            if max_steps > 0 and self.num_steps >= max_steps:
                # Truncate episode
                self.agent.agent_end(0.0)
                return False

        return True

    def rl_cleanup(self) -> None:
        """Cleanup after experiment."""
        self.environment.env_cleanup()
        self.agent.agent_cleanup()

    def rl_agent_message(self, message: str) -> Any:
        """Send message to agent."""
        return self.agent.agent_message(message)

    def rl_env_message(self, message: str) -> Any:
        """Send message to environment."""
        return self.environment.env_message(message)
