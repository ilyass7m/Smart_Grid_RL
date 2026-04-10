"""
Eco-Adaptive Home Storage: Multi-Objective RL for Smart Grid Energy Management
===============================================================================

A reinforcement learning project for intelligent battery arbitrage in smart homes.

Modules:
    rl_glue: RL-GLUE framework (BaseAgent, BaseEnvironment, RLGlue)
    smart_grid_env: SmartGridEnvironment - RL-GLUE compatible environment
    agents: All agents (DQN, SARSA, Greedy, EcoGreedy, Threshold, Random, Idle)
    sarsa_agent: SARSA agents with tile coding (no PyTorch required)
    experiment: Training, evaluation, and visualization utilities
"""

from .rl_glue import RLGlue, BaseAgent, BaseEnvironment
from .smart_grid_env import SmartGridEnvironment
from .agents import (
    GreedyAgent, EcoGreedyAgent, ThresholdAgent,
    RandomAgent, IdleAgent, get_agent, create_agent
)
from .sarsa_agent import SarsaAgent, ExpectedSarsaAgent

# DQN requires PyTorch
try:
    from .agents import DQNAgent
except ImportError:
    DQNAgent = None

__version__ = "2.1.0"
__author__ = "Adam Younsi, Ilyas Madah, Zaid El Kasemy"
