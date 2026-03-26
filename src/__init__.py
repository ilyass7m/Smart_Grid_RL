"""
Eco-Adaptive Home Storage: Multi-Objective RL for Smart Grid Energy Management
===============================================================================

A reinforcement learning project for intelligent battery arbitrage in smart homes.

Modules:
    environment: SmartGridEnv - Gymnasium-compatible environment
    dqn_agent: DQNAgent - Deep Q-Network implementation
    baseline_agent: Rule-based baseline agents
    train: Training and evaluation utilities
"""

from .environment import SmartGridEnv
from .dqn_agent import DQNAgent
from .baseline_agent import GreedyAgent, EcoGreedyAgent, ThresholdAgent, RandomAgent

__version__ = "1.0.0"
__author__ = "Adam Younsi, Ilyas Madah, Zaid El Kasemy"
