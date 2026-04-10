"""
Experiment Utilities for Smart Grid RL
======================================
Training, evaluation, and statistical comparison utilities.

Features:
- Training loop with progress tracking
- Multi-run statistical evaluation
- Agent comparison with confidence intervals
- Visualization utilities
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Type
from dataclasses import dataclass
from scipy import stats
import time
import os

from rl_glue import RLGlue, BaseAgent, BaseEnvironment
from smart_grid_env import SmartGridEnvironment
from agents import (
    DQNAgent, GreedyAgent, EcoGreedyAgent,
    ThresholdAgent, RandomAgent, IdleAgent,
    get_agent, create_agent
)


@dataclass
class TrainingResult:
    """Results from a training run."""
    rewards: List[float]
    costs: List[float]
    emissions: List[float]
    epsilon_history: List[float]
    training_time: float
    final_agent: BaseAgent


@dataclass
class EvaluationResult:
    """Results from evaluating an agent."""
    rewards: np.ndarray
    costs: np.ndarray
    emissions: np.ndarray
    baseline_costs: np.ndarray
    baseline_emissions: np.ndarray
    euros_saved: np.ndarray
    co2_avoided: np.ndarray
    action_distributions: np.ndarray


@dataclass
class StatisticalSummary:
    """Statistical summary of evaluation results."""
    mean: float
    std: float
    ci_low: float
    ci_high: float
    min_val: float
    max_val: float
    n_samples: int


def compute_statistics(values: np.ndarray, confidence: float = 0.95) -> StatisticalSummary:
    """
    Compute statistical summary with confidence interval.

    Args:
        values: Array of values
        confidence: Confidence level (default 95%)

    Returns:
        StatisticalSummary object
    """
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)

    # Confidence interval using t-distribution
    t_critical = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_critical * std / np.sqrt(n)

    return StatisticalSummary(
        mean=mean,
        std=std,
        ci_low=mean - margin,
        ci_high=mean + margin,
        min_val=np.min(values),
        max_val=np.max(values),
        n_samples=n
    )


def train_agent(
    agent_class: Type[BaseAgent],
    env_class: Type[BaseEnvironment],
    agent_info: Dict = None,
    env_info: Dict = None,
    num_episodes: int = 1000,
    eval_interval: int = 100,
    verbose: bool = True
) -> TrainingResult:
    """
    Train an agent using RL-GLUE interface.

    Args:
        agent_class: Agent class to train
        env_class: Environment class
        agent_info: Agent initialization parameters
        env_info: Environment initialization parameters
        num_episodes: Number of training episodes
        eval_interval: Episodes between progress reports
        verbose: Print progress

    Returns:
        TrainingResult object
    """
    agent_info = agent_info or {}
    env_info = env_info or {}

    # Initialize RL-GLUE
    rl_glue = RLGlue(env_class, agent_class)
    rl_glue.rl_init(agent_info, env_info)

    # Tracking
    rewards = []
    costs = []
    emissions = []
    epsilon_history = []

    start_time = time.time()

    for episode in range(num_episodes):
        # Run episode
        rl_glue.rl_episode(max_steps=0)

        # Get episode summary
        summary = rl_glue.rl_env_message("get_episode_summary")

        rewards.append(rl_glue.total_reward)
        costs.append(summary.get('total_cost', 0))
        emissions.append(summary.get('total_emissions', 0))

        # Get epsilon if available
        eps = rl_glue.rl_agent_message("get_epsilon")
        epsilon_history.append(eps if eps is not None else 0)

        # Progress report
        if verbose and (episode + 1) % eval_interval == 0:
            recent_rewards = rewards[-eval_interval:]
            recent_costs = costs[-eval_interval:]
            print(f"Episode {episode + 1}/{num_episodes} | "
                  f"Reward: {np.mean(recent_rewards):.3f} | "
                  f"Cost: €{np.mean(recent_costs):.3f} | "
                  f"ε: {epsilon_history[-1]:.3f}")

    training_time = time.time() - start_time

    if verbose:
        print(f"\nTraining completed in {training_time:.1f}s")

    return TrainingResult(
        rewards=rewards,
        costs=costs,
        emissions=emissions,
        epsilon_history=epsilon_history,
        training_time=training_time,
        final_agent=rl_glue.agent
    )


def evaluate_agent(
    agent: BaseAgent,
    env_class: Type[BaseEnvironment],
    env_info: Dict = None,
    num_episodes: int = 100,
    seed_offset: int = 1000
) -> EvaluationResult:
    """
    Evaluate an agent over multiple episodes.

    Args:
        agent: Agent to evaluate (already initialized)
        env_class: Environment class
        env_info: Environment parameters
        num_episodes: Number of evaluation episodes
        seed_offset: Starting seed for reproducibility

    Returns:
        EvaluationResult object
    """
    env_info = env_info or {}

    # Set agent to evaluation mode
    agent.agent_message("set_eval_mode")

    # Results storage
    rewards = []
    costs = []
    emissions = []
    baseline_costs = []
    baseline_emissions = []
    action_counts = []

    for ep in range(num_episodes):
        # Create environment with unique seed
        env = env_class()
        ep_env_info = env_info.copy()
        ep_env_info['seed'] = seed_offset + ep
        env.env_init(ep_env_info)

        # Run episode
        state = env.env_start()
        action = agent.agent_start(state)
        action_dist = [0, 0, 0]

        total_reward = 0
        terminal = False

        while not terminal:
            reward, state, terminal = env.env_step(action)
            total_reward += reward
            action_dist[action] += 1

            if not terminal:
                action = agent.agent_step(reward, state)

        agent.agent_end(reward)

        # Get summary
        summary = env.get_episode_summary()

        rewards.append(total_reward)
        costs.append(summary['total_cost'])
        emissions.append(summary['total_emissions'])
        baseline_costs.append(summary['baseline_cost'])
        baseline_emissions.append(summary['baseline_emissions'])
        action_counts.append(action_dist)

    # Convert to arrays
    rewards = np.array(rewards)
    costs = np.array(costs)
    emissions = np.array(emissions)
    baseline_costs = np.array(baseline_costs)
    baseline_emissions = np.array(baseline_emissions)
    action_counts = np.array(action_counts)

    return EvaluationResult(
        rewards=rewards,
        costs=costs,
        emissions=emissions,
        baseline_costs=baseline_costs,
        baseline_emissions=baseline_emissions,
        euros_saved=baseline_costs - costs,
        co2_avoided=baseline_emissions - emissions,
        action_distributions=action_counts
    )


def compare_agents(
    agents: Dict[str, BaseAgent],
    env_class: Type[BaseEnvironment],
    env_info: Dict = None,
    num_episodes: int = 100,
    verbose: bool = True
) -> Dict[str, Dict[str, StatisticalSummary]]:
    """
    Compare multiple agents with statistical analysis.

    Args:
        agents: Dictionary of agent_name -> agent instance
        env_class: Environment class
        env_info: Environment parameters
        num_episodes: Number of evaluation episodes per agent
        verbose: Print progress

    Returns:
        Dictionary of agent_name -> metrics -> StatisticalSummary
    """
    results = {}

    for name, agent in agents.items():
        if verbose:
            print(f"Evaluating {name}...")

        eval_result = evaluate_agent(
            agent, env_class, env_info, num_episodes
        )

        results[name] = {
            'reward': compute_statistics(eval_result.rewards),
            'cost': compute_statistics(eval_result.costs),
            'emissions': compute_statistics(eval_result.emissions),
            'euros_saved': compute_statistics(eval_result.euros_saved),
            'co2_avoided': compute_statistics(eval_result.co2_avoided),
            'action_dist': eval_result.action_distributions.mean(axis=0)
        }

        if verbose:
            print(f"  Reward: {results[name]['reward'].mean:.3f} ± {results[name]['reward'].std:.3f}")
            print(f"  Savings: €{results[name]['euros_saved'].mean:.3f} ± €{results[name]['euros_saved'].std:.3f}")
            print(f"  CO2: {results[name]['co2_avoided'].mean:.3f} ± {results[name]['co2_avoided'].std:.3f} kg")

    return results


def run_policy_episode(
    agent: BaseAgent,
    env_class: Type[BaseEnvironment],
    env_info: Dict = None
) -> Dict[str, Any]:
    """
    Run a single episode and collect detailed trajectory data.

    Args:
        agent: Agent to run
        env_class: Environment class
        env_info: Environment parameters

    Returns:
        Dictionary with trajectory data
    """
    env_info = env_info or {}
    env = env_class()
    env.env_init(env_info)

    # Set eval mode
    agent.agent_message("set_eval_mode")

    # Run episode
    state = env.env_start()
    action = agent.agent_start(state)

    states = [state.copy()]
    actions = []
    rewards = []

    terminal = False
    while not terminal:
        reward, state, terminal = env.env_step(action)
        rewards.append(reward)
        actions.append(action)
        states.append(state.copy())  # Always append state (including final)

        if not terminal:
            action = agent.agent_step(reward, state)

    agent.agent_end(reward)

    # Get profiles and summary
    profiles = env.env_message("get_profiles")
    summary = env.get_episode_summary()

    return {
        'states': np.array(states),  # Length: steps + 1 (includes initial and final)
        'actions': np.array(actions),  # Length: steps
        'rewards': np.array(rewards),  # Length: steps
        'profiles': profiles,
        'summary': summary,
        'episode_info': env.episode_info  # prices, carbons, etc. have length: steps
    }


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_training_curves(result: TrainingResult, save_path: str = None, show: bool = True):
    """Plot training curves with smoothing."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    window = min(50, len(result.rewards) // 10) if len(result.rewards) > 50 else 1

    # Rewards
    ax = axes[0, 0]
    if window > 1:
        smoothed = np.convolve(result.rewards, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, 'b-', linewidth=1.5, label='Smoothed')
        ax.plot(result.rewards, 'b-', alpha=0.2, label='Raw')
    else:
        ax.plot(result.rewards, 'b-', linewidth=1.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Learning Curve - Episode Rewards')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Costs
    ax = axes[0, 1]
    if window > 1:
        smoothed = np.convolve(result.costs, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, 'r-', linewidth=1.5)
        ax.plot(result.costs, 'r-', alpha=0.2)
    else:
        ax.plot(result.costs, 'r-')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cost (€)')
    ax.set_title('Episode Electricity Cost')
    ax.grid(True, alpha=0.3)

    # Emissions
    ax = axes[1, 0]
    if window > 1:
        smoothed = np.convolve(result.emissions, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, 'g-', linewidth=1.5)
        ax.plot(result.emissions, 'g-', alpha=0.2)
    else:
        ax.plot(result.emissions, 'g-')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Emissions (kg CO₂)')
    ax.set_title('Episode Carbon Emissions')
    ax.grid(True, alpha=0.3)

    # Epsilon
    ax = axes[1, 1]
    ax.plot(result.epsilon_history, 'purple', linewidth=1.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Epsilon (ε)')
    ax.set_title('Exploration Rate Decay')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_comparison_bars(
    results: Dict[str, Dict[str, StatisticalSummary]],
    save_path: str = None,
    show: bool = True
):
    """Plot agent comparison with error bars."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    agents = list(results.keys())
    x = np.arange(len(agents))
    width = 0.6

    # Colors: best in green
    def get_colors(values, higher_better=True):
        if higher_better:
            best = max(values)
        else:
            best = min(values)
        return ['#2ecc71' if v == best else '#3498db' for v in values]

    # Rewards
    ax = axes[0]
    means = [results[a]['reward'].mean for a in agents]
    stds = [results[a]['reward'].std for a in agents]
    colors = get_colors(means, higher_better=True)
    bars = ax.bar(x, means, width, yerr=stds, capsize=5, color=colors, alpha=0.8)
    ax.set_ylabel('Mean Reward')
    ax.set_title('Reward Comparison\n(Higher is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Euros saved
    ax = axes[1]
    means = [results[a]['euros_saved'].mean for a in agents]
    stds = [results[a]['euros_saved'].std for a in agents]
    colors = get_colors(means, higher_better=True)
    ax.bar(x, means, width, yerr=stds, capsize=5, color=colors, alpha=0.8)
    ax.set_ylabel('Euros Saved (€/day)')
    ax.set_title('Cost Savings vs No-Storage\n(Higher is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # CO2 avoided
    ax = axes[2]
    means = [results[a]['co2_avoided'].mean for a in agents]
    stds = [results[a]['co2_avoided'].std for a in agents]
    colors = get_colors(means, higher_better=True)
    ax.bar(x, means, width, yerr=stds, capsize=5, color=colors, alpha=0.8)
    ax.set_ylabel('CO₂ Avoided (kg/day)')
    ax.set_title('Emissions Reduction vs No-Storage\n(Higher is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_policy_analysis(
    trajectory: Dict[str, Any],
    agent_name: str = "Agent",
    save_path: str = None,
    show: bool = True
):
    """Plot detailed policy analysis."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)

    states = trajectory['states']
    actions = trajectory['actions']
    info = trajectory['episode_info']

    # states has length steps+1 (initial + all step states)
    # actions, prices, carbons have length steps
    num_steps = len(actions)

    # Time arrays
    time_states = np.arange(len(states)) * 0.5  # For states (steps+1 points)
    time_actions = np.arange(num_steps) * 0.5   # For actions/prices/carbons (steps points)

    # 1. SoC trajectory
    ax = axes[0]
    soc = states[:, 1]
    ax.plot(time_states, soc, 'b-', linewidth=2, label='State of Charge')
    ax.fill_between(time_states, 0, soc, alpha=0.3)
    ax.set_ylabel('SoC')
    ax.set_ylim(0, 1)
    ax.set_title(f'{agent_name} - Battery State of Charge')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # 2. Actions
    ax = axes[1]
    action_colors = ['#2ecc71', '#e74c3c', '#95a5a6']
    action_names = ['Charge', 'Discharge', 'Idle']
    for i, a in enumerate(actions):
        ax.bar(time_actions[i] + 0.25, 1, width=0.5, color=action_colors[a], alpha=0.8)
    ax.set_ylabel('Action')
    ax.set_yticks([0.5])
    ax.set_yticklabels([''])
    ax.set_title('Actions Taken')
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=n) for c, n in zip(action_colors, action_names)]
    ax.legend(handles=legend_elements, loc='upper right', ncol=3)

    # 3. Price
    ax = axes[2]
    prices = info['prices']
    ax.plot(time_actions, prices, 'orange', linewidth=2)
    ax.fill_between(time_actions, 0, prices, alpha=0.3, color='orange')
    ax.set_ylabel('Price (€/kWh)')
    ax.set_title('Electricity Price')
    ax.grid(True, alpha=0.3)

    # 4. Carbon
    ax = axes[3]
    carbons = info['carbons']
    ax.plot(time_actions, carbons, 'purple', linewidth=2)
    ax.fill_between(time_actions, 0, carbons, alpha=0.3, color='purple')
    ax.set_ylabel('Carbon (gCO₂/kWh)')
    ax.set_xlabel('Hour of Day')
    ax.set_title('Grid Carbon Intensity')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_action_distribution(
    results: Dict[str, Dict[str, StatisticalSummary]],
    save_path: str = None,
    show: bool = True
):
    """Plot action distribution comparison."""
    fig, ax = plt.subplots(figsize=(12, 6))

    agents = list(results.keys())
    x = np.arange(len(agents))
    width = 0.25

    action_names = ['Charge', 'Discharge', 'Idle']
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']

    for i, (action_name, color) in enumerate(zip(action_names, colors)):
        counts = [results[a]['action_dist'][i] for a in agents]
        ax.bar(x + (i - 1) * width, counts, width, label=action_name, color=color, alpha=0.8)

    ax.set_ylabel('Average Action Count per Episode')
    ax.set_title('Action Distribution by Agent')
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def print_comparison_table(results: Dict[str, Dict[str, StatisticalSummary]]):
    """Print formatted comparison table."""
    print("\n" + "=" * 90)
    print("AGENT COMPARISON RESULTS (95% Confidence Intervals)")
    print("=" * 90)

    header = f"{'Agent':<15} {'Reward':>15} {'Savings (€)':>18} {'CO₂ Avoided (kg)':>20}"
    print(header)
    print("-" * 90)

    for name, metrics in results.items():
        reward = metrics['reward']
        savings = metrics['euros_saved']
        co2 = metrics['co2_avoided']

        row = (f"{name:<15} "
               f"{reward.mean:>7.2f} ± {reward.std:>5.2f} "
               f"{savings.mean:>8.3f} ± {savings.std:>6.3f} "
               f"{co2.mean:>10.3f} ± {co2.std:>6.3f}")
        print(row)

    print("=" * 90)


def save_results_csv(
    results: Dict[str, Dict[str, StatisticalSummary]],
    filepath: str
):
    """Save results to CSV file."""
    import csv

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Agent', 'Reward_Mean', 'Reward_Std', 'Reward_CI_Low', 'Reward_CI_High',
            'Savings_Mean', 'Savings_Std', 'CO2_Mean', 'CO2_Std'
        ])

        for name, metrics in results.items():
            writer.writerow([
                name,
                metrics['reward'].mean, metrics['reward'].std,
                metrics['reward'].ci_low, metrics['reward'].ci_high,
                metrics['euros_saved'].mean, metrics['euros_saved'].std,
                metrics['co2_avoided'].mean, metrics['co2_avoided'].std
            ])

    print(f"Results saved to {filepath}")
