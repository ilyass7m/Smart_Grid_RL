"""
Training Script for Smart Grid RL Agent
========================================
Trains DQN agent and evaluates against baselines.

Features:
- Training loop with progress tracking
- Periodic evaluation
- Learning curve visualization
- Model checkpointing
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import os
import time
from datetime import datetime

from environment import SmartGridEnv
from dqn_agent import DQNAgent
from baseline_agent import GreedyAgent, EcoGreedyAgent, ThresholdAgent, RandomAgent


def train_dqn(
    env: SmartGridEnv,
    agent: DQNAgent,
    num_episodes: int = 500,
    eval_interval: int = 50,
    verbose: bool = True
) -> Dict:
    """
    Train DQN agent on the Smart Grid environment.

    Args:
        env: SmartGridEnv instance
        agent: DQNAgent instance
        num_episodes: Number of training episodes
        eval_interval: Episodes between evaluations
        verbose: Print progress

    Returns:
        Dictionary with training history
    """
    history = {
        'episode_rewards': [],
        'episode_costs': [],
        'episode_emissions': [],
        'episode_lengths': [],
        'epsilon_values': [],
        'eval_rewards': [],
        'eval_costs': [],
        'eval_emissions': []
    }

    start_time = time.time()

    for episode in range(num_episodes):
        # Reset environment
        state, _ = env.reset()
        action = agent.agent_start(state)

        episode_reward = 0
        done = False
        steps = 0

        while not done:
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated

            if done:
                agent.agent_end(reward)
            else:
                action = agent.agent_step(reward, next_state)
                state = next_state

        # Record episode statistics
        summary = env.get_episode_summary()
        history['episode_rewards'].append(episode_reward)
        history['episode_costs'].append(summary['total_cost'])
        history['episode_emissions'].append(summary['total_emissions'])
        history['episode_lengths'].append(steps)
        history['epsilon_values'].append(agent.epsilon)

        # Periodic evaluation
        if (episode + 1) % eval_interval == 0:
            eval_reward, eval_cost, eval_emissions = evaluate_agent(env, agent, num_episodes=10)
            history['eval_rewards'].append(eval_reward)
            history['eval_costs'].append(eval_cost)
            history['eval_emissions'].append(eval_emissions)

            if verbose:
                elapsed = time.time() - start_time
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Reward: {eval_reward:.2f} | "
                      f"Cost: €{eval_cost:.3f} | "
                      f"Emissions: {eval_emissions:.2f}kg | "
                      f"ε: {agent.epsilon:.3f} | "
                      f"Time: {elapsed:.1f}s")

    return history


def evaluate_agent(
    env: SmartGridEnv,
    agent,
    num_episodes: int = 10,
    render: bool = False
) -> Tuple[float, float, float]:
    """
    Evaluate an agent over multiple episodes.

    Returns:
        (mean_reward, mean_cost, mean_emissions)
    """
    rewards = []
    costs = []
    emissions = []

    # Store original epsilon for DQN
    original_epsilon = getattr(agent, 'epsilon', None)
    if original_epsilon is not None:
        agent.epsilon = 0.0  # Greedy evaluation

    for _ in range(num_episodes):
        state, _ = env.reset()

        # Handle both DQN and baseline agents
        if hasattr(agent, 'agent_start'):
            action = agent.agent_start(state)
        else:
            action = agent.select_action(state, training=False)

        episode_reward = 0
        done = False

        while not done:
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

            if not done:
                if hasattr(agent, 'agent_step'):
                    # For baseline agents, don't update (just get action)
                    if hasattr(agent, '_select_action'):
                        action = agent._select_action(state)
                    else:
                        action = agent.select_action(state, training=False)
                else:
                    action = agent.select_action(state, training=False)

            if render:
                env.render()

        summary = env.get_episode_summary()
        rewards.append(episode_reward)
        costs.append(summary['total_cost'])
        emissions.append(summary['total_emissions'])

    # Restore epsilon
    if original_epsilon is not None:
        agent.epsilon = original_epsilon

    return np.mean(rewards), np.mean(costs), np.mean(emissions)


def compare_agents(
    env: SmartGridEnv,
    agents: Dict[str, object],
    num_episodes: int = 100,
    verbose: bool = True
) -> Dict:
    """
    Compare multiple agents on the same environment.

    Args:
        env: SmartGridEnv instance
        agents: Dictionary of agent_name -> agent instance
        num_episodes: Number of evaluation episodes

    Returns:
        Dictionary with comparison results
    """
    results = {}

    for name, agent in agents.items():
        if verbose:
            print(f"Evaluating {name}...")

        reward, cost, emissions = evaluate_agent(env, agent, num_episodes)
        results[name] = {
            'mean_reward': reward,
            'mean_cost': cost,
            'mean_emissions': emissions
        }

        if verbose:
            print(f"  Reward: {reward:.2f}, Cost: €{cost:.3f}, Emissions: {emissions:.2f}kg")

    return results


def plot_training_curves(history: Dict, save_path: str = None):
    """Plot training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Episode rewards (smoothed)
    ax = axes[0, 0]
    rewards = history['episode_rewards']
    window = min(50, len(rewards) // 10)
    if window > 1:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, 'b-', linewidth=1.5)
        ax.plot(rewards, 'b-', alpha=0.2)
    else:
        ax.plot(rewards, 'b-')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Episode Rewards (Training)')
    ax.grid(True, alpha=0.3)

    # Episode costs
    ax = axes[0, 1]
    costs = history['episode_costs']
    if window > 1:
        smoothed = np.convolve(costs, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, 'r-', linewidth=1.5)
        ax.plot(costs, 'r-', alpha=0.2)
    else:
        ax.plot(costs, 'r-')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cost (€)')
    ax.set_title('Episode Costs')
    ax.grid(True, alpha=0.3)

    # Episode emissions
    ax = axes[1, 0]
    emissions = history['episode_emissions']
    if window > 1:
        smoothed = np.convolve(emissions, np.ones(window)/window, mode='valid')
        ax.plot(smoothed, 'g-', linewidth=1.5)
        ax.plot(emissions, 'g-', alpha=0.2)
    else:
        ax.plot(emissions, 'g-')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Emissions (kgCO2)')
    ax.set_title('Episode Emissions')
    ax.grid(True, alpha=0.3)

    # Epsilon decay
    ax = axes[1, 1]
    ax.plot(history['epsilon_values'], 'purple', linewidth=1.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Epsilon')
    ax.set_title('Exploration Rate (ε)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")

    plt.show()


def plot_comparison(results: Dict, save_path: str = None):
    """Plot agent comparison bar chart."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    agents = list(results.keys())
    x = np.arange(len(agents))
    width = 0.6

    # Rewards
    ax = axes[0]
    rewards = [results[a]['mean_reward'] for a in agents]
    colors = ['#2ecc71' if r == max(rewards) else '#3498db' for r in rewards]
    ax.bar(x, rewards, width, color=colors)
    ax.set_ylabel('Mean Reward')
    ax.set_title('Reward Comparison (Higher is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Costs
    ax = axes[1]
    costs = [results[a]['mean_cost'] for a in agents]
    colors = ['#2ecc71' if c == min(costs) else '#e74c3c' for c in costs]
    ax.bar(x, costs, width, color=colors)
    ax.set_ylabel('Mean Cost (€)')
    ax.set_title('Cost Comparison (Lower is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Emissions
    ax = axes[2]
    emissions = [results[a]['mean_emissions'] for a in agents]
    colors = ['#2ecc71' if e == min(emissions) else '#9b59b6' for e in emissions]
    ax.bar(x, emissions, width, color=colors)
    ax.set_ylabel('Mean Emissions (kgCO2)')
    ax.set_title('Emissions Comparison (Lower is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")

    plt.show()


def plot_policy_analysis(env: SmartGridEnv, agent, num_episodes: int = 1, save_path: str = None):
    """
    Visualize the agent's policy over a day.
    Shows SoC, actions, price, and carbon intensity.
    """
    # Run one episode and collect data
    state, _ = env.reset(seed=42)

    if hasattr(agent, 'epsilon'):
        original_epsilon = agent.epsilon
        agent.epsilon = 0.0

    action = agent.agent_start(state) if hasattr(agent, 'agent_start') else agent.select_action(state, training=False)

    hours = [state[0]]
    socs = [state[1]]
    actions = []
    prices = [state[3]]
    co2s = [state[4]]

    done = False
    while not done:
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        hours.append(next_state[0])
        socs.append(next_state[1])
        actions.append(action)
        prices.append(next_state[3])
        co2s.append(next_state[4])

        if not done:
            if hasattr(agent, '_select_action'):
                action = agent._select_action(next_state)
            elif hasattr(agent, 'select_action'):
                action = agent.select_action(next_state, training=False)
            else:
                action = agent.agent_step(reward, next_state)

    if hasattr(agent, 'epsilon'):
        agent.epsilon = original_epsilon

    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    time_steps = range(len(hours))

    # SoC
    ax = axes[0]
    ax.plot(time_steps, socs, 'b-', linewidth=2, label='State of Charge')
    ax.fill_between(time_steps, 0, socs, alpha=0.3)
    ax.set_ylabel('SoC (%)')
    ax.set_ylim(0, 1)
    ax.set_title('Battery State of Charge Over 24 Hours')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Actions
    ax = axes[1]
    action_colors = ['#2ecc71', '#e74c3c', '#95a5a6']  # Green=Charge, Red=Discharge, Gray=Idle
    action_labels = ['Charge', 'Discharge', 'Idle']
    for i, a in enumerate(actions):
        ax.bar(i + 0.5, 1, width=1, color=action_colors[a], alpha=0.7)
    ax.set_ylabel('Action')
    ax.set_yticks([0.5])
    ax.set_yticklabels([''])
    ax.set_title('Agent Actions')
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=l) for c, l in zip(action_colors, action_labels)]
    ax.legend(handles=legend_elements, loc='upper right', ncol=3)

    # Price
    ax = axes[2]
    ax.plot(time_steps, prices, 'orange', linewidth=2)
    ax.fill_between(time_steps, 0, prices, alpha=0.3, color='orange')
    ax.set_ylabel('Price (€/kWh)')
    ax.set_title('Grid Electricity Price')
    ax.grid(True, alpha=0.3)

    # CO2 Intensity
    ax = axes[3]
    ax.plot(time_steps, co2s, 'purple', linewidth=2)
    ax.fill_between(time_steps, 0, co2s, alpha=0.3, color='purple')
    ax.set_ylabel('CO2 (g/kWh)')
    ax.set_xlabel('Time Step (30-min intervals)')
    ax.set_title('Grid Carbon Intensity')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved policy analysis to {save_path}")

    plt.show()


def main():
    """Main training and evaluation script."""
    print("=" * 60)
    print("Eco-Adaptive Home Storage: Smart Grid RL Training")
    print("=" * 60)

    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
    os.makedirs(output_dir, exist_ok=True)

    # Initialize environment
    env = SmartGridEnv(
        alpha=1.0,      # Cost weight
        beta=0.5,       # Emissions weight
        gamma_wear=0.1, # Battery wear weight
        seed=42
    )

    print(f"\nEnvironment initialized:")
    print(f"  State space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Episode length: {env.max_steps} steps ({env.episode_hours}h)")

    # Initialize DQN agent
    agent = DQNAgent(
        state_dim=5,
        action_dim=3,
        hidden_dims=[128, 128],
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        batch_size=64,
        target_update_freq=100,
        seed=42
    )

    print(f"\nDQN Agent initialized:")
    print(f"  Network: 5 -> 128 -> 128 -> 3")
    print(f"  Device: {agent.device}")

    # Train DQN
    print("\n" + "=" * 60)
    print("Training DQN Agent...")
    print("=" * 60)

    history = train_dqn(
        env=env,
        agent=agent,
        num_episodes=500,
        eval_interval=50,
        verbose=True
    )

    # Save model
    model_path = os.path.join(output_dir, 'dqn_model.pth')
    agent.save(model_path)
    print(f"\nModel saved to {model_path}")

    # Plot training curves
    plot_training_curves(
        history,
        save_path=os.path.join(output_dir, 'training_curves.png')
    )

    # Compare with baselines
    print("\n" + "=" * 60)
    print("Comparing with Baseline Agents...")
    print("=" * 60)

    agents = {
        'DQN': agent,
        'Greedy': GreedyAgent(),
        'EcoGreedy': EcoGreedyAgent(),
        'Threshold': ThresholdAgent(),
        'Random': RandomAgent(seed=42)
    }

    # Initialize baselines
    for name, a in agents.items():
        if hasattr(a, 'agent_init'):
            a.agent_init({})

    comparison = compare_agents(env, agents, num_episodes=100, verbose=True)

    # Plot comparison
    plot_comparison(
        comparison,
        save_path=os.path.join(output_dir, 'agent_comparison.png')
    )

    # Policy analysis
    print("\n" + "=" * 60)
    print("Policy Analysis...")
    print("=" * 60)

    plot_policy_analysis(
        env, agent,
        save_path=os.path.join(output_dir, 'policy_analysis.png')
    )

    # Print summary table
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Agent':<15} {'Reward':>12} {'Cost (€)':>12} {'CO2 (kg)':>12}")
    print("-" * 55)
    for name, res in comparison.items():
        print(f"{name:<15} {res['mean_reward']:>12.2f} {res['mean_cost']:>12.3f} {res['mean_emissions']:>12.2f}")

    # Calculate savings vs baseline
    baseline_cost = comparison['Greedy']['mean_cost']
    dqn_cost = comparison['DQN']['mean_cost']
    baseline_emissions = comparison['Greedy']['mean_emissions']
    dqn_emissions = comparison['DQN']['mean_emissions']

    print("\n" + "-" * 55)
    print(f"DQN vs Greedy:")
    print(f"  Cost savings: €{baseline_cost - dqn_cost:.3f}/day ({(baseline_cost - dqn_cost)/baseline_cost*100:.1f}%)")
    print(f"  CO2 reduction: {baseline_emissions - dqn_emissions:.2f}kg/day ({(baseline_emissions - dqn_emissions)/baseline_emissions*100:.1f}%)")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
