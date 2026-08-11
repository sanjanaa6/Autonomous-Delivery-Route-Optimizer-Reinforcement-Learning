import numpy as np
import pandas as pd
import torch
import time
from typing import Dict, List, Tuple, Any, Callable, Optional
from rl_core.env import DeliveryEnv
from rl_core.q_learning import QLearningAgent
from rl_core.dqn_agent import DQNAgent

def train_q_learning(
    env: DeliveryEnv,
    num_episodes: int = 150,
    lr: float = 0.1,
    gamma: float = 0.95,
    epsilon_decay: float = 0.99,
    progress_callback: Optional[Callable[[int, int, float, float], None]] = None
) -> Tuple[QLearningAgent, pd.DataFrame]:
    """
    Train Tabular Q-Learning Agent on DeliveryEnv.
    """
    agent = QLearningAgent(
        action_dim=env.action_space.n,
        lr=lr,
        gamma=gamma,
        epsilon=1.0,
        epsilon_decay=epsilon_decay,
        epsilon_min=0.05
    )

    history = []

    for ep in range(1, num_episodes + 1):
        obs, info = env.reset()
        state_key = agent.get_state_key(info)
        ep_reward = 0.0
        done = False
        steps = 0

        while not done:
            action = agent.select_action(env, eval_mode=False)
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            next_state_key = agent.get_state_key(next_info)
            done = terminated or truncated

            agent.update(state_key, action, reward, next_state_key, done)

            state_key = next_state_key
            ep_reward += reward
            steps += 1

        agent.decay_epsilon()
        completion = (next_info["delivered_count"] / max(1, env.num_orders)) * 100.0

        history.append({
            "episode": ep,
            "reward": round(ep_reward, 2),
            "delivered_count": next_info["delivered_count"],
            "completion_rate": round(completion, 1),
            "distance": round(next_info["total_distance"], 2),
            "battery_left": round(next_info["battery"], 1),
            "steps": steps,
            "epsilon": round(agent.epsilon, 3)
        })

        if progress_callback is not None and (ep % 5 == 0 or ep == num_episodes):
            progress_callback(ep, num_episodes, ep_reward, completion)

    df_history = pd.DataFrame(history)
    return agent, df_history


def train_dqn(
    env: DeliveryEnv,
    num_episodes: int = 150,
    lr: float = 1e-3,
    gamma: float = 0.98,
    epsilon_decay: float = 0.99,
    progress_callback: Optional[Callable[[int, int, float, float], None]] = None
) -> Tuple[DQNAgent, pd.DataFrame]:
    """
    Train PyTorch Deep Q-Network Agent on DeliveryEnv.
    """
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        lr=lr,
        gamma=gamma,
        epsilon=1.0,
        epsilon_decay=epsilon_decay,
        epsilon_min=0.05,
        batch_size=32,
        buffer_capacity=10000
    )

    history = []

    for ep in range(1, num_episodes + 1):
        obs, info = env.reset()
        ep_reward = 0.0
        losses = []
        done = False
        steps = 0

        while not done:
            action = agent.select_action(obs, eval_mode=False)
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated

            agent.memory.push(obs, action, reward, next_obs, done)
            loss = agent.update()
            if loss > 0:
                losses.append(loss)

            obs = next_obs
            ep_reward += reward
            steps += 1

        agent.decay_epsilon()
        completion = (next_info["delivered_count"] / max(1, env.num_orders)) * 100.0
        avg_loss = float(np.mean(losses)) if losses else 0.0

        history.append({
            "episode": ep,
            "reward": round(ep_reward, 2),
            "loss": round(avg_loss, 4),
            "delivered_count": next_info["delivered_count"],
            "completion_rate": round(completion, 1),
            "distance": round(next_info["total_distance"], 2),
            "battery_left": round(next_info["battery"], 1),
            "steps": steps,
            "epsilon": round(agent.epsilon, 3)
        })

        if progress_callback is not None and (ep % 5 == 0 or ep == num_episodes):
            progress_callback(ep, num_episodes, ep_reward, completion)

    df_history = pd.DataFrame(history)
    return agent, df_history


def evaluate_runner(
    env: DeliveryEnv,
    agent_or_solver: Any,
    solver_type: str = "q_learning",
    num_episodes: int = 10
) -> Dict[str, Any]:
    """
    Evaluates an agent or baseline solver over test episodes and returns comprehensive
    performance telemetry and trajectory logs.
    """
    rewards = []
    distances = []
    completions = []
    batteries = []
    step_counts = []

    sample_trajectory = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        ep_reward = 0.0
        done = False
        steps = 0
        ep_trajectory = []

        while not done:
            # Capture step snapshot for playback visualization
            ep_trajectory.append({
                "step": steps,
                "current_node": info["current_node"],
                "battery": info["battery"],
                "payload": info["payload"],
                "total_distance": info["total_distance"],
                "delivered_count": info["delivered_count"],
                "orders": copy_orders(info["orders"])
            })

            if solver_type == "q_learning":
                action = agent_or_solver.select_action(env, eval_mode=True)
            elif solver_type == "dqn":
                action = agent_or_solver.select_action(obs, eval_mode=True)
            else:
                # Traditional baseline solver
                action = agent_or_solver.select_action(env)

            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated

            obs = next_obs
            info = next_info
            ep_reward += reward
            steps += 1

        # Final step snapshot
        ep_trajectory.append({
            "step": steps,
            "current_node": info["current_node"],
            "battery": info["battery"],
            "payload": info["payload"],
            "total_distance": info["total_distance"],
            "delivered_count": info["delivered_count"],
            "orders": copy_orders(info["orders"])
        })

        rewards.append(ep_reward)
        distances.append(info["total_distance"])
        completions.append((info["delivered_count"] / max(1, env.num_orders)) * 100.0)
        batteries.append(info["battery"])
        step_counts.append(steps)

        if ep == 0:
            sample_trajectory = ep_trajectory

    return {
        "solver_name": solver_type.upper(),
        "mean_reward": round(float(np.mean(rewards)), 2),
        "std_reward": round(float(np.std(rewards)), 2),
        "completion_rate": round(float(np.mean(completions)), 1),
        "avg_distance": round(float(np.mean(distances)), 2),
        "avg_battery_left": round(float(np.mean(batteries)), 1),
        "avg_steps": round(float(np.mean(step_counts)), 1),
        "trajectory": sample_trajectory
    }


def copy_orders(orders_list):
    return [
        {
            "order_id": o["order_id"],
            "node": o["node"],
            "priority": o["priority"],
            "weight": o["weight"],
            "deadline": o["deadline"],
            "status": o["status"],
            "delivery_time": o["delivery_time"]
        } for o in orders_list
    ]
