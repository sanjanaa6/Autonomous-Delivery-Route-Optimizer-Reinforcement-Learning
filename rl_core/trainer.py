import numpy as np
import pandas as pd
import random
import time
from typing import Dict, List, Tuple, Any, Callable, Optional
from rl_core.route_env import RealWorldRouteEnv
from rl_core.dqn_agent import DQNAgent
from rl_core.explainability import XAIRouteExplainer

def train_dqn_route_agent(
    env: RealWorldRouteEnv,
    num_episodes: int = 150,
    lr: float = 1e-3,
    gamma: float = 0.95,
    epsilon_decay: float = 0.98,
    progress_callback: Optional[Callable[[int, int, float], None]] = None
) -> Tuple[DQNAgent, pd.DataFrame]:
    """
    Train PyTorch Deep Q-Network Agent on RealWorldRouteEnv across varying
    traffic levels and delivery priority conditions.
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
        batch_size=16,
        buffer_capacity=5000
    )

    history = []

    for ep in range(1, num_episodes + 1):
        # Randomize priority & toll budget per episode for robust multi-task policy
        prio = random.choice([1, 2, 3])
        budget = random.choice([0.0, 5.0, 10.0, 15.0])
        obs, info = env.set_scenario(env.scenario, priority=prio, toll_budget=budget)

        action = agent.select_action(obs, eval_mode=False)
        next_obs, reward, terminated, truncated, next_info = env.step(action)
        done = terminated or truncated

        agent.memory.push(obs, action, reward, next_obs, done)
        loss = agent.update()

        agent.decay_epsilon()

        selected_name = env.routes[action]["name"] if action < len(env.routes) else "Invalid"

        history.append({
            "episode": ep,
            "reward": round(reward, 2),
            "loss": round(loss, 4),
            "selected_route": selected_name,
            "priority": prio,
            "epsilon": round(agent.epsilon, 3)
        })

        if progress_callback is not None and (ep % 5 == 0 or ep == num_episodes):
            progress_callback(ep, num_episodes, reward)

    df_history = pd.DataFrame(history)
    return agent, df_history


def evaluate_route_policy(
    env: RealWorldRouteEnv,
    agent_or_solver: Any,
    solver_type: str = "dqn"
) -> Dict[str, Any]:
    """
    Evaluates selected route for a given environment state and generates XAI explanations.
    """
    obs, info = env.reset()

    if solver_type == "dqn":
        action = agent_or_solver.select_action(obs, eval_mode=True)
    else:
        # Classical solver
        action = agent_or_solver.select_action(env)

    # Compute rewards for all route choices to enable XAI comparison
    all_rewards = {}
    all_breakdowns = {}
    for idx in range(len(env.routes)):
        r_val, r_bd = env.calculate_route_reward(idx)
        all_rewards[idx] = r_val
        all_breakdowns[idx] = r_bd

    # Run step for selected action
    next_obs, reward, term, trunc, step_info = env.step(action)

    # Generate XAI Natural Language Explanation
    explainer = XAIRouteExplainer()
    explanation_res = explainer.explain_selection(env, action, all_rewards, all_breakdowns)

    return {
        "solver_type": solver_type,
        "selected_route_idx": action,
        "selected_route": env.routes[action] if action < len(env.routes) else None,
        "total_reward": reward,
        "all_rewards": all_rewards,
        "all_breakdowns": all_breakdowns,
        "explanation": explanation_res
    }
