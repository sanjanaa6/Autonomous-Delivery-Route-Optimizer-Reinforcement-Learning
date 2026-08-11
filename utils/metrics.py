import pandas as pd
import numpy as np
from typing import Dict, List, Any
from rl_core.trainer import evaluate_runner

def run_comprehensive_benchmark(
    env,
    q_agent=None,
    dqn_agent=None,
    solvers_dict=None,
    num_episodes: int = 10
) -> pd.DataFrame:
    """
    Executes a side-by-side benchmark comparing RL algorithms against classical baselines
    across identical city environmental conditions.
    """
    results = []

    if q_agent is not None:
        res_q = evaluate_runner(env, q_agent, solver_type="q_learning", num_episodes=num_episodes)
        res_q["Algorithm"] = "Tabular Q-Learning"
        results.append(res_q)

    if dqn_agent is not None:
        res_dqn = evaluate_runner(env, dqn_agent, solver_type="dqn", num_episodes=num_episodes)
        res_dqn["Algorithm"] = "Deep Q-Network (DQN)"
        results.append(res_dqn)

    if solvers_dict is not None:
        for name, solver_inst in solvers_dict.items():
            res = evaluate_runner(env, solver_inst, solver_type=name, num_episodes=num_episodes)
            res["Algorithm"] = solver_inst.name
            results.append(res)

    df_results = pd.DataFrame([
        {
            "Algorithm": r["Algorithm"],
            "Completion Rate (%)": r["completion_rate"],
            "Mean Reward": r["mean_reward"],
            "Avg Distance (km)": r["avg_distance"],
            "Avg Battery Left (%)": r["avg_battery_left"],
            "Avg Steps": r["avg_steps"],
            "Reward Std Dev": r["std_reward"]
        } for r in results
    ])

    return df_results, results
