import pandas as pd
import numpy as np
from typing import Dict, List, Any
from rl_core.trainer import evaluate_route_policy

def run_route_benchmarks(
    env,
    dqn_agent=None,
    solvers_dict=None
) -> pd.DataFrame:
    """
    Runs side-by-side benchmark comparing Deep Q-Network policy against classical heuristics
    across the active Google Maps route options.
    """
    rows = []

    if dqn_agent is not None:
        eval_res = evaluate_route_policy(env, dqn_agent, solver_type="dqn")
        r = eval_res["selected_route"]
        if r:
            rows.append({
                "Algorithm / Policy": "🤖 Deep Q-Network (DQN)",
                "Selected Route": r["name"],
                "Distance (km)": r["distance_km"],
                "Duration (min)": r["duration_min"],
                "Traffic Factor": f"{r['traffic_factor']:.2f}x",
                "Toll Cost ($)": f"${r['toll_cost']:.2f}",
                "Efficiency Reward": eval_res["total_reward"]
            })

    if solvers_dict:
        for solver_name, solver_inst in solvers_dict.items():
            eval_res = evaluate_route_policy(env, solver_inst, solver_type=solver_name)
            r = eval_res["selected_route"]
            if r:
                rows.append({
                    "Algorithm / Policy": solver_inst.name,
                    "Selected Route": r["name"],
                    "Distance (km)": r["distance_km"],
                    "Duration (min)": r["duration_min"],
                    "Traffic Factor": f"{r['traffic_factor']:.2f}x",
                    "Toll Cost ($)": f"${r['toll_cost']:.2f}",
                    "Efficiency Reward": eval_res["total_reward"]
                })

    return pd.DataFrame(rows)
