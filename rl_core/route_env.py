import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy
from typing import Dict, List, Tuple, Any

class RealWorldRouteEnv(gym.Env):
    """
    OpenAI Gymnasium Environment for Real-World Autonomous Delivery Route Selection.
    Evaluates candidate routes from Google Maps considering distance, travel time,
    traffic congestion multipliers, toll costs, and delivery urgency priorities.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, route_scenario: Dict[str, Any], max_routes: int = 3):
        super(RealWorldRouteEnv, self).__init__()

        self.scenario = copy.deepcopy(route_scenario)
        self.routes = self.scenario.get("routes", [])
        self.num_routes = min(len(self.routes), max_routes)
        self.max_routes = max_routes

        # Action Space: Select Route Option 0, 1, ..., num_routes-1
        self.action_space = spaces.Discrete(self.max_routes)

        # State dimensions:
        # [orig_lat, orig_lng, dest_lat, dest_lng, priority_norm, toll_budget_norm,
        #  for each route: (dist_norm, time_norm, traffic_norm, toll_norm)]
        obs_dim = 6 + (4 * self.max_routes)
        self.observation_space = spaces.Box(
            low=0.0, high=5.0, shape=(obs_dim,), dtype=np.float32
        )

        self.priority = 2  # Default Medium (1=Low, 2=Med, 3=High)
        self.max_toll_budget = 10.0  # Max tolerable toll
        self.current_step = 0

        self.reset()

    def set_scenario(self, route_scenario: Dict[str, Any], priority: int = 2, toll_budget: float = 10.0):
        self.scenario = copy.deepcopy(route_scenario)
        self.routes = self.scenario.get("routes", [])
        self.num_routes = min(len(self.routes), self.max_routes)
        self.priority = priority
        self.max_toll_budget = toll_budget
        return self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def _get_obs(self) -> np.ndarray:
        orig_coords = self.scenario.get("origin_coords", (40.7580, -73.9855))
        dest_coords = self.scenario.get("dest_coords", (40.7075, -74.0089))

        obs = [
            float(orig_coords[0]) / 90.0,
            float(orig_coords[1]) / 180.0,
            float(dest_coords[0]) / 90.0,
            float(dest_coords[1]) / 180.0,
            float(self.priority) / 3.0,
            float(self.max_toll_budget) / 20.0
        ]

        for i in range(self.max_routes):
            if i < len(self.routes):
                r = self.routes[i]
                dist_norm = float(r["distance_km"]) / 50.0
                time_norm = float(r["duration_min"]) / 90.0
                traffic_norm = float(r["traffic_factor"]) / 3.0
                toll_norm = float(r["toll_cost"]) / 20.0
                obs.extend([dist_norm, time_norm, traffic_norm, toll_norm])
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0])

        return np.array(obs, dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "num_routes": self.num_routes,
            "priority": self.priority,
            "toll_budget": self.max_toll_budget
        }

    def calculate_route_reward(self, route_idx: int) -> Tuple[float, Dict[str, float]]:
        if route_idx >= len(self.routes):
            # Invalid route selection penalty
            return -50.0, {"invalid_penalty": -50.0}

        r = self.routes[route_idx]
        dist = r["distance_km"]
        duration = r["duration_min"]
        traffic_tf = r["traffic_factor"]
        toll = r["toll_cost"]

        # Base efficiency reward components
        base_score = 100.0
        time_penalty = duration * 0.5 * (1.0 + (self.priority * 0.25))  # High priority punishes delays more
        dist_penalty = dist * 0.3
        traffic_penalty = (traffic_tf - 1.0) * 15.0
        toll_penalty = toll * 0.8

        budget_exceeded_penalty = 0.0
        if toll > self.max_toll_budget:
            budget_exceeded_penalty = (toll - self.max_toll_budget) * 2.0

        total_reward = base_score - time_penalty - dist_penalty - traffic_penalty - toll_penalty - budget_exceeded_penalty

        breakdown = {
            "base_score": base_score,
            "time_penalty": -round(time_penalty, 2),
            "dist_penalty": -round(dist_penalty, 2),
            "traffic_penalty": -round(traffic_penalty, 2),
            "toll_penalty": -round(toll_penalty, 2),
            "budget_penalty": -round(budget_exceeded_penalty, 2),
            "total_reward": round(total_reward, 2)
        }

        return float(total_reward), breakdown

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        reward, breakdown = self.calculate_route_reward(action)

        info = self._get_info()
        info["selected_route_idx"] = action
        info["reward_breakdown"] = breakdown
        if action < len(self.routes):
            info["selected_route"] = self.routes[action]

        terminated = True  # One-shot route selection task per episode
        truncated = False

        return self._get_obs(), float(reward), terminated, truncated, info
