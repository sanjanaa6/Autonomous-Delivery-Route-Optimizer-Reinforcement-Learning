import numpy as np
from typing import Dict, List, Any

class BaselineRouteSolver:
    """
    Base class for classical non-RL route selection heuristics.
    """
    def __init__(self, name: str):
        self.name = name

    def select_action(self, env) -> int:
        raise NotImplementedError


class ShortestDistanceSolver(BaselineRouteSolver):
    """
    Classical Shortest Path Solver: Always selects route with minimum distance (km).
    """
    def __init__(self):
        super().__init__("Shortest Distance Solver")

    def select_action(self, env) -> int:
        routes = env.routes
        if not routes:
            return 0
        min_idx = 0
        min_dist = float('inf')
        for idx, r in enumerate(routes):
            if r["distance_km"] < min_dist:
                min_dist = r["distance_km"]
                min_idx = idx
        return min_idx


class FastestDurationSolver(BaselineRouteSolver):
    """
    Fastest Duration Solver: Always selects route with minimum travel time (mins), ignoring tolls & priority.
    """
    def __init__(self):
        super().__init__("Fastest Duration Solver")

    def select_action(self, env) -> int:
        routes = env.routes
        if not routes:
            return 0
        min_idx = 0
        min_time = float('inf')
        for idx, r in enumerate(routes):
            if r["duration_min"] < min_time:
                min_time = r["duration_min"]
                min_idx = idx
        return min_idx


class LowestCostSolver(BaselineRouteSolver):
    """
    Lowest Toll Cost Solver: Always selects route with zero or lowest toll cost (₹).
    """
    def __init__(self):
        super().__init__("Lowest Cost Solver")

    def select_action(self, env) -> int:
        routes = env.routes
        if not routes:
            return 0
        min_idx = 0
        min_cost = float('inf')
        for idx, r in enumerate(routes):
            if r["toll_cost"] < min_cost:
                min_cost = r["toll_cost"]
                min_idx = idx
        return min_idx
