import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Any

class BaselineSolver:
    """
    Base class for non-RL classical route planning solvers.
    """
    def __init__(self, name: str):
        self.name = name

    def select_action(self, env) -> int:
        raise NotImplementedError

class DijkstraSolver(BaselineSolver):
    """
    Static shortest path Dijkstra solver. Computes shortest path based on base road distances.
    """
    def __init__(self):
        super().__init__("Static Dijkstra Shortest Path")

    def select_action(self, env) -> int:
        G = env.G
        current = env.current_node
        orders = env.orders
        battery = env.battery
        num_nodes = env.num_nodes

        # Filter pending orders
        pending_orders = [o for o in orders if o["status"] == "pending"]

        if not pending_orders:
            # If all delivered or failed, head back to depot
            if current == env.depot_node:
                return num_nodes  # Wait
            try:
                path = nx.shortest_path(G, source=current, target=env.depot_node, weight='base_distance')
                return path[1] if len(path) > 1 else num_nodes
            except nx.NetworkXNoPath:
                return num_nodes

        # Check if battery low (< 25%), target nearest charger
        if battery < 25.0 and current not in env.charger_nodes:
            charger_paths = []
            for c in env.charger_nodes + [env.depot_node]:
                try:
                    length = nx.shortest_path_length(G, source=current, target=c, weight='base_distance')
                    charger_paths.append((length, c))
                except nx.NetworkXNoPath:
                    pass
            if charger_paths:
                charger_paths.sort()
                target_charger = charger_paths[0][1]
                path = nx.shortest_path(G, source=current, target=target_charger, weight='base_distance')
                return path[1] if len(path) > 1 else num_nodes

        # If at charger and battery < 80%, charge
        if (current in env.charger_nodes or current == env.depot_node) and battery < 85.0:
            return num_nodes  # Action N = Recharge

        # Find closest pending customer target node
        best_target = None
        min_dist = float('inf')

        for order in pending_orders:
            target = order["node"]
            try:
                dist = nx.shortest_path_length(G, source=current, target=target, weight='base_distance')
                if dist < min_dist:
                    min_dist = dist
                    best_target = target
            except nx.NetworkXNoPath:
                pass

        if best_target is not None:
            if best_target == current:
                # Arrived at customer node, move to neighbor or wait
                neighbors = list(G.neighbors(current))
                return neighbors[0] if neighbors else num_nodes
            path = nx.shortest_path(G, source=current, target=best_target, weight='base_distance')
            return path[1] if len(path) > 1 else num_nodes

        # Fallback to random neighbor
        neighbors = list(G.neighbors(current))
        return neighbors[0] if neighbors else num_nodes


class GreedyTSPSolver(BaselineSolver):
    """
    Greedy Priority & Nearest-Neighbor TSP solver.
    Selects highest priority customer first, breaking ties with shortest distance.
    """
    def __init__(self):
        super().__init__("Greedy Priority TSP")

    def select_action(self, env) -> int:
        G = env.G
        current = env.current_node
        orders = env.orders
        battery = env.battery
        num_nodes = env.num_nodes

        pending_orders = [o for o in orders if o["status"] == "pending"]

        if not pending_orders:
            if current == env.depot_node:
                return num_nodes
            try:
                path = nx.shortest_path(G, source=current, target=env.depot_node, weight='base_distance')
                return path[1] if len(path) > 1 else num_nodes
            except nx.NetworkXNoPath:
                return num_nodes

        # Check battery limit
        if battery < 20.0 and current not in env.charger_nodes:
            charger_paths = []
            for c in env.charger_nodes + [env.depot_node]:
                try:
                    length = nx.shortest_path_length(G, source=current, target=c, weight='base_distance')
                    charger_paths.append((length, c))
                except nx.NetworkXNoPath:
                    pass
            if charger_paths:
                charger_paths.sort()
                path = nx.shortest_path(G, source=current, target=charger_paths[0][1], weight='base_distance')
                return path[1] if len(path) > 1 else num_nodes

        if (current in env.charger_nodes or current == env.depot_node) and battery < 80.0:
            return num_nodes

        # Sort pending orders by Priority (descending) then distance (ascending)
        order_candidates = []
        for o in pending_orders:
            t = o["node"]
            try:
                d = nx.shortest_path_length(G, source=current, target=t, weight='base_distance')
                # Rank score: high priority gets lower score (higher precedence)
                score = (-o["priority"] * 100) + d
                order_candidates.append((score, t))
            except nx.NetworkXNoPath:
                pass

        if order_candidates:
            order_candidates.sort()
            best_target = order_candidates[0][1]
            if best_target == current:
                neighbors = list(G.neighbors(current))
                return neighbors[0] if neighbors else num_nodes
            path = nx.shortest_path(G, source=current, target=best_target, weight='base_distance')
            return path[1] if len(path) > 1 else num_nodes

        neighbors = list(G.neighbors(current))
        return neighbors[0] if neighbors else num_nodes


class DynamicTrafficDijkstraSolver(BaselineSolver):
    """
    Traffic-Aware Dynamic Dijkstra Solver. Recalculates edge weights using base_distance * traffic_factor.
    """
    def __init__(self):
        super().__init__("Dynamic Traffic-Aware Dijkstra")

    def _get_dynamic_weight(self, u, v, d):
        return float(d.get('base_distance', 5.0)) * float(d.get('traffic_factor', 1.0))

    def select_action(self, env) -> int:
        G = env.G
        current = env.current_node
        orders = env.orders
        battery = env.battery
        num_nodes = env.num_nodes

        pending_orders = [o for o in orders if o["status"] == "pending"]

        if not pending_orders:
            if current == env.depot_node:
                return num_nodes
            try:
                path = nx.shortest_path(G, source=current, target=env.depot_node, weight=self._get_dynamic_weight)
                return path[1] if len(path) > 1 else num_nodes
            except nx.NetworkXNoPath:
                return num_nodes

        if battery < 25.0 and current not in env.charger_nodes:
            charger_paths = []
            for c in env.charger_nodes + [env.depot_node]:
                try:
                    length = nx.shortest_path_length(G, source=current, target=c, weight=self._get_dynamic_weight)
                    charger_paths.append((length, c))
                except nx.NetworkXNoPath:
                    pass
            if charger_paths:
                charger_paths.sort()
                path = nx.shortest_path(G, source=current, target=charger_paths[0][1], weight=self._get_dynamic_weight)
                return path[1] if len(path) > 1 else num_nodes

        if (current in env.charger_nodes or current == env.depot_node) and battery < 85.0:
            return num_nodes

        best_target = None
        min_cost = float('inf')

        for o in pending_orders:
            t = o["node"]
            try:
                cost = nx.shortest_path_length(G, source=current, target=t, weight=self._get_dynamic_weight)
                # Weighted by order priority
                weighted_cost = cost / (o["priority"] * 0.8)
                if weighted_cost < min_cost:
                    min_cost = weighted_cost
                    best_target = t
            except nx.NetworkXNoPath:
                pass

        if best_target is not None:
            if best_target == current:
                neighbors = list(G.neighbors(current))
                return neighbors[0] if neighbors else num_nodes
            path = nx.shortest_path(G, source=current, target=best_target, weight=self._get_dynamic_weight)
            return path[1] if len(path) > 1 else num_nodes

        neighbors = list(G.neighbors(current))
        return neighbors[0] if neighbors else num_nodes
