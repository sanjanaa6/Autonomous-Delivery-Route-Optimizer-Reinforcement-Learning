import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
import copy
from typing import Dict, List, Tuple, Any

class DeliveryEnv(gym.Env):
    """
    OpenAI Gymnasium Environment for Autonomous Delivery Route Optimization.
    Simulates real-world conditions including dynamic traffic, delivery deadlines,
    vehicle battery & payload capacity constraints, and charging infrastructure.
    """
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(
        self, 
        G: nx.Graph, 
        scenario_meta: Dict[str, Any],
        max_battery: float = 100.0,
        max_capacity: float = 35.0,
        max_steps: int = 80,
        dynamic_traffic: bool = True
    ):
        super(DeliveryEnv, self).__init__()

        self.initial_G = copy.deepcopy(G)
        self.G = copy.deepcopy(G)
        self.scenario_meta = copy.deepcopy(scenario_meta)
        self.num_nodes = G.number_of_nodes()
        self.depot_node = scenario_meta.get("depot_node", 0)
        self.customer_nodes = scenario_meta.get("customer_nodes", [])
        self.charger_nodes = scenario_meta.get("charger_nodes", [])

        self.max_battery = max_battery
        self.max_capacity = max_capacity
        self.max_steps = max_steps
        self.dynamic_traffic = dynamic_traffic

        self.initial_orders = copy.deepcopy(scenario_meta.get("orders", []))
        self.num_orders = len(self.initial_orders)

        # Action Space: Move to node 0..N-1, or Wait/Recharge (action = num_nodes)
        self.action_space = spaces.Discrete(self.num_nodes + 1)

        # State dimensions:
        # [agent_node_norm, battery_pct_norm, payload_used_norm, current_step_norm, avg_traffic_norm,
        #  order_1_pending, order_1_node_norm, order_1_priority_norm, order_1_time_left_norm, ...]
        obs_dim = 5 + 4 * self.num_orders
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.G = copy.deepcopy(self.initial_G)
        self.orders = copy.deepcopy(self.initial_orders)
        
        self.current_node = self.depot_node
        self.battery = float(self.max_battery)
        self.payload = sum(o["weight"] for o in self.orders)  # Starting loaded payload
        self.current_step = 0
        self.total_distance = 0.0
        self.delivered_count = 0
        self.failed_count = 0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def _get_obs(self) -> np.ndarray:
        # Average traffic factor in network
        traffics = [d.get('traffic_factor', 1.0) for u, v, d in self.G.edges(data=True)]
        avg_traffic = float(np.mean(traffics)) if len(traffics) > 0 else 1.0

        obs = [
            float(self.current_node) / max(1, self.num_nodes - 1),
            float(self.battery) / self.max_battery,
            float(self.payload) / max(1.0, self.max_capacity),
            float(self.current_step) / float(self.max_steps),
            (avg_traffic - 1.0) / 2.0  # normalize [1.0, 3.0] to [0.0, 1.0]
        ]

        for order in self.orders:
            pending_flag = 1.0 if order["status"] == "pending" else 0.0
            node_norm = float(order["node"]) / max(1, self.num_nodes - 1)
            priority_norm = float(order["priority"]) / 3.0
            time_left = max(0, order["deadline"] - self.current_step)
            time_left_norm = float(time_left) / float(self.max_steps)

            obs.extend([pending_flag, node_norm, priority_norm, time_left_norm])

        return np.array(obs, dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        return {
            "current_node": self.current_node,
            "battery": self.battery,
            "payload": self.payload,
            "current_step": self.current_step,
            "total_distance": self.total_distance,
            "delivered_count": self.delivered_count,
            "failed_count": self.failed_count,
            "orders": copy.deepcopy(self.orders)
        }

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        reward = 0.0
        terminated = False
        truncated = False

        # Check if battery is already depleted
        if self.battery <= 0:
            reward -= 40.0
            terminated = True
            return self._get_obs(), reward, terminated, truncated, self._get_info()

        # Action = self.num_nodes represents RECHARGE / WAIT
        if action == self.num_nodes:
            if self.current_node in self.charger_nodes or self.current_node == self.depot_node:
                recharged_amount = self.max_battery - self.battery
                self.battery = self.max_battery
                reward += 10.0 if recharged_amount > 20 else 2.0  # Reward for smart charging
            else:
                # Waiting in non-charging spot consumes a bit of battery
                self.battery = max(0.0, self.battery - 1.0)
                reward -= 2.0
        else:
            target_node = action
            if self.G.has_edge(self.current_node, target_node):
                # Valid movement to adjacent node
                base_dist = float(self.G[self.current_node][target_node].get('base_distance', 5.0))
                traffic_tf = float(self.G[self.current_node][target_node].get('traffic_factor', 1.0))
                
                effective_dist = base_dist * traffic_tf
                self.total_distance += effective_dist
                
                # Energy loss proportional to distance and payload weight
                energy_cost = base_dist * 0.4 * (1.0 + (self.payload / max(1.0, self.max_capacity)))
                self.battery = max(0.0, self.battery - energy_cost)

                self.current_node = target_node

                # Travel cost penalty
                reward -= (effective_dist * 0.5)

                # Check if arrived at a customer location with pending order
                for order in self.orders:
                    if order["status"] == "pending" and order["node"] == self.current_node:
                        order["status"] = "delivered"
                        order["delivery_time"] = self.current_step
                        self.delivered_count += 1
                        self.payload = max(0.0, self.payload - order["weight"])

                        # Calculate reward based on priority & deadline timeliness
                        on_time_bonus = 20.0 if self.current_step <= order["deadline"] else 0.0
                        reward += (40.0 * order["priority"]) + on_time_bonus

            else:
                # Invalid movement penalty (target node not adjacent)
                reward -= 8.0

        # Check for expired deadlines
        for order in self.orders:
            if order["status"] == "pending" and self.current_step > order["deadline"]:
                order["status"] = "failed"
                self.failed_count += 1
                reward -= (15.0 * order["priority"])

        # Dynamic traffic fluctuation per step
        if self.dynamic_traffic:
            for u, v in self.G.edges():
                if np.random.rand() < 0.15:
                    delta = round(float(np.random.uniform(-0.3, 0.3)), 2)
                    new_tf = max(1.0, min(3.0, round(self.G[u][v].get('traffic_factor', 1.0) + delta, 2)))
                    self.G[u][v]['traffic_factor'] = new_tf

        # Check completion conditions
        all_resolved = all(o["status"] in ["delivered", "failed"] for o in self.orders)
        if all_resolved:
            terminated = True
            if self.delivered_count == self.num_orders:
                reward += 60.0  # Perfect delivery run bonus

        if self.battery <= 0:
            reward -= 50.0
            terminated = True

        if self.current_step >= self.max_steps:
            truncated = True

        return self._get_obs(), float(reward), terminated, truncated, self._get_info()
