import numpy as np
import pickle
import random
from typing import Dict, Tuple, Any

class QLearningAgent:
    """
    Tabular Q-Learning Agent for Autonomous Route Planning with State Discretization.
    """
    def __init__(
        self,
        action_dim: int,
        lr: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05
    ):
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table dict: state_tuple -> np.ndarray of shape (action_dim,)
        self.q_table: Dict[Tuple, np.ndarray] = {}

    def get_state_key(self, env_info: Dict[str, Any], env_G=None) -> Tuple:
        """
        Discretizes continuous state environment info into a compact hashable tuple.
        """
        current_node = env_info["current_node"]
        battery_bin = int(min(100.0, max(0.0, env_info["battery"])) // 20)  # 0 to 5
        payload_bin = int(min(50.0, max(0.0, env_info["payload"])) // 10)    # 0 to 5
        
        # Tuple of pending order statuses
        pending_tuple = tuple(
            (o["node"], o["priority"]) 
            for o in env_info["orders"] 
            if o["status"] == "pending"
        )
        
        return (current_node, battery_bin, payload_bin, pending_tuple)

    def get_q_values(self, state_key: Tuple) -> np.ndarray:
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_dim, dtype=np.float32)
        return self.q_table[state_key]

    def select_action(self, env, eval_mode: bool = False) -> int:
        state_key = self.get_state_key(env._get_info())
        
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        q_vals = self.get_q_values(state_key)
        return int(np.argmax(q_vals))

    def update(self, state_key: Tuple, action: int, reward: float, next_state_key: Tuple, done: bool):
        q_vals = self.get_q_values(state_key)
        next_q_vals = self.get_q_values(next_state_key)

        target = reward
        if not done:
            target += self.gamma * np.max(next_q_vals)

        # TD update
        q_vals[action] += self.lr * (target - q_vals[action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump({
                "q_table": self.q_table,
                "epsilon": self.epsilon,
                "lr": self.lr,
                "gamma": self.gamma
            }, f)

    def load(self, filepath: str):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data["q_table"]
            self.epsilon = data.get("epsilon", self.epsilon_min)
            self.lr = data.get("lr", self.lr)
            self.gamma = data.get("gamma", self.gamma)
