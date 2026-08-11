# Implementation Plan - Autonomous Delivery Route Optimizer Using Reinforcement Learning

This project implements an intelligent autonomous delivery routing system using Reinforcement Learning (Tabular Q-Learning and Deep Q-Networks - DQN) compared against traditional route-planning algorithms (Dijkstra's Shortest Path and Greedy Nearest-Neighbor TSP). The system includes a custom OpenAI Gymnasium environment, dynamic traffic simulation, delivery priority handling, vehicle battery/capacity constraints, and a feature-rich interactive Streamlit dashboard.

---

## User Review Required


> [!IMPORTANT]
> **Tech Stack & Environment Setup**:
> We will use `gymnasium`, `numpy`, `pandas`, `torch`, `networkx`, `plotly`, `streamlit`, and `matplotlib`.
> If `stable-baselines3` is available, we will also provide SB3 DQN wrappers alongside our standalone PyTorch DQN implementation for maximum flexibility.
>
> **Interactive Visualization**:
> The Streamlit dashboard will feature real-time Plotly interactive city graphs, live step-by-step simulation playback, custom training controls, reward metrics history, and side-by-side performance benchmark comparisons.

---

## Proposed System Architecture

### 1. Environment Core (`rl_core/env.py`)
- Custom `DeliveryEnv(gymnasium.Env)` representing an urban delivery network.
- **Nodes & Edges**: Graph structure (warehouses, customer locations, recharging stations) with spatial coordinates and distance matrices.
- **Dynamic Factors**:
  - Dynamic traffic congestion levels updated per timestep or triggered by traffic events (e.g. rush hour, road blockages).
  - Delivery orders with priorities (High, Medium, Low), time windows/deadlines, and package weights.
  - Autonomous Vehicle constraints: payload capacity, battery level, energy consumption rate, charging stations.
- **State Vector**:
  - `[agent_node, battery_pct, remaining_capacity, current_time, order_1_delivered, order_1_deadline, ..., traffic_factors...]`
- **Action Space**:
  - Discrete action space corresponding to target next-node selections or charging/waiting actions.
- **Reward Function**:
  - Positive reward for timely delivery weighted by order priority (`+50 * priority`).
  - Distance & traffic penalty (`- distance * traffic_multiplier`).
  - Energy consumption penalty (`- energy_cost`).
  - Deadline missed penalty (`- 20 * priority`).
  - Battery empty / invalid move penalty (`- 30`).
  - Bonus for completing all deliveries (`+ 100`).

---

### 2. Reinforcement Learning Agents (`rl_core/`)
- **Tabular Q-Learning Agent (`rl_core/q_learning.py`)**:
  - State discretization / hash mapping.
  - Epsilon-greedy exploration schedule with decay.
  - Q-table update rule: $Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]$.
- **Deep Q-Network (DQN) Agent (`rl_core/dqn_agent.py`)**:
  - PyTorch Neural Network architecture (Multi-Layer Perceptron with ReLU activations).
  - Experience Replay Buffer (`deque` based) for decorrelated batch sampling.
  - Target Q-Network with periodic target updates or soft polyak updates ($\tau$).
  - Epsilon-greedy exploration strategy.
  - Loss function: Huber / MSE Loss.
- **Trainer & Pipeline (`rl_core/trainer.py`)**:
  - Functions for training loops, metric logging (episode rewards, delivery completion rates, steps per episode, battery efficiency), model checkpointing (`.pt` and `.npy`), and inference evaluation.

---

### 3. Traditional Baseline Solvers (`baselines/solvers.py`)
- **Dijkstra / A* Shortest Path Solver**:
  - Computes optimal paths based on static shortest distances between pending delivery targets.
- **Greedy Nearest-Neighbor TSP Solver**:
  - Dynamically selects the closest undelivered high-priority customer location based on current traffic-adjusted distance.
- **Dynamic Traffic-Aware Dijkstra**:
  - Re-calculates shortest path dynamically using real-time edge congestion weights.

---

### 4. Utility Modules (`utils/`)
- **`utils/map_generator.py`**:
  - Generates realistic synthetic city graph layouts (Grid city, Hub-and-Spoke, Ring-Radial networks) with randomly generated customer delivery demand, priorities, and traffic hotspots.
- **`utils/metrics.py`**:
  - Calculates comparative metrics: Total Distance Traveled (km), Total Delivery Time (min), Delivery Completion Rate (%), On-Time Delivery Rate (%), Battery Efficiency (km/kWh), Cumulative Reward.
- **`utils/visualization.py`**:
  - Builds Plotly interactive node graphs with edge color encoding for traffic congestion, distinct node icons for Depot/Customers/Charging Stations, and animated/step-by-step route trajectories.

---

### 5. Interactive Streamlit Dashboard (`app.py`)
Multi-tab interactive dashboard:
1. 🌆 **City Network Explorer**: Customize map topologies (Grid, Random Graph, Radial), view depot/customers/charging nodes, traffic density heatmaps, and order priority lists.
2. 🚗 **Live Simulation Playback**: Run pre-trained RL models or baselines interactively step-by-step with play/pause, speed controls, vehicle battery gauge, payload status, and live route mapping.
3. 🏋️ **RL Model Training Studio**: Interactive training controls (Episodes, Learning Rate, Gamma, Epsilon Decay, Network Architecture), real-time reward curve chart, delivery success rate graphs, and save model options.
4. 📊 **Performance Benchmarking**: Comparative radar charts, bar charts, and metric breakdown tables comparing **Q-Learning**, **DQN**, **Static Dijkstra**, and **Greedy TSP** across varying traffic congestion scenarios and deadline strictness.

---

## Verification Plan

### Automated Tests & Code Validation
- Run unit test suite verifying Gymnasium environment specs (reset, step, action_space, observation_space).
- Validate Q-learning and PyTorch DQN training on a small environment to ensure convergence (positive reward gain over episodes).
- Verify baseline algorithms (Dijkstra and Greedy TSP) generate valid feasible paths without infinite loops.

### Manual Verification via Streamlit Dashboard
- Launch Streamlit application locally (`streamlit run app.py`).
- Verify graph visualization renders smoothly.
- Run interactive simulation step-by-step to confirm vehicle movement, order completion, and metric updates.
- Verify model training tab updates charts live without errors.
