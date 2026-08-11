# 🚚 Autonomous Delivery Route Optimizer Using Reinforcement Learning

An intelligent, adaptive autonomous delivery vehicle routing system built with **Python**, **OpenAI Gymnasium**, **PyTorch**, **NetworkX**, **Plotly**, and **Streamlit**.

This system addresses the limitations of traditional fixed route-planning algorithms (such as Dijkstra or Greedy Nearest-Neighbor TSP) by training **Tabular Q-Learning** and **Deep Q-Network (DQN)** agents to learn optimal routing decisions under dynamic real-time traffic conditions, delivery package priorities, tight time window deadlines, payload capacities, and vehicle battery constraints.

---

## 🌟 Key Features

1. **Custom Gymnasium Environment (`DeliveryEnv`)**:
   - Represents complex urban road networks as graphs with warehouses, charging stations, road junctions, and customer delivery points.
   - Simulates dynamic traffic congestion fluctuations, vehicle energy consumption (distance & payload weight dependent), battery recharging, and order priority deadlines.
2. **Reinforcement Learning Core**:
   - **Tabular Q-Learning**: Custom discrete state-hashing Q-table implementation with epsilon-greedy exploration.
   - **PyTorch Deep Q-Network (DQN)**: Multi-Layer Perceptron neural network with Replay Buffer and target Q-network updates.
3. **Classical Baseline Routing Solvers**:
   - **Static Dijkstra's Shortest Path**: Computes paths based on static road distances.
   - **Greedy Priority TSP**: Always targets highest-priority undelivered customer locations.
   - **Dynamic Traffic-Aware Dijkstra**: Recalculates shortest paths using real-time edge congestion multipliers.
4. **Interactive Streamlit Dashboard (`app.py`)**:
   - **City Network Explorer**: Customize map grid size, customer order counts, charging station locations, and view traffic heatmaps.
   - **Live Route Simulation Playback**: Step-by-step playback with vehicle telemetry (battery gauge, payload, order statuses, interactive map trajectory).
   - **RL Training Studio**: Hyperparameter tuning controls, live training progress bars, and real-time reward curve charts.
   - **Performance Benchmarking**: Side-by-side performance comparison tables, multi-bar charts, and multi-axis radar charts comparing RL agents against classical solvers.

---

## 📁 Repository Structure

```
c:\Files\Rein_learn/
├── app.py                             # Streamlit interactive web dashboard
├── requirements.txt                   # Dependency requirements
├── README.md                          # Documentation & project guide
├── rl_core/
│   ├── __init__.py
│   ├── env.py                         # Gymnasium DeliveryEnv custom environment
│   ├── q_learning.py                  # Tabular Q-Learning agent
│   ├── dqn_agent.py                   # PyTorch Deep Q-Network (DQN) agent
│   └── trainer.py                     # Training routines & evaluation runners
├── baselines/
│   ├── __init__.py
│   └── solvers.py                     # Classical Dijkstra, Greedy TSP & Dynamic solvers
└── utils/
    ├── __init__.py
    ├── map_generator.py               # Synthetic city graph & delivery order scenario generator
    ├── metrics.py                     # Metrics computation & benchmark aggregator
    └── visualization.py               # Plotly interactive graphs, heatmaps & radar charts
```

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📊 Performance Comparison Summary

| Algorithm | Delivery Completion Rate (%) | Dynamic Traffic Adaptability | Proactive Battery Management | Priority Handling |
| :--- | :---: | :---: | :---: | :---: |
| **Deep Q-Network (DQN)** | **High (~90-100%)** | ✅ Dynamic Avoidance | ✅ Learned Recharge Policy | ✅ Balanced |
| **Tabular Q-Learning** | **High (~85-95%)** | ✅ Adaptive | ✅ Learned Recharge Policy | ✅ Balanced |
| **Static Dijkstra** | Medium (~60-75%) | ❌ Blind to Congestion | ⚠️ Reactive | ❌ Nearest Only |
| **Greedy Priority TSP** | Medium (~65-80%) | ❌ High Backtrack Cost | ⚠️ Reactive | ✅ Strict Priority |
| **Dynamic Dijkstra** | High (~80-90%) | ✅ Recalculated | ⚠️ Reactive | ⚠️ Heuristic |

---

## 🔬 Reward Function Formulation

$$R = R_{\text{delivery}} - P_{\text{travel}} - P_{\text{energy}} - P_{\text{deadline}} - P_{\text{invalid}}$$

- **Delivery Reward ($R_{\text{delivery}}$)**: $+40 \times \text{priority} + \text{timeliness bonus}$
- **Travel Cost ($P_{\text{travel}}$)**: $- \text{distance} \times \text{traffic\_factor} \times 0.5$
- **Energy Cost ($P_{\text{energy}}$)**: $- d \times 0.4 \times \left(1 + \frac{\text{payload}}{\text{capacity}}\right)$
- **Deadline Penalty ($P_{\text{deadline}}$)**: $- 15 \times \text{priority}$ for expired pending orders.
- **Battery Depleted / Invalid Action Penalty**: $-50$ / $-8$.
