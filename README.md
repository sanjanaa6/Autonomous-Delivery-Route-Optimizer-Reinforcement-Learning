# 🗺️ AI-Based Autonomous Delivery Route Optimization Using Reinforcement Learning & Google Maps API

An intelligent, real-world autonomous delivery vehicle routing system built with **Deep Q-Networks (DQN)**, **Google Maps Platform APIs**, **Explainable AI (XAI)**, **PyTorch**, **PyDeck**, **Plotly**, and **Streamlit**.

Unlike traditional route-selection systems that depend solely on static shortest-distance algorithms, this system evaluates real-world route options—including **distance, duration, live traffic congestion, toll fees, and delivery priorities**—and uses **Reinforcement Learning** to select the optimal route while providing transparent natural-language reasoning explaining *why* a specific route was chosen.

---

## 🌟 Key Features

### 1. Google Maps Platform API Integration

**`google_maps/api_client.py`**

* Fetches real-world route choices.
* Retrieves travel times and distances.
* Considers live traffic conditions.
* Retrieves toll information where available.
* Uses Google Maps Directions and Distance Matrix APIs.
* Includes an out-of-the-box fallback generator for pre-configured real-world cities, allowing the application to work without an API key.

Supported fallback locations include:

* New York
* San Francisco
* London
* Bengaluru

---

### 2. Deep Q-Network (DQN) Routing Environment

**`rl_core/route_env.py`**

The `RealWorldRouteEnv(gymnasium.Env)` models multi-objective route-selection trade-offs, including:

* Travel duration
* Route distance
* Traffic bottlenecks
* Toll costs
* Delivery urgency and priority

The DQN agent learns which route provides the best overall reward rather than simply choosing the shortest or fastest path.

---

### 3. Explainable AI (XAI) Reasoning Engine

**`rl_core/explainability.py`**

The XAI engine:

* Calculates individual sub-reward components.
* Explains the trade-offs considered by the RL agent.
* Generates human-readable natural-language reasoning.
* Makes route decisions transparent and interpretable.

Example explanation:

> *Route A was selected because it avoids 18 minutes of heavy traffic congestion and saves $4.50 in tolls, despite being 1.4 km longer than Route B.*

---

### 4. Interactive Streamlit Web Application

**`app.py`**

The Streamlit interface provides:

* Real-world route visualization using PyDeck and Plotly.
* Candidate route comparison matrix.
* Explainable AI route-reasoning panel.
* Baseline benchmark comparison.
* Interactive DQN model training studio.

Baseline comparisons include:

* DQN-selected route
* Shortest-distance route
* Fastest-duration route
* Lowest-toll-cost route

---

## 🧠 Technologies Used

* **Python**
* **PyTorch**
* **Deep Q-Networks (DQN)**
* **Reinforcement Learning**
* **Gymnasium**
* **Google Maps Platform APIs**
* **Explainable AI (XAI)**
* **Streamlit**
* **PyDeck**
* **Plotly**

---

## 📁 Repository Structure

```text
Rein_learn/
├── app.py                             # Streamlit interactive web application
├── requirements.txt                   # Dependency list
├── README.md                          # Documentation
│
├── google_maps/
│   ├── __init__.py
│   └── api_client.py                  # Google Maps API client & fallback router
│
├── rl_core/
│   ├── __init__.py
│   ├── route_env.py                   # RealWorldRouteEnv Gymnasium environment
│   ├── dqn_agent.py                   # PyTorch Deep Q-Network agent
│   ├── explainability.py              # XAIRouteExplainer natural-language engine
│   └── trainer.py                     # Multi-scenario DQN training & evaluation
│
├── baselines/
│   ├── __init__.py
│   └── solvers.py                     # Classical route-selection solvers
│
└── utils/
    ├── __init__.py
    ├── geo_visualization.py           # PyDeck & Plotly geographic route maps
    └── metrics.py                     # Benchmark aggregator
```

---

## ⚙️ How the System Works

```text
Delivery Request
       ↓
Google Maps API / Fallback Route Generator
       ↓
Candidate Routes
       ↓
Route Feature Extraction
(Distance, Duration, Traffic, Tolls, Priority)
       ↓
DQN Routing Environment
       ↓
Deep Q-Network Agent
       ↓
Optimal Route Selection
       ↓
XAI Reward Analysis
       ↓
Natural-Language Explanation + Map Visualization
```

---

## 🎯 Reinforcement Learning Objective

Instead of optimizing only one factor, the system uses a reward function that considers multiple real-world objectives:

* Minimize travel duration.
* Minimize travel distance.
* Avoid heavy traffic congestion.
* Reduce toll costs.
* Prioritize urgent deliveries.
* Balance conflicting route-selection factors.

This allows the DQN agent to learn a policy that can outperform simple rule-based route-selection strategies.

---

## 📊 Baseline Comparison

The trained DQN agent can be evaluated against traditional route-selection strategies:

| Method                | Optimization Goal                               |
| --------------------- | ----------------------------------------------- |
| **DQN Agent**         | Learns a multi-objective route-selection policy |
| **Shortest Distance** | Minimizes route distance                        |
| **Fastest Duration**  | Minimizes travel time                           |
| **Lowest Toll Cost**  | Minimizes toll expenses                         |

The benchmark system aggregates results to compare the effectiveness of reinforcement learning against classical approaches.

---

## 🔍 Explainable AI

The XAI module breaks down the reward associated with each route-selection decision.

For example, a route may be selected because:

* It has lower traffic congestion.
* It reduces total delivery time.
* It avoids expensive toll roads.
* It better satisfies delivery urgency requirements.

This makes the system more transparent than a conventional black-box AI model.

---

## 🚀 Running the Application

### 1. Clone or Open the Project

Navigate to the project directory:

```bash
cd c:\Files\Rein_learn
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit Application

```bash
python -m streamlit run app.py
```

### 4. Open the Application

Open the following address in your browser:

```text
http://localhost:8501
```

---

## 🔑 Google Maps API Configuration

If you have a Google Maps API key, configure it according to the project's API client requirements.

The system also includes a fallback route generator for supported cities, allowing the application to demonstrate route optimization functionality even without a Google Maps API key.

---

## 📈 Future Improvements

Potential future enhancements include:

* Real-time GPS vehicle tracking
* Dynamic rerouting during delivery
* Multi-vehicle fleet optimization
* Multi-stop delivery planning
* Advanced traffic prediction
* Weather-aware route optimization
* Prioritized emergency deliveries
* More advanced RL algorithms such as PPO or SAC
* Model persistence and checkpointing
* Cloud deployment
* Historical delivery analytics

---

## 📄 License

This project is intended for **educational, research, and development purposes**.

---

## 👩‍💻 Author

Built as an AI and Reinforcement Learning project demonstrating how **Deep Reinforcement Learning, real-world mapping data, and Explainable AI** can be combined for intelligent autonomous delivery route optimization.
