# 🗺️ AI-Based Autonomous Delivery Route Optimization Using Reinforcement Learning & Google Maps API

An intelligent, real-world autonomous delivery vehicle routing system built with **Deep Q-Networks (DQN)**, **Google Maps Platform APIs**, **Explainable AI (XAI)**, **PyTorch**, **PyDeck**, **Plotly**, and **Streamlit**.

Unlike traditional route selection systems that depend solely on static shortest-distance algorithms, this system evaluates real-world route options (distance, duration, live traffic congestion, toll fees, delivery priorities) and uses **Reinforcement Learning** to select the optimal route while providing transparent natural-language reasoning explaining *why* a specific route was chosen.

---

## 🌟 Key Features

1. **Google Maps Platform API Integration (`google_maps/api_client.py`)**:
   - Fetches real-world route choices, travel times, live traffic conditions, distances, and toll info via Google Maps Directions & Distance Matrix APIs.
   - Includes an out-of-the-box fallback generator for pre-configured real-world cities (New York, San Francisco, London, Bengaluru) so the application works seamlessly without an API key.
2. **Deep Q-Network (DQN) Routing Environment (`rl_core/route_env.py`)**:
   - `RealWorldRouteEnv(gymnasium.Env)` models multi-objective trade-offs between travel duration, distance, traffic bottlenecks, toll costs, and delivery urgency priorities.
   - 
3. **Explainable AI (XAI) Reasoning Engine (`rl_core/explainability.py`)**:
   - Calculates sub-reward component breakdowns and produces natural-language explanations (e.g. *"Route A was selected because it avoids 18 mins of heavy traffic congestion and saves $4.50 in tolls, despite being 1.4 km longer than Route B"*).
   - 
4. **Interactive Streamlit Web in (`app.py`)**:
   - Real-world map visualizer (PyDeck / Plotly).
   - Candidate route comparison matrix.
   - XAI transparent route reasoning box.
   - Baseline benchmark comparison (DQN vs Shortest Distance vs Fastest Duration vs Lowest Toll Cost).
   - Interactive DQN model training studio.

---

## 📁 Repository Structure

```
c:\Files\Rein_learn/
├── app.py                             # Streamlit interactive web application
├── requirements.txt                   # Dependency list
├── README.md                          # Documentation
├── google_maps/
│   ├── __init__.py
│   └── api_client.py                  # Google Maps API client & fallback router
├── rl_core/
│   ├── __init__.py
│   ├── route_env.py                   # RealWorldRouteEnv Gymnasium environment
│   ├── dqn_agent.py                   # PyTorch Deep Q-Network agent
│   ├── explainability.py              # XAIRouteExplainer natural language engine
│   └── trainer.py                     # Multi-scenario DQN training & evaluation
├── baselines/
│   ├── __init__.py
│   └── solvers.py                     # Classical route selection solvers
└── utils/
    ├── __init__.py
    ├── geo_visualization.py           # PyDeck & Plotly geographic route maps
    └── metrics.py                     # Benchmark aggregator
```
---

## Running the Application

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Streamlit
```bash
python -m streamlit run app.py
```
Open `http://localhost:8501` in your browser.
