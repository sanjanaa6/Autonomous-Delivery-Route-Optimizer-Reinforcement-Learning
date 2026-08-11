import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import torch
import time
import os

# Set Page Configuration with Wide Layout and Dark Theme
st.set_page_config(
    page_title="Autonomous Delivery Route Optimizer | RL",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import custom core modules
from utils.map_generator import CityMapGenerator
from rl_core.env import DeliveryEnv
from rl_core.q_learning import QLearningAgent
from rl_core.dqn_agent import DQNAgent
from rl_core.trainer import train_q_learning, train_dqn, evaluate_runner
from baselines.solvers import DijkstraSolver, GreedyTSPSolver, DynamicTrafficDijkstraSolver
from utils.metrics import run_comprehensive_benchmark
from utils.visualization import plot_city_graph, plot_reward_curves, plot_benchmark_comparison, plot_radar_chart

# Custom CSS for Premium Glassmorphism Aesthetics
st.markdown("""
<style>
    /* Dark Glassmorphism Styling */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
        margin-bottom: 12px;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38BDF8;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .status-badge-high {
        background-color: #7F1D1D;
        color: #FCA5A5;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
    }
    
    .status-badge-med {
        background-color: #7C2D12;
        color: #FDBA74;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
    }
    
    .status-badge-low {
        background-color: #1E3A8A;
        color: #93C5FD;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
    }
    
    /* Hide Streamlit Menu details for clean UI */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if 'seed' not in st.session_state:
    st.session_state.seed = 42

if 'map_gen' not in st.session_state:
    st.session_state.map_gen = CityMapGenerator(seed=st.session_state.seed)

if 'city_graph' not in st.session_state:
    G, meta = st.session_state.map_gen.create_grid_city(grid_size=4, num_customers=6, num_chargers=2)
    st.session_state.city_graph = G
    st.session_state.scenario_meta = meta

if 'env' not in st.session_state:
    st.session_state.env = DeliveryEnv(
        st.session_state.city_graph, 
        st.session_state.scenario_meta,
        max_battery=100.0,
        max_capacity=40.0,
        max_steps=80,
        dynamic_traffic=True
    )

if 'q_agent' not in st.session_state:
    st.session_state.q_agent = None

if 'dqn_agent' not in st.session_state:
    st.session_state.dqn_agent = None

if 'q_history' not in st.session_state:
    st.session_state.q_history = None

if 'dqn_history' not in st.session_state:
    st.session_state.dqn_history = None


# ==========================================
# SIDEBAR CONTROLS & CITY SCENARIO CONFIG
# ==========================================
st.sidebar.title("🚚 Delivery Optimizer")
st.sidebar.markdown("---")
st.sidebar.subheader("🌆 City Environment Setup")

grid_size = st.sidebar.slider("Grid Size (NxN)", min_value=3, max_value=6, value=4, step=1)
num_customers = st.sidebar.slider("Customer Delivery Orders", min_value=3, max_value=10, value=6, step=1)
num_chargers = st.sidebar.slider("Charging Stations", min_value=1, max_value=4, value=2, step=1)
high_traffic_prob = st.sidebar.slider("Traffic Congestion Probability", min_value=0.0, max_value=0.8, value=0.3, step=0.05)
seed_val = st.sidebar.number_input("Environment Random Seed", value=42, step=1)

if st.sidebar.button("🔄 Regenerate City Map & Orders", use_container_width=True):
    st.session_state.seed = seed_val
    map_gen = CityMapGenerator(seed=seed_val)
    G, meta = map_gen.create_grid_city(
        grid_size=grid_size, 
        num_customers=num_customers, 
        num_chargers=num_chargers,
        high_traffic_prob=high_traffic_prob
    )
    st.session_state.city_graph = G
    st.session_state.scenario_meta = meta
    st.session_state.env = DeliveryEnv(
        G, meta, max_battery=100.0, max_capacity=40.0, max_steps=80, dynamic_traffic=True
    )
    # Reset trained agents on map change
    st.session_state.q_agent = None
    st.session_state.dqn_agent = None
    st.session_state.q_history = None
    st.session_state.dqn_history = None
    st.success("New City Map and Delivery Orders Generated!")

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Autonomous Delivery Agent** uses Reinforcement Learning (Q-Learning & Deep Q-Networks) "
    "to make smart dynamic routing decisions under changing traffic, battery, and delivery deadlines."
)


# ==========================================
# MAIN APPLICATION INTERFACE
# ==========================================
st.title("🤖 Autonomous Delivery Route Optimizer Using Reinforcement Learning")
st.markdown(
    "An intelligent autonomous delivery vehicle system comparing **Tabular Q-Learning** and **Deep Q-Networks (DQN)** "
    "against classical **Dijkstra's Shortest Path** and **Greedy Priority TSP** solvers under dynamic traffic and operational constraints."
)
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "🌆 City Network Explorer", 
    "🚗 Live Route Simulation", 
    "🏋️ RL Model Training Studio", 
    "📊 Performance Benchmarking"
])


# ----------------------------------------------------
# TAB 1: CITY NETWORK EXPLORER
# ----------------------------------------------------
with tab1:
    st.header("🌆 Urban Road Network & Delivery Orders")
    
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    
    num_nodes = st.session_state.city_graph.number_of_nodes()
    orders = st.session_state.scenario_meta["orders"]
    chargers = st.session_state.scenario_meta["charger_nodes"]
    high_prio_count = sum(1 for o in orders if o["priority"] == 3)
    
    traffics = [d.get('traffic_factor', 1.0) for u, v, d in st.session_state.city_graph.edges(data=True)]
    avg_traffic = np.mean(traffics) if traffics else 1.0

    with col_m1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Nodes</div><div class="metric-value">{num_nodes}</div></div>', unsafe_allow_html=True)
    with col_m2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Active Orders</div><div class="metric-value">{len(orders)}</div></div>', unsafe_allow_html=True)
    with col_m3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">High Priority</div><div class="metric-value">{high_prio_count}</div></div>', unsafe_allow_html=True)
    with col_m4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Charging Hubs</div><div class="metric-value">{len(chargers)}</div></div>', unsafe_allow_html=True)
    with col_m5:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Avg Traffic Congestion</div><div class="metric-value">{avg_traffic:.2f}x</div></div>', unsafe_allow_html=True)

    col_map, col_table = st.columns([1.5, 1])

    with col_map:
        fig_map = plot_city_graph(
            st.session_state.city_graph, 
            vehicle_node=0, 
            orders=orders,
            title="Interactive City Road Network & Edge Traffic Density"
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with col_table:
        st.subheader("📋 Active Delivery Orders Schedule")
        
        df_orders = pd.DataFrame(orders)
        df_orders_display = df_orders[["order_id", "node", "priority", "weight", "deadline"]].copy()
        df_orders_display.columns = ["Order ID", "Target Node", "Priority (1-3)", "Weight (kg)", "Time Window (steps)"]
        
        prio_map = {3: "🔴 High", 2: "🟠 Medium", 1: "🔵 Low"}
        df_orders_display["Priority (1-3)"] = df_orders_display["Priority (1-3)"].map(prio_map)
        
        st.dataframe(df_orders_display, use_container_width=True, hide_index=True)

        st.markdown("""
        **Legend & Node Identifiers:**
        - 🏭 **Gold Square**: Central Dispatch Depot (Node 0)
        - ⚡ **Cyan Diamond**: EV Charging Stations
        - 🔴/🟠/🔵 **Circles**: Delivery Locations (Color = High / Med / Low Priority)
        - 🛣️ **Red Edges**: Heavy Traffic Congestion (> 2.0x travel time)
        - 🛣️ **Blue Edges**: Free Flowing Traffic (1.0x - 1.3x)
        """)


# ----------------------------------------------------
# TAB 2: LIVE ROUTE SIMULATION PLAYBACK
# ----------------------------------------------------
with tab2:
    st.header("🚗 Interactive Route Simulation & Agent Telemetry")
    
    col_sel, col_btn = st.columns([2, 1])
    
    with col_sel:
        solver_choice = st.selectbox(
            "Select Vehicle Routing Algorithm for Simulation:",
            [
                "Tabular Q-Learning Agent",
                "Deep Q-Network (DQN) Agent",
                "Static Dijkstra Shortest Path",
                "Greedy Priority TSP Solver",
                "Dynamic Traffic-Aware Dijkstra"
            ]
        )

    eval_solver = None
    solver_key = "q_learning"

    if solver_choice == "Tabular Q-Learning Agent":
        if st.session_state.q_agent is None:
            st.warning("⚠️ Q-Learning Agent not trained yet. Defaulting to untrained random policy or train in Tab 3.")
            st.session_state.q_agent = QLearningAgent(st.session_state.env.action_space.n)
        eval_solver = st.session_state.q_agent
        solver_key = "q_learning"

    elif solver_choice == "Deep Q-Network (DQN) Agent":
        if st.session_state.dqn_agent is None:
            st.warning("⚠️ PyTorch DQN Agent not trained yet. Defaulting to untrained policy or train in Tab 3.")
            obs_dim = st.session_state.env.observation_space.shape[0]
            action_dim = st.session_state.env.action_space.n
            st.session_state.dqn_agent = DQNAgent(obs_dim, action_dim)
        eval_solver = st.session_state.dqn_agent
        solver_key = "dqn"

    elif solver_choice == "Static Dijkstra Shortest Path":
        eval_solver = DijkstraSolver()
        solver_key = "dijkstra"

    elif solver_choice == "Greedy Priority TSP Solver":
        eval_solver = GreedyTSPSolver()
        solver_key = "greedy_tsp"

    elif solver_choice == "Dynamic Traffic-Aware Dijkstra":
        eval_solver = DynamicTrafficDijkstraSolver()
        solver_key = "dynamic_dijkstra"

    if st.button("▶️ Run Simulation Episode", use_container_width=True):
        res = evaluate_runner(st.session_state.env, eval_solver, solver_type=solver_key, num_episodes=1)
        st.session_state.current_sim_trajectory = res["trajectory"]
        st.session_state.current_sim_summary = res

    if 'current_sim_trajectory' in st.session_state and st.session_state.current_sim_trajectory:
        traj = st.session_state.current_sim_trajectory
        summary = st.session_state.current_sim_summary
        
        st.markdown("---")
        col_t1, col_t2, col_t3, col_t4 = st.columns(4)
        
        with col_t1:
            st.metric("Total Reward", f"{summary['mean_reward']:.1f}")
        with col_t2:
            st.metric("Delivery Completion", f"{summary['completion_rate']:.1f}%")
        with col_t3:
            st.metric("Total Travel Distance", f"{summary['avg_distance']:.1f} km")
        with col_t4:
            st.metric("Battery Remaining", f"{summary['avg_battery_left']:.1f}%")

        step_idx = st.slider("Simulation Step Timeline", min_value=0, max_value=len(traj)-1, value=0, step=1)
        
        state_snap = traj[step_idx]

        col_sim_map, col_sim_info = st.columns([1.5, 1])

        path_so_far = [t["current_node"] for t in traj[:step_idx+1]]

        with col_sim_map:
            fig_sim = plot_city_graph(
                st.session_state.env.G,
                vehicle_node=state_snap["current_node"],
                orders=state_snap["orders"],
                path_history=path_so_far,
                title=f"Step {step_idx}/{len(traj)-1} - Vehicle at Node {state_snap['current_node']}"
            )
            st.plotly_chart(fig_sim, use_container_width=True)

        with col_sim_info:
            st.subheader(f"📊 Vehicle Telemetry (Step {step_idx})")
            
            batt = state_snap["battery"]
            batt_color = "red" if batt < 25 else "green"
            st.write(f"**Battery Level:** {batt:.1f}%")
            st.progress(max(0.0, min(1.0, batt / 100.0)))

            payload = state_snap["payload"]
            max_cap = st.session_state.env.max_capacity
            st.write(f"**Vehicle Payload:** {payload:.1f} kg / {max_cap:.1f} kg")
            st.progress(max(0.0, min(1.0, payload / max_cap)))

            st.write(f"**Delivered Packages:** {state_snap['delivered_count']} / {st.session_state.env.num_orders}")
            st.write(f"**Distance Traveled:** {state_snap['total_distance']:.1f} km")

            st.markdown("#### 📦 Order Statuses Snapshot:")
            df_order_snap = pd.DataFrame(state_snap["orders"])
            df_order_snap = df_order_snap[["order_id", "node", "priority", "status", "delivery_time"]]
            st.dataframe(df_order_snap, use_container_width=True, hide_index=True)


# ----------------------------------------------------
# TAB 3: RL MODEL TRAINING STUDIO
# ----------------------------------------------------
with tab3:
    st.header("🏋️ Reinforcement Learning Agent Training Studio")
    st.markdown("Configure hyperparameters and train RL agents live on the active city environment.")

    col_algo, col_params = st.columns([1, 1.5])

    with col_algo:
        algo_type = st.radio("Select RL Algorithm to Train:", ["Tabular Q-Learning", "PyTorch Deep Q-Network (DQN)"])
        episodes_input = st.number_input("Number of Training Episodes", value=150, min_value=20, max_value=1000, step=20)
        lr_input = st.number_input("Learning Rate (Alpha / LR)", value=0.1 if algo_type == "Tabular Q-Learning" else 0.001, format="%.4f")
        gamma_input = st.slider("Discount Factor (Gamma)", min_value=0.80, max_value=0.99, value=0.95, step=0.01)
        decay_input = st.slider("Epsilon Decay Rate", min_value=0.90, max_value=0.999, value=0.99, step=0.005)

    with col_params:
        st.markdown("### ⚙️ Training Info & Settings")
        if algo_type == "Tabular Q-Learning":
            st.info(
                "**Tabular Q-Learning**: Uses state discretization (node, battery bin, payload bin, pending order statuses) "
                "to update a Q-table mapping state-action values."
            )
        else:
            st.info(
                "**Deep Q-Network (DQN)**: Uses a PyTorch Multi-Layer Perceptron neural network with a Replay Buffer "
                "and Huber Loss to approximate Q-values over continuous observation vectors."
            )

        if st.button(f"🚀 Start Training {algo_type}", use_container_width=True):
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def update_progress(current_ep, total_eps, ep_reward, completion):
                pct = current_ep / total_eps
                progress_bar.progress(pct)
                status_text.markdown(f"**Episode {current_ep}/{total_eps}** | Ep Reward: `{ep_reward:.1f}` | Delivery Completion: `{completion:.1f}%`")

            start_t = time.time()

            if algo_type == "Tabular Q-Learning":
                q_agent, q_df = train_q_learning(
                    st.session_state.env,
                    num_episodes=episodes_input,
                    lr=lr_input,
                    gamma=gamma_input,
                    epsilon_decay=decay_input,
                    progress_callback=update_progress
                )
                st.session_state.q_agent = q_agent
                st.session_state.q_history = q_df
            else:
                dqn_agent, dqn_df = train_dqn(
                    st.session_state.env,
                    num_episodes=episodes_input,
                    lr=lr_input,
                    gamma=gamma_input,
                    epsilon_decay=decay_input,
                    progress_callback=update_progress
                )
                st.session_state.dqn_agent = dqn_agent
                st.session_state.dqn_history = dqn_df

            elapsed = time.time() - start_t
            st.success(f"🎉 Training completed in {elapsed:.2f} seconds!")

    st.markdown("---")

    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        if st.session_state.q_history is not None:
            fig_q = plot_reward_curves(st.session_state.q_history, algo_name="Tabular Q-Learning")
            st.plotly_chart(fig_q, use_container_width=True)

    with col_c2:
        if st.session_state.dqn_history is not None:
            fig_dqn = plot_reward_curves(st.session_state.dqn_history, algo_name="PyTorch Deep Q-Network (DQN)")
            st.plotly_chart(fig_dqn, use_container_width=True)


# ----------------------------------------------------
# TAB 4: PERFORMANCE BENCHMARKING
# ----------------------------------------------------
with tab4:
    st.header("📊 Comprehensive Algorithm Performance Benchmarking")
    st.markdown(
        "Run an empirical evaluation comparing RL agents against classical non-learning routing baselines "
        "across identical test delivery episodes."
    )

    eval_episodes_bench = st.slider("Evaluation Episodes per Algorithm", min_value=5, max_value=50, value=15, step=5)

    if st.button("🔥 Run Side-by-Side Benchmark Evaluation", use_container_width=True):
        with st.spinner("Running benchmark simulations across all routing strategies..."):
            
            # Create baseline solvers
            solvers_dict = {
                "dijkstra": DijkstraSolver(),
                "greedy_tsp": GreedyTSPSolver(),
                "dynamic_dijkstra": DynamicTrafficDijkstraSolver()
            }

            df_bench, bench_raw = run_comprehensive_benchmark(
                st.session_state.env,
                q_agent=st.session_state.q_agent,
                dqn_agent=st.session_state.dqn_agent,
                solvers_dict=solvers_dict,
                num_episodes=eval_episodes_bench
            )

            st.session_state.df_benchmark = df_bench

    if 'df_benchmark' in st.session_state and st.session_state.df_benchmark is not None:
        df_b = st.session_state.df_benchmark
        
        st.subheader("🏆 Summary Performance Table")
        st.dataframe(df_b, use_container_width=True, hide_index=True)

        col_b1, col_b2 = st.columns(2)

        with col_b1:
            fig_bar = plot_benchmark_comparison(df_b)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_b2:
            fig_radar = plot_radar_chart(df_b)
            st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("""
        > [!TIP]
        > **Key Architectural Takeaways & Evaluation Findings**:
        > 1. **Adaptability to Dynamic Traffic**: RL Agents (especially DQN) continuously observe changing traffic congestion multipliers on network edges, choosing routes that avoid bottleneck traffic spikes where static Dijkstra gets delayed.
        > 2. **Constraint-Aware Decision Making**: Unlike static Dijkstra which ignores battery state until critical, RL agents learn proactive charging policies, stopping at charging hubs before complete battery depletion.
        > 3. **Priority & Time Window Balancing**: Greedy TSP prioritizes order urgency but can make inefficient backtrack loops. RL agents balance priority weight against travel distance cost for optimal cumulative throughput.
        """)
