import streamlit as st
import numpy as np
import pandas as pd
import torch
import time
import os

# Set Page Configuration
st.set_page_config(
    page_title="AI Delivery Route Optimizer | Google Maps & RL",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Modules
from google_maps.api_client import GoogleMapsRouteClient, CITY_PRESETS
from rl_core.route_env import RealWorldRouteEnv
from rl_core.dqn_agent import DQNAgent
from rl_core.trainer import train_dqn_route_agent, evaluate_route_policy
from baselines.solvers import ShortestDistanceSolver, FastestDurationSolver, LowestCostSolver
from utils.geo_visualization import render_plotly_geo_map, render_pydeck_route_map, plot_subreward_breakdown
from utils.metrics import run_route_benchmarks

# Glassmorphism Styling
st.markdown("""
<style>
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    
    .xai-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
        margin-bottom: 20px;
    }
    
    .xai-headline {
        font-size: 1.3rem;
        font-weight: 700;
        color: #38BDF8;
        margin-bottom: 8px;
    }
    
    .xai-body {
        font-size: 1.05rem;
        color: #E2E8F0;
        line-height: 1.6;
    }
    
    .route-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 10px;
    }
    
    .route-card-selected {
        background: rgba(16, 185, 129, 0.15);
        border: 2px solid #10B981;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 10px;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if 'gmaps_key' not in st.session_state:
    st.session_state.gmaps_key = os.getenv("GOOGLE_MAPS_API_KEY", "")

if 'gmaps_client' not in st.session_state:
    st.session_state.gmaps_client = GoogleMapsRouteClient(st.session_state.gmaps_key)

if 'active_preset' not in st.session_state:
    st.session_state.active_preset = "New York, NY"

if 'route_scenario' not in st.session_state:
    st.session_state.route_scenario = st.session_state.gmaps_client.fetch_routes("New York, NY")

if 'route_env' not in st.session_state:
    st.session_state.route_env = RealWorldRouteEnv(st.session_state.route_scenario)

if 'dqn_agent' not in st.session_state:
    obs_dim = st.session_state.route_env.observation_space.shape[0]
    action_dim = st.session_state.route_env.action_space.n
    st.session_state.dqn_agent = DQNAgent(obs_dim, action_dim)
    # Quick pre-train on active scenario so agent is immediately smart
    train_dqn_route_agent(st.session_state.route_env, num_episodes=60)

if 'train_history' not in st.session_state:
    st.session_state.train_history = None


# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.title("🗺️ Google Maps & RL Setup")
st.sidebar.markdown("---")

api_key_input = st.sidebar.text_input("Google Maps API Key (Optional)", value=st.session_state.gmaps_key, type="password")
if api_key_input != st.session_state.gmaps_key:
    st.session_state.gmaps_key = api_key_input
    st.session_state.gmaps_client = GoogleMapsRouteClient(api_key_input)
    st.sidebar.success("Google Maps API Client Updated!")

st.sidebar.subheader("📍 Delivery Location Presets")
city_choice = st.sidebar.selectbox("Select Real-World City Scenario:", list(CITY_PRESETS.keys()), index=0)

use_custom_addr = st.sidebar.checkbox("Use Custom Address Inputs", value=False)
custom_orig = ""
custom_dest = ""

if use_custom_addr:
    custom_orig = st.sidebar.text_input("Origin Address", "Times Square, New York, NY")
    custom_dest = st.sidebar.text_input("Destination Address", "Wall Street, New York, NY")

if st.sidebar.button("🔄 Fetch Real Routes & Update Environment", use_container_width=True):
    st.session_state.active_preset = city_choice
    scen = st.session_state.gmaps_client.fetch_routes(
        city_preset_name=city_choice,
        origin_input=custom_orig,
        dest_input=custom_dest
    )
    st.session_state.route_scenario = scen
    st.session_state.route_env.set_scenario(scen)
    # Re-train agent quickly for new scenario
    train_dqn_route_agent(st.session_state.route_env, num_episodes=60)
    st.success("New Route Candidates Loaded!")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Operational Constraints")
priority_input = st.sidebar.select_slider("Delivery Urgency Priority", options=[1, 2, 3], value=2, format_func=lambda x: {1: "🔵 Low Priority", 2: "🟠 Medium Priority", 3: "🔴 High Urgency"}[x])
toll_budget_input = st.sidebar.slider("Max Toll Budget ($ / INR)", min_value=0.0, max_value=20.0, value=10.0, step=1.0)

st.session_state.route_env.priority = priority_input
st.session_state.route_env.max_toll_budget = toll_budget_input


# ==========================================
# MAIN APPLICATION INTERFACE
# ==========================================
st.title("🤖 AI Autonomous Delivery Route Optimization")
st.markdown(
    "Powered by **Reinforcement Learning (Deep Q-Network - DQN)**, **Google Maps Platform APIs**, and an **Explainable AI (XAI)** transparent reasoning engine."
)
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ AI Route Optimizer & XAI", 
    "📊 Baseline Algorithm Benchmarking", 
    "🏋️ DQN Model Training Studio", 
    "📄 Abstract & System Architecture"
])


# ----------------------------------------------------
# TAB 1: AI ROUTE OPTIMIZER & XAI
# ----------------------------------------------------
with tab1:
    scen = st.session_state.route_scenario
    routes = scen.get("routes", [])

    st.subheader(f"📍 Origin: `{scen.get('origin_name', 'Start')}` ➡️ Destination: `{scen.get('dest_name', 'End')}`")

    # Evaluate active policy
    eval_res = evaluate_route_policy(st.session_state.route_env, st.session_state.dqn_agent, solver_type="dqn")
    selected_idx = eval_res["selected_route_idx"]
    xai_info = eval_res["explanation"]
    breakdown = eval_res["all_breakdowns"].get(selected_idx, {})

    col_map_view, col_xai_view = st.columns([1.6, 1])

    with col_map_view:
        st.subheader("🗺️ Real-World Interactive Route Map")
        fig_geo = render_plotly_geo_map(scen, selected_route_idx=selected_idx)
        st.plotly_chart(fig_geo, use_container_width=True)

    with col_xai_view:
        st.subheader("🧠 Explainable AI (XAI) Transparent Reasoning")
        
        st.markdown(f"""
        <div class="xai-card">
            <div class="xai-headline">💡 {xai_info['headline']}</div>
            <div class="xai-body">{xai_info['explanation']}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### ⚖️ Key Route Trade-Off Comparisons:")
        for pt in xai_info["tradeoff_points"]:
            st.markdown(f"- {pt}")

        fig_breakdown = plot_subreward_breakdown(breakdown, title="RL Reward Sub-Score Breakdown")
        st.plotly_chart(fig_breakdown, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Candidate Routes Evaluation Matrix")

    cols = st.columns(len(routes))
    for idx, r in enumerate(routes):
        is_chosen = (idx == selected_idx)
        card_class = "route-card-selected" if is_chosen else "route-card"
        
        with cols[idx]:
            st.markdown(f"""
            <div class="{card_class}">
                <h4>{'⭐ ' if is_chosen else ''}{r['name']}</h4>
                <p><b>Distance:</b> {r['distance_km']} km</p>
                <p><b>Duration:</b> {r['duration_min']} mins</p>
                <p><b>Traffic Congestion:</b> {r['traffic_factor']:.2f}x</p>
                <p><b>Toll Cost:</b> ${r['toll_cost']:.2f}</p>
                <p><b>RL Reward Score:</b> <span style="color:{'#10B981' if is_chosen else '#38BDF8'}; font-weight:bold;">{eval_res['all_rewards'].get(idx, 0.0):.1f}</span></p>
            </div>
            """, unsafe_allow_html=True)


# ----------------------------------------------------
# TAB 2: BASELINE ALGORITHM BENCHMARKING
# ----------------------------------------------------
with tab2:
    st.header("📊 Benchmark Evaluation: DQN vs Classical Route Solvers")
    st.markdown(
        "Comparing the **DQN Agent** against traditional route selection heuristics "
        "(Shortest Distance, Fastest Duration, Lowest Toll Cost) on the active real-world route choices."
    )

    solvers = {
        "shortest": ShortestDistanceSolver(),
        "fastest": FastestDurationSolver(),
        "lowest_cost": LowestCostSolver()
    }

    df_bench = run_route_benchmarks(
        st.session_state.route_env,
        dqn_agent=st.session_state.dqn_agent,
        solvers_dict=solvers
    )

    st.dataframe(df_bench, use_container_width=True, hide_index=True)

    st.markdown("""
    > [!TIP]
    > **Why Deep Q-Network Outperforms Fixed Baseline Algorithms**:
    > - **Shortest Distance Solver** blindly chooses physical shortest distance, often steering delivery vehicles into heavy urban gridlocks and congestion bottlenecks.
    > - **Fastest Duration Solver** ignores toll costs and delivery urgency priorities, accumulating high operational expenses.
    > - **Lowest Cost Solver** avoids tolls entirely, resulting in massive travel time delays for high-priority packages.
    > - **Deep Q-Network (DQN)** dynamically optimizes across all competing factors simultaneously!
    """)


# ----------------------------------------------------
# TAB 3: DQN MODEL TRAINING STUDIO
# ----------------------------------------------------
with tab3:
    st.header("🏋️ DQN Agent Training Studio")
    st.markdown("Train the Deep Q-Network policy on dynamic multi-scenario traffic fluctuations.")

    col_t1, col_t2 = st.columns([1, 1.5])

    with col_t1:
        episodes_val = st.number_input("Training Episodes", value=150, min_value=20, max_value=1000, step=20)
        lr_val = st.number_input("Learning Rate", value=0.001, format="%.4f")
        gamma_val = st.slider("Discount Factor (Gamma)", 0.80, 0.99, 0.95, 0.01)
        decay_val = st.slider("Epsilon Decay Rate", 0.90, 0.999, 0.98, 0.005)

        if st.button("🚀 Train DQN Policy", use_container_width=True):
            pbar = st.progress(0.0)
            status = st.empty()

            def p_cb(ep, total, rew):
                pbar.progress(ep / total)
                status.markdown(f"**Episode {ep}/{total}** | Latest Reward: `{rew:.1f}`")

            t0 = time.time()
            agent, df_hist = train_dqn_route_agent(
                st.session_state.route_env,
                num_episodes=episodes_val,
                lr=lr_val,
                gamma=gamma_val,
                epsilon_decay=decay_val,
                progress_callback=p_cb
            )
            st.session_state.dqn_agent = agent
            st.session_state.train_history = df_hist
            st.success(f"🎉 Model Trained in {time.time()-t0:.2f} seconds!")

    with col_t2:
        if st.session_state.train_history is not None:
            st.subheader("📈 Training Loss & Reward Convergence")
            st.line_chart(st.session_state.train_history.set_index("episode")[["reward", "loss"]])


# ----------------------------------------------------
# TAB 4: ABSTRACT & ARCHITECTURE
# ----------------------------------------------------
with tab4:
    st.header("📄 Project Abstract & System Specifications")
    
    st.markdown("""
    ### **AI-Based Autonomous Delivery Route Optimization Using Reinforcement Learning and Google Maps API**

    The **AI-Based Autonomous Delivery Route Optimization System Using Reinforcement Learning** is a real-world intelligent routing system designed to identify and recommend the most efficient route for delivery vehicles. The system integrates the **Google Maps Platform APIs** to obtain real-world road information such as possible routes, distance, estimated travel time, traffic conditions, and other route-related parameters. Unlike traditional route-selection systems that primarily depend on predefined shortest-path algorithms, the proposed system uses **Reinforcement Learning (RL)** to learn and select the best route based on multiple dynamic factors.

    In this project, **Deep Q-Network (DQN)** is used as the Reinforcement Learning technique. The RL agent considers the current location, destination, available routes, travel time, distance, traffic conditions, toll information, and delivery priorities as part of its environment state. The agent selects a route as an action and receives a reward based on route efficiency. Lower travel time, shorter distance, reduced traffic, lower cost, and successful delivery result in higher rewards, while delays, congestion, unnecessary distance, or higher costs result in penalties. Through repeated training and simulation, the agent learns a routing policy that can select an efficient route under changing conditions.

    The system also provides an **explanation for the selected route**, allowing users to understand *why* a particular route was recommended. For example, the system may identify that Route A was selected because it has lower estimated travel time and traffic compared with Route B, even if Route B has a slightly shorter distance. A web-based dashboard displays the Google Maps route, alternative routes, RL-selected route, route comparison, reward score, and the factors influencing the final decision.
    """)
