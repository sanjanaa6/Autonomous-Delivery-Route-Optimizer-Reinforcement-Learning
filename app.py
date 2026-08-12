import streamlit as st
import numpy as np
import pandas as pd
import torch
import time
import os

# Set Page Configuration
st.set_page_config(
    page_title="AI Delivery Route Optimizer & Driver Assistant",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Core Modules
from google_maps.api_client import GoogleMapsRouteClient, CITY_PRESETS
from rl_core.route_env import RealWorldRouteEnv
from rl_core.dqn_agent import DQNAgent
from rl_core.trainer import train_dqn_route_agent, evaluate_route_policy
from baselines.solvers import ShortestDistanceSolver, FastestDurationSolver, LowestCostSolver
from utils.geo_visualization import (
    render_folium_route_map,
    render_plotly_geo_map,
    render_pydeck_route_map,
    plot_subreward_breakdown,
    HAS_FOLIUM
)
from utils.metrics import run_route_benchmarks

try:
    from streamlit_folium import st_folium
    HAS_ST_FOLIUM = True
except ImportError:
    HAS_ST_FOLIUM = False


# Glassmorphism Styling
st.markdown("""
<style>
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    
    .driver-recommend-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(15, 23, 42, 0.95));
        border: 2px solid #10B981;
        border-radius: 14px;
        padding: 24px;
        box-shadow: 0 10px 40px rgba(16, 185, 129, 0.25);
        margin-bottom: 24px;
    }
    
    .recommend-badge {
        background-color: #10B981;
        color: #064E3B;
        font-weight: 800;
        font-size: 0.9rem;
        padding: 6px 14px;
        border-radius: 20px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        display: inline-block;
        margin-bottom: 12px;
    }
    
    .recommend-title {
        font-size: 1.8rem;
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 8px;
    }
    
    .location-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 12px 16px;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    .xai-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 12px;
        padding: 18px;
        margin-top: 14px;
    }

    .route-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }
    
    .route-card-selected {
        background: rgba(16, 185, 129, 0.15);
        border: 2px solid #10B981;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# Initialize Session State Safely
if 'gmaps_key' not in st.session_state:
    st.session_state.gmaps_key = os.getenv("GOOGLE_MAPS_API_KEY", "")

if 'gmaps_client' not in st.session_state:
    st.session_state.gmaps_client = GoogleMapsRouteClient(st.session_state.gmaps_key)

if 'source_input' not in st.session_state:
    st.session_state.source_input = "Times Square, New York, NY"

if 'dest_input' not in st.session_state:
    st.session_state.dest_input = "Financial District, New York, NY"

if 'vehicle_type' not in st.session_state:
    st.session_state.vehicle_type = "Delivery Van"

if 'train_history' not in st.session_state:
    st.session_state.train_history = None

if 'user_selected_route_override' not in st.session_state:
    st.session_state.user_selected_route_override = None

if 'route_scenario' not in st.session_state:
    st.session_state.route_scenario = st.session_state.gmaps_client.fetch_routes(
        st.session_state.source_input,
        st.session_state.dest_input,
        st.session_state.vehicle_type
    )

if 'route_env' not in st.session_state:
    st.session_state.route_env = RealWorldRouteEnv(st.session_state.route_scenario)

if 'dqn_agent' not in st.session_state:
    obs_dim = st.session_state.route_env.observation_space.shape[0]
    action_dim = st.session_state.route_env.action_space.n
    st.session_state.dqn_agent = DQNAgent(obs_dim, action_dim)
    train_dqn_route_agent(st.session_state.route_env, num_episodes=60)


# ==========================================
# SIDEBAR SETTINGS
# ==========================================
st.sidebar.title("⚙️ System Configurations")

st.sidebar.subheader("🔑 Google Maps API Key")
api_key_field = st.sidebar.text_input("API Key (Optional)", value=st.session_state.gmaps_key, type="password", help="Enter your live Google Maps Platform API key to fetch real-world traffic & directions.")
if api_key_field != st.session_state.gmaps_key:
    st.session_state.gmaps_key = api_key_field
    st.session_state.gmaps_client = GoogleMapsRouteClient(api_key_field)
    st.sidebar.success("Updated Google Maps Client!")

st.sidebar.markdown("---")
st.sidebar.subheader("🏙️ Quick Location Presets")
preset_choice = st.sidebar.selectbox("Select Preset City Scenario:", list(CITY_PRESETS.keys()))

if st.sidebar.button("Apply Preset Location", width="stretch"):
    p_data = CITY_PRESETS[preset_choice]
    st.session_state.source_input = p_data["origin_name"]
    st.session_state.dest_input = p_data["dest_name"]
    scen = st.session_state.gmaps_client.fetch_routes(st.session_state.source_input, st.session_state.dest_input, st.session_state.vehicle_type)
    st.session_state.route_scenario = scen
    st.session_state.route_env.set_scenario(scen)
    st.session_state.user_selected_route_override = None
    train_dqn_route_agent(st.session_state.route_env, num_episodes=60)
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Precise OpenStreetMap & Google Maps Integration**: Uses Geopy Nominatim & OpenStreetMap OSRM driving services for high-accuracy route bounds and street paths. "
    "Queries live Google Maps API directly when API Key is provided."
)


# ==========================================
# MAIN PAGE ROUTE DISCOVERY & INPUTS
# ==========================================
st.title("🚚 AI Delivery Route Optimizer & Driver Assistant")
st.markdown("Enter Source & Destination locations below to discover accurate real candidate routes and obtain the **AI Recommended Route for Delivery Executive**.")

st.markdown("---")

st.markdown("### 📍 Real-World Route Discovery Console")

c1, c2, c3 = st.columns([2.2, 2.2, 1.5])
with c1:
    src_val = st.text_input("🟢 Source / Departure Address", value=st.session_state.source_input, key="src_field")
with c2:
    dst_val = st.text_input("🔴 Destination / Delivery Address", value=st.session_state.dest_input, key="dst_field")
with c3:
    vehicle_val = st.selectbox("🛵 Vehicle Type", ["Delivery Van", "Motorbike", "Cargo Truck"], index=0)

c4, c5, c6, c7 = st.columns(4)
with c4:
    priority_val = st.select_slider("⚡ Delivery Urgency", options=[1, 2, 3], value=2, format_func=lambda x: {1: "🔵 Standard", 2: "🟠 Express", 3: "🔴 Rush Hour Emergency"}[x])
with c5:
    payload_val = st.number_input("📦 Package Weight (kg)", min_value=1.0, max_value=200.0, value=25.0, step=5.0)
with c6:
    toll_budget_val = st.slider("💵 Max Toll Budget ($)", min_value=0.0, max_value=25.0, value=10.0, step=1.0)
with c7:
    st.write("")
    st.write("")
    discover_btn = st.button("🔍 Discover & Optimize Route", type="primary", width="stretch")

if discover_btn:
    st.session_state.source_input = src_val
    st.session_state.dest_input = dst_val
    st.session_state.vehicle_type = vehicle_val
    
    with st.spinner("Geocoding real locations & querying live driving routes..."):
        scen = st.session_state.gmaps_client.fetch_routes(src_val, dst_val, vehicle_val)
        st.session_state.route_scenario = scen
        st.session_state.route_env.set_scenario(scen, priority=priority_val, toll_budget=toll_budget_val)
        st.session_state.user_selected_route_override = None
        agent, _ = train_dqn_route_agent(st.session_state.route_env, num_episodes=60)
        st.session_state.dqn_agent = agent
        st.session_state.route_env.set_scenario(scen, priority=priority_val, toll_budget=toll_budget_val)
    st.success("Real Routes Discovered & Optimized!")


# ==========================================
# EVALUATE AI ROUTE SELECTION
# ==========================================
scen = st.session_state.route_scenario
routes = scen.get("routes", [])

eval_res = evaluate_route_policy(st.session_state.route_env, st.session_state.dqn_agent, solver_type="dqn")

selected_idx = st.session_state.user_selected_route_override if st.session_state.user_selected_route_override is not None else eval_res["selected_route_idx"]
chosen_route = routes[selected_idx] if selected_idx < len(routes) else routes[0]
xai_info = eval_res["explanation"]
breakdown = eval_res["all_breakdowns"].get(selected_idx, {})

# Display Location Resolution Info
orig_c = scen.get("origin_coords", (40.7580, -73.9855))
dest_c = scen.get("dest_coords", (40.7075, -74.0089))

st.markdown(f"""
<div class="location-card">
    <b>📍 Resolved Geocoded Locations:</b><br>
    🟢 <b>Origin:</b> {scen.get('origin_name')} <i>({orig_c[0]:.4f}, {orig_c[1]:.4f})</i><br>
    🔴 <b>Destination:</b> {scen.get('dest_name')} <i>({dest_c[0]:.4f}, {dest_c[1]:.4f})</i>
</div>
""", unsafe_allow_html=True)


# ==========================================
# HERO DRIVER RECOMMENDATION BANNER
# ==========================================
st.markdown(f"""
<div class="driver-recommend-card">
    <div class="recommend-badge">RECOMMENDED ROUTE FOR DELIVERY EXECUTIVE</div>
    <div class="recommend-title">🚚 Adopt {chosen_route['name']}</div>
    <p style="font-size:1.1rem; color:#CBD5E1;">
        Optimal real-world driving corridor for <b>{scen.get('origin_name')}</b> ➡️ <b>{scen.get('dest_name')}</b>.
    </p>
</div>
""", unsafe_allow_html=True)

col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
with col_m1:
    st.metric("Estimated Time", f"{chosen_route['duration_min']:.1f} mins")
with col_m2:
    st.metric("Travel Distance", f"{chosen_route['distance_km']:.1f} km")
with col_m3:
    st.metric("Traffic Congestion", f"{chosen_route['traffic_factor']:.2f}x")
with col_m4:
    st.metric("Toll Charges", f"${chosen_route['toll_cost']:.2f}")
with col_m5:
    st.metric("Efficiency Score", f"{eval_res['all_rewards'].get(selected_idx, 0.0):.1f}")


# ==========================================
# TABS INTERFACE
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Real Interactive Map & Driver XAI", 
    "📊 Algorithm Benchmark Comparison", 
    "🏋️ DQN Agent Training Studio", 
    "📄 System Abstract & Specifications"
])


# ----------------------------------------------------
# TAB 1: REAL INTERACTIVE MAP & DRIVER XAI
# ----------------------------------------------------
with tab1:
    col_map_box, col_xai_box = st.columns([1.6, 1])

    with col_map_box:
        st.subheader("🗺️ Real Interactive Street Map (Auto-Zoomed Bounding Box)")
        
        map_engine = st.radio("Select Map View Mode:", ["Real Leaflet / OpenStreetMap (Folium)", "Plotly Auto-Zoom Map", "3D PyDeck Map"], horizontal=True)
        
        if map_engine == "Real Leaflet / OpenStreetMap (Folium)" and HAS_FOLIUM and HAS_ST_FOLIUM:
            folium_m = render_folium_route_map(scen, selected_route_idx=selected_idx)
            if folium_m:
                st_folium(folium_m, width=850, height=520, returned_objects=[])
        elif map_engine == "3D PyDeck Map":
            deck = render_pydeck_route_map(scen, selected_route_idx=selected_idx)
            st.pydeck_chart(deck)
        else:
            fig_geo = render_plotly_geo_map(scen, selected_route_idx=selected_idx)
            st.plotly_chart(fig_geo, width="stretch")

    with col_xai_box:
        st.subheader("🧠 Why the Driver Should Adopt This Route")
        
        st.markdown(f"""
        <div class="xai-card">
            <div style="font-weight:700; color:#38BDF8; font-size:1.1rem; margin-bottom:6px;">💡 {xai_info['headline']}</div>
            <div style="color:#E2E8F0; font-size:0.95rem; line-height:1.5;">{xai_info['explanation']}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### ⚖️ Route Trade-Off Breakdown:")
        for pt in xai_info["tradeoff_points"]:
            st.markdown(f"- {pt}")

        fig_breakdown = plot_subreward_breakdown(breakdown, title="RL Multi-Objective Reward Weights")
        st.plotly_chart(fig_breakdown, width="stretch")

    st.markdown("---")
    st.subheader("📋 Candidate Routes Matrix & Driver Adoption Toggle")

    card_cols = st.columns(len(routes))
    for idx, r in enumerate(routes):
        is_chosen = (idx == selected_idx)
        card_style = "route-card-selected" if is_chosen else "route-card"

        with card_cols[idx]:
            st.markdown(f"""
            <div class="{card_style}">
                <h4>{'⭐ ADOPTED ' if is_chosen else ''}{r['name']}</h4>
                <p><b>Distance:</b> {r['distance_km']} km</p>
                <p><b>Travel Time:</b> {r['duration_min']} mins</p>
                <p><b>Traffic Factor:</b> {r['traffic_factor']:.2f}x</p>
                <p><b>Tolls:</b> ${r['toll_cost']:.2f}</p>
                <p><b>Reward Score:</b> <span style="color:{'#10B981' if is_chosen else '#38BDF8'}; font-weight:bold;">{eval_res['all_rewards'].get(idx, 0.0):.1f}</span></p>
            </div>
            """, unsafe_allow_html=True)
            
            if not is_chosen:
                if st.button(f"Select {r['name']}", key=f"btn_adopt_{idx}", width="stretch"):
                    st.session_state.user_selected_route_override = idx
                    st.rerun()


# ----------------------------------------------------
# TAB 2: ALGORITHM BENCHMARK
# ----------------------------------------------------
with tab2:
    st.header("📊 Benchmark: Deep Q-Network vs Traditional Solvers")
    st.markdown("Empirical comparison of the **DQN Agent** against classical route selection heuristics on the active Google Maps options.")

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

    st.dataframe(df_bench, width="stretch", hide_index=True)


# ----------------------------------------------------
# TAB 3: MODEL TRAINING STUDIO
# ----------------------------------------------------
with tab3:
    st.header("🏋️ Deep Q-Network Training Studio")
    col_t1, col_t2 = st.columns([1, 1.5])

    with col_t1:
        eps_input = st.number_input("Episodes", value=150, min_value=20, max_value=1000, step=20)
        lr_input = st.number_input("Learning Rate", value=0.001, format="%.4f")
        gamma_input = st.slider("Gamma", 0.80, 0.99, 0.95, 0.01)
        decay_input = st.slider("Epsilon Decay", 0.90, 0.999, 0.98, 0.005)

        if st.button("🚀 Train Agent", width="stretch"):
            pbar = st.progress(0.0)
            status = st.empty()

            def p_cb(ep, total, rew):
                pbar.progress(ep / total)
                status.markdown(f"**Episode {ep}/{total}** | Latest Reward: `{rew:.1f}`")

            t0 = time.time()
            agent, df_hist = train_dqn_route_agent(st.session_state.route_env, num_episodes=eps_input, lr=lr_input, gamma=gamma_input, epsilon_decay=decay_input, progress_callback=p_cb)
            st.session_state.dqn_agent = agent
            st.session_state.train_history = df_hist
            st.success(f"🎉 Model Trained in {time.time()-t0:.2f} seconds!")

    with col_t2:
        if st.session_state.train_history is not None:
            st.subheader("📈 Training Convergence")
            st.line_chart(st.session_state.train_history.set_index("episode")[["reward", "loss"]])


# ----------------------------------------------------
# TAB 4: ABSTRACT & SYSTEM ARCHITECTURE
# ----------------------------------------------------
with tab4:
    st.header("📄 Project Abstract & System Specifications")
    st.markdown("""
    ### **AI-Based Autonomous Delivery Route Optimization Using Reinforcement Learning and Google Maps API**

    The **AI-Based Autonomous Delivery Route Optimization System Using Reinforcement Learning** is a real-world intelligent routing system designed to identify and recommend the most efficient route for delivery vehicles. The system integrates the **Google Maps Platform APIs** to obtain real-world road information such as possible routes, distance, estimated travel time, traffic conditions, and other route-related parameters. Unlike traditional route-selection systems that primarily depend on predefined shortest-path algorithms, the proposed system uses **Reinforcement Learning (RL)** to learn and select the best route based on multiple dynamic factors.

    In this project, **Deep Q-Network (DQN)** is used as the Reinforcement Learning technique. The RL agent considers the current location, destination, available routes, travel time, distance, traffic conditions, toll information, and delivery priorities as part of its environment state. The agent selects a route as an action and receives a reward based on route efficiency. Lower travel time, shorter distance, reduced traffic, lower cost, and successful delivery result in higher rewards, while delays, congestion, unnecessary distance, or higher costs result in penalties. Through repeated training and simulation, the agent learns a routing policy that can select an efficient route under changing conditions.

    The system also provides an **explanation for the selected route**, allowing users to understand *why* a particular route was recommended. For example, the system may identify that Route A was selected because it has lower estimated travel time and traffic compared with Route B, even if Route B has a slightly shorter distance. A web-based dashboard displays the Google Maps route, alternative routes, RL-selected route, route comparison, reward score, and the factors influencing the final decision.
    """)
