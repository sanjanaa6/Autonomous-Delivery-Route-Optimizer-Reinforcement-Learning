import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional

def plot_city_graph(
    G: nx.Graph,
    vehicle_node: int = 0,
    orders: Optional[List[Dict[str, Any]]] = None,
    path_history: Optional[List[int]] = None,
    title: str = "City Delivery Road Network & Traffic State"
) -> go.Figure:
    """
    Renders an interactive Plotly graph showing city nodes, traffic congestion levels,
    customer order statuses, and vehicle current location & path trajectory.
    """
    pos = nx.get_node_attributes(G, 'pos')
    node_types = nx.get_node_attributes(G, 'type')

    # Map order statuses to target customer nodes
    order_status_map = {}
    order_priority_map = {}
    if orders:
        for o in orders:
            order_status_map[o["node"]] = o["status"]
            order_priority_map[o["node"]] = o["priority"]

    fig = go.Figure()

    # Draw Edges with traffic color coding
    edge_x = []
    edge_y = []
    
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        tf = data.get('traffic_factor', 1.0)
        
        # Color based on traffic congestion factor
        if tf >= 2.0:
            edge_color = 'rgba(239, 68, 68, 0.85)'   # Red (Heavy Traffic)
            width = 4.0
        elif tf >= 1.5:
            edge_color = 'rgba(245, 158, 11, 0.85)'  # Amber (Moderate Traffic)
            width = 3.0
        else:
            edge_color = 'rgba(59, 130, 246, 0.45)'  # Blue (Normal Flow)
            width = 2.0

        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode='lines',
            line=dict(width=width, color=edge_color),
            hoverinfo='text',
            text=f"Road ({u} - {v}) | Distance: {data.get('base_distance', 0):.1f}km | Traffic Factor: {tf:.2f}x",
            showlegend=False
        ))

    # Draw Vehicle Trajectory Path if provided
    if path_history and len(path_history) > 1:
        path_x = [pos[n][0] for n in path_history]
        path_y = [pos[n][1] for n in path_history]
        fig.add_trace(go.Scatter(
            x=path_x, y=path_y,
            mode='lines+markers',
            line=dict(width=4, color='#10B981', dash='dash'),
            marker=dict(size=8, color='#10B981'),
            name='Vehicle Route Trajectory'
        ))

    # Draw Nodes grouped by type and priority
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_symbol = []
    node_size = []

    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)

        ntype = node_types.get(n, 'junction')
        
        if n == vehicle_node:
            symbol = 'square-dot'
            color = '#8B5CF6'  # Purple for Vehicle Current Location
            size = 24
            label = f"🚘 VEHICLE HERE (Node {n})"
        elif ntype == 'depot':
            symbol = 'square'
            color = '#F59E0B'  # Gold Depot
            size = 20
            label = f"🏭 DEPOT / WAREHOUSE (Node {n})"
        elif ntype == 'recharge':
            symbol = 'diamond'
            color = '#06B6D4'  # Cyan Charger
            size = 18
            label = f"⚡ CHARGING STATION (Node {n})"
        elif ntype == 'customer':
            status = order_status_map.get(n, 'pending')
            prio = order_priority_map.get(n, 1)
            
            if status == 'delivered':
                color = '#10B981'  # Green Delivered
                symbol = 'circle'
                label = f"✅ Customer (Node {n}) - Delivered!"
            elif status == 'failed':
                color = '#6B7280'  # Grey Expired
                symbol = 'x'
                label = f"❌ Customer (Node {n}) - Missed Deadline"
            else:
                symbol = 'circle'
                if prio == 3:
                    color = '#EF4444'  # Red High Priority
                    label = f"🔴 Customer (Node {n}) - HIGH Priority Order"
                elif prio == 2:
                    color = '#F97316'  # Orange Medium Priority
                    label = f"🟠 Customer (Node {n}) - MED Priority Order"
                else:
                    color = '#3B82F6'  # Blue Low Priority
                    label = f"🔵 Customer (Node {n}) - LOW Priority Order"
            size = 18
        else:
            symbol = 'circle-open'
            color = '#9CA3AF'
            size = 12
            label = f"🛣️ Road Junction (Node {n})"

        node_text.append(label)
        node_color.append(color)
        node_symbol.append(symbol)
        node_size.append(size)

    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(
            symbol=node_symbol,
            size=node_size,
            color=node_color,
            line=dict(width=2, color='#1E293B')
        ),
        text=[f"{n}" for n in G.nodes()],
        textposition="top center",
        hoverinfo='text',
        hovertext=node_text,
        name='City Nodes'
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#F8FAFC')),
        paper_bgcolor='#0F172A',
        plot_bgcolor='#0F172A',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(font=dict(color='#F8FAFC'), orientation='h', yanchor='bottom', y=1.02),
        margin=dict(l=20, r=20, t=50, b=20),
        height=550
    )

    return fig


def plot_reward_curves(df_history: pd.DataFrame, algo_name: str = "RL Agent") -> go.Figure:
    """
    Renders episode training reward curve with rolling average trendline.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_history["episode"],
        y=df_history["reward"],
        mode='lines',
        name='Raw Episode Reward',
        line=dict(color='rgba(99, 102, 241, 0.4)', width=1.5)
    ))

    # Rolling mean trendline
    rolling_window = max(5, len(df_history) // 10)
    df_history['rolling_reward'] = df_history['reward'].rolling(window=rolling_window, min_periods=1).mean()

    fig.add_trace(go.Scatter(
        x=df_history["episode"],
        y=df_history["rolling_reward"],
        mode='lines',
        name=f'Moving Average (window={rolling_window})',
        line=dict(color='#818CF8', width=3.0)
    ))

    fig.update_layout(
        title=f"Training Reward Progress - {algo_name}",
        xaxis_title="Episode",
        yaxis_title="Total Episode Reward",
        paper_bgcolor='#0F172A',
        plot_bgcolor='#1E293B',
        font=dict(color='#F8FAFC'),
        legend=dict(font=dict(color='#F8FAFC')),
        margin=dict(l=40, r=40, t=50, b=40),
        height=400
    )

    return fig


def plot_benchmark_comparison(df_benchmark: pd.DataFrame) -> go.Figure:
    """
    Renders multi-bar performance comparison charts across algorithms.
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Completion Rate (%)',
        x=df_benchmark["Algorithm"],
        y=df_benchmark["Completion Rate (%)"],
        marker_color='#10B981',
        text=df_benchmark["Completion Rate (%)"],
        textposition='auto'
    ))

    fig.add_trace(go.Bar(
        name='Mean Reward',
        x=df_benchmark["Algorithm"],
        y=df_benchmark["Mean Reward"],
        marker_color='#6366F1',
        text=df_benchmark["Mean Reward"],
        textposition='auto'
    ))

    fig.add_trace(go.Bar(
        name='Avg Travel Distance (km)',
        x=df_benchmark["Algorithm"],
        y=df_benchmark["Avg Distance (km)"],
        marker_color='#F59E0B',
        text=df_benchmark["Avg Distance (km)"],
        textposition='auto'
    ))

    fig.update_layout(
        barmode='group',
        title="Side-by-Side Algorithm Benchmark Comparison",
        paper_bgcolor='#0F172A',
        plot_bgcolor='#1E293B',
        font=dict(color='#F8FAFC'),
        legend=dict(font=dict(color='#F8FAFC')),
        margin=dict(l=40, r=40, t=50, b=40),
        height=420
    )

    return fig


def plot_radar_chart(df_benchmark: pd.DataFrame) -> go.Figure:
    """
    Renders multi-axis radar chart showing normalized performance profile.
    """
    categories = ['Delivery Rate', 'Reward Score', 'Distance Efficiency', 'Battery Conservation', 'Time Efficiency']

    fig = go.Figure()

    for idx, row in df_benchmark.iterrows():
        # Normalize metrics to scale [0, 100] for radar display
        deliv_score = min(100.0, max(0.0, row["Completion Rate (%)"]))
        reward_score = min(100.0, max(0.0, (row["Mean Reward"] + 100) / 2.5))
        dist_score = min(100.0, max(0.0, 100.0 - (row["Avg Distance (km)"] * 1.5)))
        batt_score = min(100.0, max(0.0, row["Avg Battery Left (%)"]))
        time_score = min(100.0, max(0.0, 100.0 - (row["Avg Steps"] * 1.2)))

        values = [deliv_score, reward_score, dist_score, batt_score, time_score]
        values.append(values[0])  # Close radar loop

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + [categories[0]],
            fill='toself',
            name=row["Algorithm"]
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color='#94A3B8'),
            angularaxis=dict(color='#F8FAFC'),
            bgcolor='#1E293B'
        ),
        paper_bgcolor='#0F172A',
        font=dict(color='#F8FAFC'),
        title="Multi-Dimensional Operational Performance Profile",
        legend=dict(font=dict(color='#F8FAFC')),
        height=450
    )

    return fig
