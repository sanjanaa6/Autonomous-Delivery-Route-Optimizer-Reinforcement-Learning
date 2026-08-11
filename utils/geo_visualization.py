import pydeck as pdk
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

def render_pydeck_route_map(scenario: Dict[str, Any], selected_route_idx: int = 0) -> pdk.Deck:
    """
    Renders an interactive 3D geographic PyDeck map showing origin/destination pins
    and candidate route polylines with traffic color highlights.
    """
    orig_coords = scenario.get("origin_coords", (40.7580, -73.9855))
    dest_coords = scenario.get("dest_coords", (40.7075, -74.0089))
    routes = scenario.get("routes", [])

    # Center view between origin and destination
    center_lat = (orig_coords[0] + dest_coords[0]) / 2.0
    center_lng = (orig_coords[1] + dest_coords[1]) / 2.0

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lng,
        zoom=11.5,
        pitch=30,
        bearing=0
    )

    layers = []

    # Pin Layer (Origin & Destination)
    pins_data = [
        {"name": f"Origin: {scenario.get('origin_name', 'Start')}", "coordinates": [orig_coords[1], orig_coords[0]], "color": [16, 185, 129, 255], "radius": 150},
        {"name": f"Destination: {scenario.get('dest_name', 'End')}", "coordinates": [dest_coords[1], dest_coords[0]], "color": [239, 68, 68, 255], "radius": 150}
    ]

    pin_layer = pdk.Layer(
        "ScatterplotLayer",
        pins_data,
        get_position="coordinates",
        get_color="color",
        get_radius="radius",
        pickable=True
    )
    layers.append(pin_layer)

    # Path Layers for candidate routes
    path_data = []
    for idx, r in enumerate(routes):
        is_selected = (idx == selected_route_idx)
        path = [[coords[1], coords[0]] for coords in r["path_coords"]]
        
        if is_selected:
            color = [16, 185, 129, 255]  # Emerald Green for RL Chosen Route
            width = 6
        elif idx == 1:
            color = [245, 158, 11, 220]  # Amber for Route B
            width = 3.5
        else:
            color = [59, 130, 246, 220]  # Blue for Route C
            width = 3.5

        path_data.append({
            "name": f"{r['name']} ({r['distance_km']}km, {r['duration_min']}min)",
            "path": path,
            "color": color,
            "width": width
        })

    path_layer = pdk.Layer(
        "PathLayer",
        path_data,
        get_path="path",
        get_color="color",
        get_width="width",
        width_min_pixels=3,
        pickable=True
    )
    layers.append(path_layer)

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip={"text": "{name}"},
        map_style="mapbox://styles/mapbox/dark-v10"
    )
    return deck


def render_plotly_geo_map(scenario: Dict[str, Any], selected_route_idx: int = 0) -> go.Figure:
    """
    Fallback Plotly Geographic map for candidate route paths.
    """
    orig_coords = scenario.get("origin_coords", (40.7580, -73.9855))
    dest_coords = scenario.get("dest_coords", (40.7075, -74.0089))
    routes = scenario.get("routes", [])

    fig = go.Figure()

    # Draw Route Polylines
    for idx, r in enumerate(routes):
        is_selected = (idx == selected_route_idx)
        lats = [c[0] for c in r["path_coords"]]
        lngs = [c[1] for c in r["path_coords"]]

        if is_selected:
            color = '#10B981'
            width = 5
            name_str = f"⭐ {r['name']} [AI SELECTED]"
        elif idx == 1:
            color = '#F59E0B'
            width = 3
            name_str = r['name']
        else:
            color = '#3B82F6'
            width = 3
            name_str = r['name']

        fig.add_trace(go.Scattermap(
            mode="lines",
            lat=lats,
            lon=lngs,
            line=dict(width=width, color=color),
            name=name_str,
            hoverinfo='text',
            text=f"{r['name']}<br>Dist: {r['distance_km']}km | Time: {r['duration_min']}min | Traffic: {r['traffic_factor']}x | Toll: ${r['toll_cost']:.2f}"
        ))

    # Origin & Destination Markers
    fig.add_trace(go.Scattermap(
        mode="markers+text",
        lat=[orig_coords[0], dest_coords[0]],
        lon=[orig_coords[1], dest_coords[1]],
        marker=dict(size=[16, 16], color=['#10B981', '#EF4444']),
        text=["🟢 ORIGIN", "🔴 DESTINATION"],
        textposition="top center",
        name="Endpoints"
    ))

    center_lat = (orig_coords[0] + dest_coords[0]) / 2.0
    center_lng = (orig_coords[1] + dest_coords[1]) / 2.0

    fig.update_layout(
        map=dict(
            style="carto-darkmatter",
            center=dict(lat=center_lat, lon=center_lng),
            zoom=10.5
        ),
        paper_bgcolor='#0F172A',
        font=dict(color='#F8FAFC'),
        margin=dict(l=10, r=10, t=30, b=10),
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(color='#F8FAFC'))
    )

    return fig


def plot_subreward_breakdown(breakdown_dict: Dict[str, float], title: str = "Reward Factor Contribution") -> go.Figure:
    """
    Renders horizontal waterfall / bar chart showing reward component score contributions.
    """
    categories = ["Base Score", "Time Penalty", "Distance Penalty", "Traffic Penalty", "Toll Penalty", "Budget Penalty", "TOTAL REWARD"]
    values = [
        breakdown_dict.get("base_score", 100.0),
        breakdown_dict.get("time_penalty", 0.0),
        breakdown_dict.get("dist_penalty", 0.0),
        breakdown_dict.get("traffic_penalty", 0.0),
        breakdown_dict.get("toll_penalty", 0.0),
        breakdown_dict.get("budget_penalty", 0.0),
        breakdown_dict.get("total_reward", 0.0)
    ]

    colors = ['#38BDF8', '#EF4444', '#F97316', '#F59E0B', '#A855F7', '#EC4899', '#10B981']

    fig = go.Figure(go.Bar(
        x=values,
        y=categories,
        orientation='h',
        marker_color=colors,
        text=[f"{v:+.1f}" for v in values],
        textposition='auto'
    ))

    fig.update_layout(
        title=title,
        paper_bgcolor='#0F172A',
        plot_bgcolor='#1E293B',
        font=dict(color='#F8FAFC'),
        xaxis_title="Reward Score Contribution",
        margin=dict(l=20, r=20, t=40, b=20),
        height=320
    )

    return fig
