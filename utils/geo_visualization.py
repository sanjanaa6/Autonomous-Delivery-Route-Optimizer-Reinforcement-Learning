import pydeck as pdk
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

try:
    import folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False


def render_folium_route_map(scenario: Dict[str, Any], selected_route_idx: int = 0) -> Optional[Any]:
    """
    Renders a real interactive Leaflet/OpenStreetMap Folium map with street tiles,
    precise auto-zoomed bounding boxes (`fit_bounds`), origin/destination pins,
    and route polylines.
    """
    if not HAS_FOLIUM:
        return None

    orig_coords = scenario.get("origin_coords", (40.7580, -73.9855))
    dest_coords = scenario.get("dest_coords", (40.7075, -74.0089))
    routes = scenario.get("routes", [])

    # Collect all coordinate points across all routes to calculate tight map bounds
    all_lats = [orig_coords[0], dest_coords[0]]
    all_lngs = [orig_coords[1], dest_coords[1]]

    for r in routes:
        for lat, lng in r.get("path_coords", []):
            all_lats.append(lat)
            all_lngs.append(lng)

    min_lat, max_lat = min(all_lats), max(all_lats)
    min_lng, max_lng = min(all_lngs), max(all_lngs)

    center_lat = (min_lat + max_lat) / 2.0
    center_lng = (min_lng + max_lng) / 2.0

    # Initialize Folium Map
    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=13,
        tiles="OpenStreetMap",
        control_scale=True
    )

    # Fit map bounds tightly around all route points with padding
    bounds = [[min_lat - 0.005, min_lng - 0.005], [max_lat + 0.005, max_lng + 0.005]]
    m.fit_bounds(bounds, padding=(30, 30))

    # Add Start Green Marker
    folium.Marker(
        location=[orig_coords[0], orig_coords[1]],
        popup=f"🟢 <b>DEPARTURE:</b> {scenario.get('origin_name', 'Departure')}",
        tooltip="Origin / Departure Point",
        icon=folium.Icon(color="green", icon="play", prefix="fa")
    ).add_to(m)

    # Add End Red Marker
    folium.Marker(
        location=[dest_coords[0], dest_coords[1]],
        popup=f"🔴 <b>DESTINATION:</b> {scenario.get('dest_name', 'Delivery Target')}",
        tooltip="Destination / Delivery Target",
        icon=folium.Icon(color="red", icon="flag", prefix="fa")
    ).add_to(m)

    # Add Candidate Route Polylines
    for idx, r in enumerate(routes):
        is_selected = (idx == selected_route_idx)
        coords = r["path_coords"]

        if is_selected:
            color = "#10B981"  # Emerald Green for AI Adopted Route
            weight = 7
            opacity = 0.95
            popup_txt = f"⭐ <b>AI ADOPTED ROUTE: {r['name']}</b><br>Distance: {r['distance_km']} km<br>Travel Time: {r['duration_min']} mins<br>Traffic Congestion: {r['traffic_factor']}x<br>Tolls: ₹{r['toll_cost']:.2f}"
        elif idx == 1:
            color = "#F59E0B"  # Amber for Route B
            weight = 4.5
            opacity = 0.8
            popup_txt = f"<b>{r['name']}</b><br>Distance: {r['distance_km']} km<br>Travel Time: {r['duration_min']} mins<br>Traffic Congestion: {r['traffic_factor']}x"
        else:
            color = "#3B82F6"  # Blue for Route C
            weight = 4.5
            opacity = 0.8
            popup_txt = f"<b>{r['name']}</b><br>Distance: {r['distance_km']} km<br>Travel Time: {r['duration_min']} mins<br>Traffic Congestion: {r['traffic_factor']}x"

        folium.PolyLine(
            locations=coords,
            color=color,
            weight=weight,
            opacity=opacity,
            popup=folium.Popup(popup_txt, max_width=320),
            tooltip=r["name"]
        ).add_to(m)

    return m


def render_plotly_geo_map(scenario: Dict[str, Any], selected_route_idx: int = 0) -> go.Figure:
    """
    Plotly Mapbox map with dynamic auto-zoom calculated from coordinate bounds.
    """
    orig_coords = scenario.get("origin_coords", (40.7580, -73.9855))
    dest_coords = scenario.get("dest_coords", (40.7075, -74.0089))
    routes = scenario.get("routes", [])

    all_lats = [orig_coords[0], dest_coords[0]]
    all_lngs = [orig_coords[1], dest_coords[1]]

    for r in routes:
        for lat, lng in r.get("path_coords", []):
            all_lats.append(lat)
            all_lngs.append(lng)

    min_lat, max_lat = min(all_lats), max(all_lats)
    min_lng, max_lng = min(all_lngs), max(all_lngs)

    center_lat = (min_lat + max_lat) / 2.0
    center_lng = (min_lng + max_lng) / 2.0

    lat_span = max(0.005, max_lat - min_lat)
    lng_span = max(0.005, max_lng - min_lng)
    max_span = max(lat_span, lng_span)

    # Dynamic zoom level calculation
    zoom = max(3.0, min(15.5, round(12.5 - np.log2(max_span / 0.05), 1)))

    fig = go.Figure()

    for idx, r in enumerate(routes):
        is_selected = (idx == selected_route_idx)
        lats = [c[0] for c in r["path_coords"]]
        lngs = [c[1] for c in r["path_coords"]]

        if is_selected:
            color = '#10B981'
            width = 6
            name_str = f"⭐ {r['name']} [ADOPTED]"
        elif idx == 1:
            color = '#F59E0B'
            width = 3.5
            name_str = r['name']
        else:
            color = '#3B82F6'
            width = 3.5
            name_str = r['name']

        fig.add_trace(go.Scattermap(
            mode="lines",
            lat=lats,
            lon=lngs,
            line=dict(width=width, color=color),
            name=name_str,
            hoverinfo='text',
            text=f"{r['name']}<br>Dist: {r['distance_km']}km | Time: {r['duration_min']}min | Traffic: {r['traffic_factor']}x | Toll: ₹{r['toll_cost']:.2f}"
        ))

    fig.add_trace(go.Scattermap(
        mode="markers+text",
        lat=[orig_coords[0], dest_coords[0]],
        lon=[orig_coords[1], dest_coords[1]],
        marker=dict(size=[16, 16], color=['#10B981', '#EF4444']),
        text=["🟢 ORIGIN", "🔴 DESTINATION"],
        textposition="top center",
        name="Endpoints"
    ))

    fig.update_layout(
        map=dict(
            style="carto-darkmatter",
            center=dict(lat=center_lat, lon=center_lng),
            zoom=zoom
        ),
        paper_bgcolor='#0F172A',
        font=dict(color='#F8FAFC'),
        margin=dict(l=10, r=10, t=30, b=10),
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(color='#F8FAFC'))
    )

    return fig


def render_pydeck_route_map(scenario: Dict[str, Any], selected_route_idx: int = 0) -> pdk.Deck:
    """
    PyDeck 3D map with dynamic zoom level.
    """
    orig_coords = scenario.get("origin_coords", (40.7580, -73.9855))
    dest_coords = scenario.get("dest_coords", (40.7075, -74.0089))
    routes = scenario.get("routes", [])

    all_lats = [orig_coords[0], dest_coords[0]]
    all_lngs = [orig_coords[1], dest_coords[1]]

    for r in routes:
        for lat, lng in r.get("path_coords", []):
            all_lats.append(lat)
            all_lngs.append(lng)

    min_lat, max_lat = min(all_lats), max(all_lats)
    min_lng, max_lng = min(all_lngs), max(all_lngs)

    center_lat = (min_lat + max_lat) / 2.0
    center_lng = (min_lng + max_lng) / 2.0
    max_span = max(max_lat - min_lat, max_lng - min_lng)
    zoom = max(4.0, min(15.0, round(12.0 - np.log2(max_span / 0.05), 1)))

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lng,
        zoom=zoom,
        pitch=30,
        bearing=0
    )

    layers = []

    pins_data = [
        {"name": f"Origin: {scenario.get('origin_name', 'Start')}", "coordinates": [orig_coords[1], orig_coords[0]], "color": [16, 185, 129, 255], "radius": 180},
        {"name": f"Destination: {scenario.get('dest_name', 'End')}", "coordinates": [dest_coords[1], dest_coords[0]], "color": [239, 68, 68, 255], "radius": 180}
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

    path_data = []
    for idx, r in enumerate(routes):
        is_selected = (idx == selected_route_idx)
        path = [[coords[1], coords[0]] for coords in r["path_coords"]]
        
        if is_selected:
            color = [16, 185, 129, 255]
            width = 6
        elif idx == 1:
            color = [245, 158, 11, 220]
            width = 3.5
        else:
            color = [59, 130, 246, 220]
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


def plot_subreward_breakdown(breakdown_dict: Dict[str, float], title: str = "Reward Factor Contribution") -> go.Figure:
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
