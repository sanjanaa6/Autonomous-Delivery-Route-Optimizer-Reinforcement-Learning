import os
import random
import hashlib
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

try:
    import googlemaps
    HAS_GOOGLEMAPS_LIB = True
except ImportError:
    HAS_GOOGLEMAPS_LIB = False


CITY_PRESETS = {
    "New York, NY": {
        "origin_name": "Times Square, NYC",
        "origin_coords": (40.7580, -73.9855),
        "dest_name": "Financial District, NYC",
        "dest_coords": (40.7075, -74.0089)
    },
    "San Francisco, CA": {
        "origin_name": "Fisherman's Wharf, SF",
        "origin_coords": (37.8080, -122.4177),
        "dest_name": "Mission District, SF",
        "dest_coords": (37.7599, -122.4148)
    },
    "London, UK": {
        "origin_name": "Heathrow Airport, London",
        "origin_coords": (51.4700, -0.4543),
        "dest_name": "Tower Bridge, London",
        "dest_coords": (51.5055, -0.0754)
    },
    "Bengaluru, IN": {
        "origin_name": "Electronic City, BLR",
        "origin_coords": (12.8399, 77.6770),
        "dest_name": "Manyata Tech Park, BLR",
        "dest_coords": (13.0454, 77.6200)
    }
}


class GoogleMapsRouteClient:
    """
    Client for fetching live real-world route options from Google Maps Platform APIs.
    Includes a dynamic fallback engine capable of generating realistic route options
    for ANY custom user-entered Source and Destination address strings.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_MAPS_API_KEY")
        self.gmaps = None

        if self.api_key and HAS_GOOGLEMAPS_LIB:
            try:
                self.gmaps = googlemaps.Client(key=self.api_key)
            except Exception:
                self.gmaps = None

    def fetch_routes(
        self,
        origin_input: str = "Times Square, NYC",
        dest_input: str = "Financial District, NYC",
        vehicle_type: str = "Delivery Van"
    ) -> Dict[str, Any]:
        """
        Fetches route options (distance, time, live traffic congestion, tolls, polyline coordinates)
        either directly from live Google Maps API or generates realistic dynamic route options
        for any custom address query.
        """
        if self.gmaps and origin_input and dest_input:
            try:
                directions_result = self.gmaps.directions(
                    origin_input,
                    dest_input,
                    mode="driving",
                    alternatives=True,
                    departure_time="now"
                )

                if directions_result:
                    return self._parse_google_directions(directions_result, origin_input, dest_input)
            except Exception as e:
                print(f"Google Maps API call error ({e}). Using dynamic route synthesizer.")

        # Fallback Dynamic Synthesizer for custom address queries
        return self._generate_dynamic_custom_routes(origin_input, dest_input, vehicle_type)

    def _parse_google_directions(self, directions_result: List[Dict], origin: str, dest: str) -> Dict[str, Any]:
        routes = []
        for idx, route in enumerate(directions_result):
            leg = route['legs'][0]
            dist_km = float(leg['distance']['value']) / 1000.0
            
            if 'duration_in_traffic' in leg:
                duration_min = float(leg['duration_in_traffic']['value']) / 60.0
                base_duration = float(leg['duration']['value']) / 60.0
                traffic_tf = max(1.0, round(duration_min / max(1.0, base_duration), 2))
            else:
                duration_min = float(leg['duration']['value']) / 60.0
                traffic_tf = round(float(np.random.uniform(1.1, 1.4)), 2)

            path_coords = self._decode_polyline(route['overview_polyline']['points'])
            has_tolls = "toll" in route.get("warnings", []) or "Toll" in route.get("summary", "")
            toll_cost = 6.0 if has_tolls else 0.0

            routes.append({
                "id": idx,
                "name": f"Route {chr(65+idx)} ({route.get('summary', 'Main Corridor')})",
                "distance_km": round(dist_km, 2),
                "duration_min": round(duration_min, 1),
                "traffic_factor": traffic_tf,
                "toll_cost": toll_cost,
                "highway_pct": 0.7,
                "path_coords": path_coords
            })

        return {
            "origin_name": origin,
            "origin_coords": routes[0]["path_coords"][0] if routes else (40.7580, -73.9855),
            "dest_name": dest,
            "dest_coords": routes[0]["path_coords"][-1] if routes else (40.7075, -74.0089),
            "routes": routes
        }

    def _generate_dynamic_custom_routes(self, origin: str, dest: str, vehicle_type: str) -> Dict[str, Any]:
        """
        Generates realistic candidate route options with spatial polylines and traffic
        for any custom origin/destination strings.
        """
        # Deterministic seed based on origin & destination strings for consistent results
        seed_str = f"{origin.strip().lower()}_{dest.strip().lower()}"
        seed_val = int(hashlib.md5(seed_str.encode('utf-8')).hexdigest(), 16) % (2**32 - 1)
        rng = np.random.RandomState(seed_val)

        # Estimate realistic base distance between 6 km and 35 km
        base_dist = round(float(rng.uniform(7.5, 24.0)), 2)
        base_speed = 35.0 if vehicle_type in ["Motorbike", "Delivery Van"] else 28.0
        base_time = round((base_dist / base_speed) * 60.0, 1)

        # Synthetic center coordinates (defaulting to NYC region if not preset)
        preset_match = None
        for name, data in CITY_PRESETS.items():
            if name.lower() in origin.lower() or name.lower() in dest.lower():
                preset_match = data
                break

        if preset_match:
            o_lat, o_lng = preset_match["origin_coords"]
            d_lat, d_lng = preset_match["dest_coords"]
        else:
            o_lat, o_lng = 40.7580 + float(rng.uniform(-0.04, 0.04)), -73.9855 + float(rng.uniform(-0.04, 0.04))
            d_lat, d_lng = 40.7075 + float(rng.uniform(-0.04, 0.04)), -74.0089 + float(rng.uniform(-0.04, 0.04))

        # Candidate Route A: Express Corridor (Slightly longer km, lower traffic, optional toll)
        path_a = self._interpolate_path((o_lat, o_lng), (d_lat, d_lng), arc_curve=0.015, rng=rng)
        route_a = {
            "id": 0,
            "name": f"Route A (Express Bypass)",
            "distance_km": round(base_dist * 1.12, 1),
            "duration_min": round(base_time * 0.85, 1),
            "traffic_factor": round(float(rng.uniform(1.10, 1.25)), 2),
            "toll_cost": 4.50 if float(rng.rand()) > 0.4 else 0.0,
            "highway_pct": 0.85,
            "path_coords": path_a
        }

        # Candidate Route B: Direct Arterial Road (Shortest km, heavy urban traffic gridlock)
        path_b = self._interpolate_path((o_lat, o_lng), (d_lat, d_lng), arc_curve=-0.005, rng=rng)
        route_b = {
            "id": 1,
            "name": f"Route B (Direct City Center)",
            "distance_km": base_dist,
            "duration_min": round(base_time * 1.45, 1),
            "traffic_factor": round(float(rng.uniform(1.85, 2.40)), 2),
            "toll_cost": 0.0,
            "highway_pct": 0.30,
            "path_coords": path_b
        }

        # Candidate Route C: Scenic Outer Ring (Longer km, smooth flow, low toll)
        path_c = self._interpolate_path((o_lat, o_lng), (d_lat, d_lng), arc_curve=-0.025, rng=rng)
        route_c = {
            "id": 2,
            "name": f"Route C (Outer Ring Road)",
            "distance_km": round(base_dist * 1.25, 1),
            "duration_min": round(base_time * 1.05, 1),
            "traffic_factor": round(float(rng.uniform(1.15, 1.35)), 2),
            "toll_cost": 2.00 if float(rng.rand()) > 0.5 else 0.0,
            "highway_pct": 0.65,
            "path_coords": path_c
        }

        return {
            "origin_name": origin if origin else "Source Location",
            "origin_coords": (o_lat, o_lng),
            "dest_name": dest if dest else "Destination Location",
            "dest_coords": (d_lat, d_lng),
            "routes": [route_a, route_b, route_c]
        }

    def _interpolate_path(self, p1: Tuple[float, float], p2: Tuple[float, float], arc_curve: float, rng) -> List[Tuple[float, float]]:
        """
        Generates smooth curved geographic polylines between two points.
        """
        lats = np.linspace(p1[0], p2[0], num=8)
        lngs = np.linspace(p1[1], p2[1], num=8)

        # Add arc offset
        mid_idx = 4
        coords = []
        for i in range(len(lats)):
            offset_lat = arc_curve * np.sin(np.pi * i / 7.0)
            offset_lng = arc_curve * np.sin(np.pi * i / 7.0)
            coords.append((float(lats[i] + offset_lat), float(lngs[i] + offset_lng)))
        return coords

    def _decode_polyline(self, polyline_str: str) -> List[Tuple[float, float]]:
        index, lat, lng = 0, 0, 0
        coordinates = []
        changes = {'latitude': 0, 'longitude': 0}

        while index < len(polyline_str):
            for unit in ['latitude', 'longitude']:
                shift, result = 0, 0
                while True:
                    byte = ord(polyline_str[index]) - 63
                    index += 1
                    result |= (byte & 0x1f) << shift
                    shift += 5
                    if not byte >= 0x20:
                        break
                if result & 1:
                    changes[unit] = ~(result >> 1)
                else:
                    changes[unit] = (result >> 1)

            lat += changes['latitude']
            lng += changes['longitude']
            coordinates.append((lat / 100000.0, lng / 100000.0))

        return coordinates
