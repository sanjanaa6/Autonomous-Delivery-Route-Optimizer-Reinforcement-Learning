import os
import random
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
        "dest_coords": (40.7075, -74.0089),
        "routes": [
            {
                "name": "Route A (FDR Drive Bypass)",
                "distance_km": 11.2,
                "duration_min": 22.0,
                "traffic_factor": 1.15,  # Moderate/Light
                "toll_cost": 0.0,
                "highway_pct": 0.8,
                "path_coords": [
                    (40.7580, -73.9855), (40.7540, -73.9680), (40.7350, -73.9720),
                    (40.7140, -73.9780), (40.7075, -74.0089)
                ]
            },
            {
                "name": "Route B (West Side Highway)",
                "distance_km": 9.8,
                "duration_min": 31.0,
                "traffic_factor": 1.85,  # Heavy congestion
                "toll_cost": 0.0,
                "highway_pct": 0.6,
                "path_coords": [
                    (40.7580, -73.9855), (40.7600, -73.9990), (40.7300, -74.0100),
                    (40.7100, -74.0150), (40.7075, -74.0089)
                ]
            },
            {
                "name": "Route C (Tunnel & Expressway - Toll)",
                "distance_km": 10.5,
                "duration_min": 19.5,
                "traffic_factor": 1.10,  # Free flow
                "toll_cost": 6.50,
                "highway_pct": 0.9,
                "path_coords": [
                    (40.7580, -73.9855), (40.7480, -73.9750), (40.7250, -73.9800),
                    (40.7100, -74.0000), (40.7075, -74.0089)
                ]
            }
        ]
    },
    "San Francisco, CA": {
        "origin_name": "Fisherman's Wharf, SF",
        "origin_coords": (37.8080, -122.4177),
        "dest_name": "Mission District, SF",
        "dest_coords": (37.7599, -122.4148),
        "routes": [
            {
                "name": "Route A (US-101 South)",
                "distance_km": 8.4,
                "duration_min": 18.0,
                "traffic_factor": 1.20,
                "toll_cost": 0.0,
                "highway_pct": 0.7,
                "path_coords": [
                    (37.8080, -122.4177), (37.7950, -122.4020), (37.7780, -122.4050),
                    (37.7650, -122.4100), (37.7599, -122.4148)
                ]
            },
            {
                "name": "Route B (Van Ness Avenue)",
                "distance_km": 7.1,
                "duration_min": 26.0,
                "traffic_factor": 1.90,  # Construction & Signals
                "toll_cost": 0.0,
                "highway_pct": 0.2,
                "path_coords": [
                    (37.8080, -122.4177), (37.8000, -122.4240), (37.7800, -122.4210),
                    (37.7650, -122.4200), (37.7599, -122.4148)
                ]
            },
            {
                "name": "Route C (Embarcadero & 3rd St)",
                "distance_km": 8.9,
                "duration_min": 21.0,
                "traffic_factor": 1.35,
                "toll_cost": 0.0,
                "highway_pct": 0.4,
                "path_coords": [
                    (37.8080, -122.4177), (37.7980, -122.3960), (37.7750, -122.3920),
                    (37.7620, -122.4050), (37.7599, -122.4148)
                ]
            }
        ]
    },
    "London, UK": {
        "origin_name": "Heathrow Airport, London",
        "origin_coords": (51.4700, -0.4543),
        "dest_name": "Tower Bridge, London",
        "dest_coords": (51.5055, -0.0754),
        "routes": [
            {
                "name": "Route A (M4 & A4 Corridor)",
                "distance_km": 30.5,
                "duration_min": 42.0,
                "traffic_factor": 1.25,
                "toll_cost": 0.0,
                "highway_pct": 0.75,
                "path_coords": [
                    (51.4700, -0.4543), (51.4850, -0.3200), (51.4920, -0.2000),
                    (51.5000, -0.1200), (51.5055, -0.0754)
                ]
            },
            {
                "name": "Route B (A40 & Euston Congestion Zone)",
                "distance_km": 28.1,
                "duration_min": 58.0,
                "traffic_factor": 2.10,  # High inner city traffic
                "toll_cost": 15.00,  # Congestion Charge
                "highway_pct": 0.45,
                "path_coords": [
                    (51.4700, -0.4543), (51.5200, -0.3000), (51.5250, -0.1400),
                    (51.5150, -0.0900), (51.5055, -0.0754)
                ]
            },
            {
                "name": "Route C (South Circular A205)",
                "distance_km": 33.2,
                "duration_min": 49.0,
                "traffic_factor": 1.40,
                "toll_cost": 0.0,
                "highway_pct": 0.50,
                "path_coords": [
                    (51.4700, -0.4543), (51.4500, -0.3000), (51.4400, -0.1500),
                    (51.4800, -0.0800), (51.5055, -0.0754)
                ]
            }
        ]
    },
    "Bengaluru, IN": {
        "origin_name": "Electronic City, BLR",
        "origin_coords": (12.8399, 77.6770),
        "dest_name": "Manyata Tech Park, BLR",
        "dest_coords": (13.0454, 77.6200),
        "routes": [
            {
                "name": "Route A (Outer Ring Road Expressway)",
                "distance_km": 32.4,
                "duration_min": 52.0,
                "traffic_factor": 1.30,
                "toll_cost": 85.0,  # INR Toll
                "highway_pct": 0.70,
                "path_coords": [
                    (12.8399, 77.6770), (12.9200, 77.6850), (12.9800, 77.6950),
                    (13.0300, 77.6400), (13.0454, 77.6200)
                ]
            },
            {
                "name": "Route B (Hosur Rd & Silk Board Junction)",
                "distance_km": 27.8,
                "duration_min": 85.0,
                "traffic_factor": 2.60,  # Extreme bottleneck
                "toll_cost": 0.0,
                "highway_pct": 0.35,
                "path_coords": [
                    (12.8399, 77.6770), (12.9170, 77.6230), (12.9700, 77.6000),
                    (13.0100, 77.6100), (13.0454, 77.6200)
                ]
            },
            {
                "name": "Route C (NICE Expressway & West Bypass)",
                "distance_km": 41.2,
                "duration_min": 58.0,
                "traffic_factor": 1.10,  # Smooth tollway
                "toll_cost": 160.0,
                "highway_pct": 0.90,
                "path_coords": [
                    (12.8399, 77.6770), (12.8600, 77.5300), (12.9600, 77.5100),
                    (13.0500, 77.5600), (13.0454, 77.6200)
                ]
            }
        ]
    }
}


class GoogleMapsRouteClient:
    """
    Client for fetching real-world route options from Google Maps Platform APIs,
    with an offline fallback city router for robust demonstration without API keys.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_MAPS_API_KEY")
        self.gmaps = None

        if self.api_key and HAS_GOOGLEMAPS_LIB:
            try:
                self.gmaps = googlemaps.Client(key=self.api_key)
            except Exception:
                self.gmaps = None

    def fetch_routes(self, city_preset_name: str = "New York, NY", origin_input: str = "", dest_input: str = "") -> Dict[str, Any]:
        """
        Fetches route options (distance, time, traffic, tolls, coordinate path)
        either via Google Maps API or using preset city scenarios.
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
                print(f"Google Maps API call failed ({e}). Falling back to city preset.")

        # Fallback to Preset Cities
        preset = CITY_PRESETS.get(city_preset_name, CITY_PRESETS["New York, NY"])
        return preset

    def _parse_google_directions(self, directions_result: List[Dict], origin: str, dest: str) -> Dict[str, Any]:
        routes = []
        for idx, route in enumerate(directions_result):
            leg = route['legs'][0]
            dist_km = float(leg['distance']['value']) / 1000.0
            
            # Duration in traffic if available, else standard duration
            if 'duration_in_traffic' in leg:
                duration_min = float(leg['duration_in_traffic']['value']) / 60.0
                base_duration = float(leg['duration']['value']) / 60.0
                traffic_tf = max(1.0, round(duration_min / max(1.0, base_duration), 2))
            else:
                duration_min = float(leg['duration']['value']) / 60.0
                traffic_tf = 1.20

            # Decode polyline points
            path_coords = self._decode_polyline(route['overview_polyline']['points'])

            # Toll estimation check
            has_tolls = "toll" in route.get("warnings", []) or "Toll" in route.get("summary", "")
            toll_cost = 5.0 if has_tolls else 0.0

            routes.append({
                "name": f"Route {chr(65+idx)} ({route.get('summary', 'Google Maps Route')})",
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

    def _decode_polyline(self, polyline_str: str) -> List[Tuple[float, float]]:
        """
        Simple polyline decoder converting Encoded Polyline strings to lat/lng tuples.
        """
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
